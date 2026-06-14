import json
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PRED_DIR = Path("outputs/predictions/scifact")
METRIC_DIR = Path("outputs/metrics/scifact")
METRIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
TOPK = 10
BATCH_SIZE = 16
MAX_LENGTH = 512

# Initial non-calibrated thresholds.
TAU_S = 0.50
TAU_K = 0.50

LABELS = ["verified", "refuted", "unsupported"]

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def infer_nli_label_mapping(model):
    # For this model, expected labels usually include contradiction, entailment, neutral.
    id2label = model.config.id2label
    print("Model id2label:", id2label)

    mapping = {}
    for idx, label in id2label.items():
        l = str(label).lower()
        if "entail" in l:
            mapping["entailment"] = int(idx)
        elif "contrad" in l:
            mapping["contradiction"] = int(idx)
        elif "neutral" in l:
            mapping["neutral"] = int(idx)

    if "entailment" not in mapping or "contradiction" not in mapping:
        raise RuntimeError(f"Could not infer entailment/contradiction mapping from id2label={id2label}")

    print("Inferred mapping:", mapping)
    return mapping

def score_pairs(claims, evidences, tokenizer, model, device):
    entail_scores = []
    contrad_scores = []
    neutral_scores = []

    mapping = infer_nli_label_mapping(model)

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(claims), BATCH_SIZE), desc="NLI batches"):
            batch_claims = claims[start:start+BATCH_SIZE]
            batch_evidences = evidences[start:start+BATCH_SIZE]

            encoded = tokenizer(
                batch_evidences,
                batch_claims,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )

            encoded = {k: v.to(device) for k, v in encoded.items()}
            logits = model(**encoded).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for row in probs:
                entail_scores.append(float(row[mapping["entailment"]]))
                contrad_scores.append(float(row[mapping["contradiction"]]))
                if "neutral" in mapping:
                    neutral_scores.append(float(row[mapping["neutral"]]))
                else:
                    neutral_scores.append(float(1.0 - row[mapping["entailment"]] - row[mapping["contradiction"]]))

    return entail_scores, contrad_scores, neutral_scores

def predict_label(S, K, P, tau_s=TAU_S, tau_k=TAU_K):
    # Minimal non-calibrated version of our rule:
    # verified if support passes, contradiction does not pass, provenance exists.
    if K >= tau_k:
        return "refuted"
    if S >= tau_s and P == 1:
        return "verified"
    return "unsupported"

def evaluate(rows, split):
    y_true = [r["label"] for r in rows]
    y_pred = [r["prediction"] for r in rows]

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per_label = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    metrics = {
        "dataset": "scifact",
        "split": split,
        "method": "bm25_top10_nli_deberta_v3_base_threshold",
        "model": MODEL_NAME,
        "topk": TOPK,
        "tau_s": TAU_S,
        "tau_k": TAU_K,
        "num_examples": len(rows),
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "verified_f1": float(per_label[0]),
        "refuted_f1": float(per_label[1]),
        "unsupported_f1": float(per_label[2]),
    }

    # False verification rate:
    # among examples predicted verified, how many are not truly verified?
    pred_verified = [r for r in rows if r["prediction"] == "verified"]
    if pred_verified:
        false_verified = [r for r in pred_verified if r["label"] != "verified"]
        metrics["num_predicted_verified"] = len(pred_verified)
        metrics["false_verification_rate"] = len(false_verified) / len(pred_verified)
    else:
        metrics["num_predicted_verified"] = 0
        metrics["false_verification_rate"] = None

    report = classification_report(
        y_true,
        y_pred,
        labels=LABELS,
        zero_division=0,
        output_dict=True,
    )

    metrics["classification_report"] = report
    metrics["confusion_matrix_labels"] = LABELS
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()

    return metrics

def run_split(split, tokenizer, model, device):
    in_path = PRED_DIR / f"bm25_top{TOPK}_{split}.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Missing retrieval file: {in_path}")

    examples = read_jsonl(in_path)
    print(f"Loaded {split}: {len(examples)} examples from {in_path}")

    pair_claims = []
    pair_evidences = []
    pair_refs = []

    for ex_idx, ex in enumerate(examples):
        claim = ex["claim"]
        retrieved = ex["retrieved"][:TOPK]
        for rank_idx, ev in enumerate(retrieved):
            evidence_text = (ev.get("title", "") + ". " + ev.get("text", "")).strip()
            pair_claims.append(claim)
            pair_evidences.append(evidence_text)
            pair_refs.append((ex_idx, rank_idx))

    print(f"Total claim-evidence pairs for {split}: {len(pair_claims)}")

    ent, con, neu = score_pairs(pair_claims, pair_evidences, tokenizer, model, device)

    for (ex_idx, rank_idx), e, c, n in zip(pair_refs, ent, con, neu):
        examples[ex_idx]["retrieved"][rank_idx]["nli_entailment"] = e
        examples[ex_idx]["retrieved"][rank_idx]["nli_contradiction"] = c
        examples[ex_idx]["retrieved"][rank_idx]["nli_neutral"] = n

    output_rows = []

    for ex in examples:
        retrieved = ex["retrieved"][:TOPK]
        if retrieved:
            S = max(float(r.get("nli_entailment", 0.0)) for r in retrieved)
            K = max(float(r.get("nli_contradiction", 0.0)) for r in retrieved)
            best_support = max(retrieved, key=lambda r: float(r.get("nli_entailment", 0.0)))
            best_contradiction = max(retrieved, key=lambda r: float(r.get("nli_contradiction", 0.0)))
            P = 1
        else:
            S, K, P = 0.0, 0.0, 0
            best_support = None
            best_contradiction = None

        pred = predict_label(S, K, P)

        ex_out = {
            "id": ex["id"],
            "dataset": "scifact",
            "split": split,
            "claim": ex["claim"],
            "label": ex["label"],
            "prediction": pred,
            "S_support": S,
            "K_contradiction": K,
            "P_provenance": P,
            "tau_s": TAU_S,
            "tau_k": TAU_K,
            "gold_doc_ids": ex.get("gold_doc_ids", []),
            "best_support": best_support,
            "best_contradiction": best_contradiction,
            "retrieved": retrieved,
        }
        output_rows.append(ex_out)

    out_path = PRED_DIR / f"bm25_top{TOPK}_nli_{split}.jsonl"
    write_jsonl(out_path, output_rows)

    metrics = evaluate(output_rows, split)
    metric_path = METRIC_DIR / f"bm25_top{TOPK}_nli_{split}_metrics.json"
    with metric_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nSaved predictions: {out_path}")
    print(f"Saved metrics: {metric_path}")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ['classification_report', 'confusion_matrix']}, indent=2))

    print("\nConfusion matrix labels:", LABELS)
    print(np.array(metrics["confusion_matrix"]))

    return metrics

def main():
    print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda devices:", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    metrics = []
    for split in ["train", "dev"]:
        metrics.append(run_split(split, tokenizer, model, device))

    summary_path = METRIC_DIR / "bm25_top10_nli_summary.csv"
    flat = []
    for m in metrics:
        flat.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
    pd.DataFrame(flat).to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV: {summary_path}")

if __name__ == "__main__":
    main()
