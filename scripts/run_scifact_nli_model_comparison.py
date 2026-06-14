import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROC_DIR = Path("data/processed/scifact")
PRED_DIR = Path("outputs/predictions/scifact")
METRIC_DIR = Path("outputs/metrics/scifact")
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]

# Candidate open-source NLI models.
# The script will skip any model that cannot be loaded.
CANDIDATE_MODELS = [
    "cross-encoder/nli-deberta-v3-base",
    "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "facebook/bart-large-mnli",
]

TOP_DOCS = 10
MAX_SENTENCES_PER_DOC = 20
MAX_LENGTH = 256
BATCH_SIZE = 32

# To keep this test fast, use full dev by default. Change this if needed.
MAX_DEV_EXAMPLES = None

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

def load_corpus_map():
    corpus = read_jsonl(PROC_DIR / "corpus.jsonl")
    return {str(x["doc_id"]): x for x in corpus}

def infer_mapping(model):
    id2label = model.config.id2label
    mapping = {}
    for idx, label in id2label.items():
        l = str(label).lower()
        if "entail" in l or l == "label_2":
            mapping["entailment"] = int(idx)
        elif "contrad" in l or l == "label_0":
            mapping["contradiction"] = int(idx)
        elif "neutral" in l or l == "label_1":
            mapping["neutral"] = int(idx)

    # Common fallback for MNLI style: contradiction=0, neutral=1, entailment=2.
    if len(id2label) == 3 and ("entailment" not in mapping or "contradiction" not in mapping):
        mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}

    if "entailment" not in mapping or "contradiction" not in mapping:
        raise RuntimeError(f"Cannot infer NLI mapping from id2label={id2label}")

    return mapping

def build_dev_pairs(corpus_map):
    retrieval_path = PRED_DIR / "bm25_top10_dev.jsonl"
    examples = read_jsonl(retrieval_path)
    if MAX_DEV_EXAMPLES:
        examples = examples[:MAX_DEV_EXAMPLES]

    pair_claims = []
    pair_premises = []
    pair_refs = []

    for ex_idx, ex in enumerate(examples):
        claim = ex["claim"]
        candidates = []

        for doc_rank, doc in enumerate(ex["retrieved"][:TOP_DOCS], start=1):
            doc_id = str(doc["doc_id"])
            cdoc = corpus_map.get(doc_id)
            if not cdoc:
                continue

            title = cdoc.get("title", "")
            sentences = cdoc.get("sentences", [])
            if not isinstance(sentences, list):
                sentences = [str(sentences)]

            for sent_idx, sent in enumerate(sentences[:MAX_SENTENCES_PER_DOC]):
                sent = str(sent).strip()
                if not sent:
                    continue

                premise = (title + ". " + sent).strip()
                cand = {
                    "doc_rank": doc_rank,
                    "doc_id": doc_id,
                    "sentence_index": sent_idx,
                    "sentence": sent,
                    "bm25_doc_score": float(doc.get("score", 0.0)),
                }
                cand_idx = len(candidates)
                candidates.append(cand)

                pair_claims.append(claim)
                pair_premises.append(premise)
                pair_refs.append((ex_idx, cand_idx))

        ex["sentence_candidates"] = candidates

    return examples, pair_claims, pair_premises, pair_refs

def score_model(model_name, examples_base, pair_claims, pair_premises, pair_refs, device):
    print("\n" + "=" * 80)
    print("Testing model:", model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
        mapping = infer_mapping(model)
        print("id2label:", model.config.id2label)
        print("mapping:", mapping)
    except Exception as e:
        print("MODEL LOAD FAILED:", repr(e))
        return {
            "model": model_name,
            "status": "load_failed",
            "error": repr(e),
        }, None

    examples = json.loads(json.dumps(examples_base))

    ent, con, neu = [], [], []

    try:
        with torch.no_grad():
            for start in tqdm(range(0, len(pair_claims), BATCH_SIZE), desc=f"NLI {model_name}"):
                batch_claims = pair_claims[start:start+BATCH_SIZE]
                batch_premises = pair_premises[start:start+BATCH_SIZE]

                enc = tokenizer(
                    batch_premises,
                    batch_claims,
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                logits = model(**enc).logits
                probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

                for row in probs:
                    ent.append(float(row[mapping["entailment"]]))
                    con.append(float(row[mapping["contradiction"]]))
                    if "neutral" in mapping:
                        neu.append(float(row[mapping["neutral"]]))
                    else:
                        neu.append(float(max(0.0, 1.0 - row[mapping["entailment"]] - row[mapping["contradiction"]])))

    except Exception as e:
        print("SCORING FAILED:", repr(e))
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "model": model_name,
            "status": "score_failed",
            "error": repr(e),
        }, None

    for (ex_idx, cand_idx), e, c, n in zip(pair_refs, ent, con, neu):
        examples[ex_idx]["sentence_candidates"][cand_idx]["entailment"] = e
        examples[ex_idx]["sentence_candidates"][cand_idx]["contradiction"] = c
        examples[ex_idx]["sentence_candidates"][cand_idx]["neutral"] = n

    rows = []
    for ex in examples:
        cands = ex["sentence_candidates"]
        if cands:
            S = max(float(c["entailment"]) for c in cands)
            K = max(float(c["contradiction"]) for c in cands)
            N = max(float(c["neutral"]) for c in cands)
        else:
            S, K, N = 0.0, 0.0, 1.0

        rows.append({
            "id": ex["id"],
            "claim": ex["claim"],
            "label": ex["label"],
            "S": S,
            "K": K,
            "N": N,
            "M_SK": S - K,
            "M_KS": K - S,
        })

    metrics = evaluate_simple(rows, model_name)
    metrics["status"] = "success"

    safe_model_name = model_name.replace("/", "__")
    out_path = PRED_DIR / f"nli_model_compare_{safe_model_name}_dev_scores.jsonl"
    write_jsonl(out_path, rows)
    metrics["scores_path"] = str(out_path)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics, rows

def pred_support_only(r, tau_s):
    return "verified" if r["S"] >= tau_s else "unsupported"

def pred_margin(r, tau_s, tau_k, margin):
    S, K = r["S"], r["K"]
    if K >= tau_k and (K - S) >= margin:
        return "refuted"
    if S >= tau_s and (S - K) >= margin:
        return "verified"
    return "unsupported"

def evaluate_predictions(rows, preds):
    y_true = [r["label"] for r in rows]
    y_pred = preds

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    pred_verified = [i for i, p in enumerate(y_pred) if p == "verified"]
    fvr = sum(1 for i in pred_verified if y_true[i] != "verified") / len(pred_verified) if pred_verified else 0.0

    pred_refuted = [i for i, p in enumerate(y_pred) if p == "refuted"]
    frr = sum(1 for i in pred_refuted if y_true[i] != "refuted") / len(pred_refuted) if pred_refuted else 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "verified_f1": float(per[0]),
        "refuted_f1": float(per[1]),
        "unsupported_f1": float(per[2]),
        "num_predicted_verified": int(len(pred_verified)),
        "false_verification_rate": float(fvr),
        "num_predicted_refuted": int(len(pred_refuted)),
        "false_refuted_rate": float(frr),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }

def evaluate_simple(rows, model_name):
    vals = np.round(np.arange(0.05, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.0, 0.81, 0.05), 2)

    best_support = None
    for tau_s in vals:
        preds = [pred_support_only(r, tau_s) for r in rows]
        m = evaluate_predictions(rows, preds)
        obj = m["macro_f1"] - 0.25 * m["false_verification_rate"]
        m.update({"rule": "support_only", "tau_s": float(tau_s), "objective": float(obj)})
        if best_support is None or obj > best_support["objective"]:
            best_support = m

    best_margin = None
    for tau_s in vals:
        for tau_k in vals:
            for margin in margins:
                preds = [pred_margin(r, tau_s, tau_k, margin) for r in rows]
                m = evaluate_predictions(rows, preds)
                obj = m["macro_f1"] - 0.50 * m["false_verification_rate"] - 0.20 * m["false_refuted_rate"]
                m.update({
                    "rule": "margin_support_refute",
                    "tau_s": float(tau_s),
                    "tau_k": float(tau_k),
                    "margin": float(margin),
                    "objective": float(obj),
                })
                if best_margin is None or obj > best_margin["objective"]:
                    best_margin = m

    out = {
        "model": model_name,
        "best_support_only": best_support,
        "best_margin_rule": best_margin,
    }

    # Flatten best metrics for table.
    for prefix, m in [("support", best_support), ("margin", best_margin)]:
        for k, v in m.items():
            if not isinstance(v, list):
                out[f"{prefix}_{k}"] = v

    print("Best support-only:")
    print(json.dumps(best_support, indent=2))
    print("Best margin rule:")
    print(json.dumps(best_margin, indent=2))

    return out

def main():
    print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda devices:", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    corpus_map = load_corpus_map()
    examples, pair_claims, pair_premises, pair_refs = build_dev_pairs(corpus_map)

    print("Dev examples:", len(examples))
    print("Sentence pairs:", len(pair_claims))

    all_metrics = []

    for model_name in CANDIDATE_MODELS:
        metrics, _ = score_model(model_name, examples, pair_claims, pair_premises, pair_refs, device)
        all_metrics.append(metrics)

        metric_path = METRIC_DIR / f"nli_model_compare_{model_name.replace('/', '__')}_metrics.json"
        with metric_path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)

    flat_rows = []
    for m in all_metrics:
        row = {}
        for k, v in m.items():
            if not isinstance(v, (dict, list)):
                row[k] = v
        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    summary_path = METRIC_DIR / "nli_model_comparison_summary.csv"
    df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print("NLI model comparison summary:")
    print(df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")

if __name__ == "__main__":
    main()
