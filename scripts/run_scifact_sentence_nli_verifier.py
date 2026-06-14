import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROC_DIR = Path("data/processed/scifact")
PRED_DIR = Path("outputs/predictions/scifact")
METRIC_DIR = Path("outputs/metrics/scifact")
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "cross-encoder/nli-deberta-v3-base"
TOP_DOCS = 10
MAX_SENTENCES_PER_DOC = 20
BATCH_SIZE = 32
MAX_LENGTH = 256

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

def load_corpus_map():
    corpus = read_jsonl(PROC_DIR / "corpus.jsonl")
    return {str(x["doc_id"]): x for x in corpus}

def infer_nli_label_mapping(model):
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
        raise RuntimeError(f"Could not infer mapping from id2label={id2label}")
    print("Inferred mapping:", mapping)
    return mapping

def score_pairs(claims, premises, tokenizer, model, device):
    mapping = infer_nli_label_mapping(model)
    ent, con, neu = [], [], []

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(claims), BATCH_SIZE), desc="Sentence NLI batches"):
            batch_claims = claims[start:start+BATCH_SIZE]
            batch_premises = premises[start:start+BATCH_SIZE]

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
                    neu.append(float(1.0 - row[mapping["entailment"]] - row[mapping["contradiction"]]))

    return ent, con, neu

def expand_sentence_candidates(examples, corpus_map):
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
                candidate = {
                    "doc_rank": doc_rank,
                    "doc_id": doc_id,
                    "title": title,
                    "sentence_index": sent_idx,
                    "sentence": sent,
                    "bm25_doc_score": float(doc.get("score", 0.0)),
                }
                cand_idx = len(candidates)
                candidates.append(candidate)
                pair_claims.append(claim)
                pair_premises.append(premise)
                pair_refs.append((ex_idx, cand_idx))

        ex["sentence_candidates"] = candidates

    return pair_claims, pair_premises, pair_refs

def gold_sentence_hit_at_k(ex, candidates, k):
    gold_pairs = set()
    for ev in ex.get("gold_evidence", []):
        doc_id = str(ev.get("doc_id"))
        for sid in ev.get("sentences", []):
            try:
                gold_pairs.add((doc_id, int(sid)))
            except Exception:
                pass

    if not gold_pairs:
        return None

    pred_pairs = []
    for c in candidates[:k]:
        pred_pairs.append((str(c["doc_id"]), int(c["sentence_index"])))

    return any(p in gold_pairs for p in pred_pairs)

def build_rows_with_scores(examples):
    output = []

    for ex in examples:
        cands = ex.get("sentence_candidates", [])

        if cands:
            best_support = max(cands, key=lambda x: float(x.get("nli_entailment", 0.0)))
            best_contradiction = max(cands, key=lambda x: float(x.get("nli_contradiction", 0.0)))
            S = float(best_support.get("nli_entailment", 0.0))
            K = float(best_contradiction.get("nli_contradiction", 0.0))
            P = 1
        else:
            best_support = None
            best_contradiction = None
            S = 0.0
            K = 0.0
            P = 0

        # Sort candidates by evidence usefulness for inspection.
        sorted_cands = sorted(
            cands,
            key=lambda x: max(float(x.get("nli_entailment", 0.0)), float(x.get("nli_contradiction", 0.0))),
            reverse=True,
        )

        output.append({
            "id": ex["id"],
            "dataset": "scifact",
            "split": ex["split"],
            "claim": ex["claim"],
            "label": ex["label"],
            "S_support": S,
            "K_contradiction": K,
            "P_provenance": P,
            "gold_doc_ids": ex.get("gold_doc_ids", []),
            "best_support": best_support,
            "best_contradiction": best_contradiction,
            "sentence_candidates": sorted_cands[:50],
        })

    return output

def predict_label(S, K, P, tau_s, tau_k, mode):
    if mode == "support_only":
        if S >= tau_s and P == 1:
            return "verified"
        return "unsupported"

    if mode in ["support_contradiction", "ours_no_abstain"]:
        if K >= tau_k:
            return "refuted"
        if S >= tau_s and K < tau_k and P == 1:
            return "verified"
        return "unsupported"

    raise ValueError(mode)

def evaluate(rows, tau_s, tau_k, mode):
    y_true = []
    y_pred = []
    out = []

    for r in rows:
        pred = predict_label(
            float(r["S_support"]),
            float(r["K_contradiction"]),
            int(r["P_provenance"]),
            tau_s,
            tau_k,
            mode,
        )
        rr = dict(r)
        rr["prediction"] = pred
        rr["tau_s"] = float(tau_s)
        rr["tau_k"] = float(tau_k)
        rr["mode"] = mode
        out.append(rr)
        y_true.append(r["label"])
        y_pred.append(pred)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    pred_verified = [i for i, p in enumerate(y_pred) if p == "verified"]
    fvr = (
        sum(1 for i in pred_verified if y_true[i] != "verified") / len(pred_verified)
        if pred_verified else 0.0
    )

    pred_refuted = [i for i, p in enumerate(y_pred) if p == "refuted"]
    frr = (
        sum(1 for i in pred_refuted if y_true[i] != "refuted") / len(pred_refuted)
        if pred_refuted else 0.0
    )

    metrics = {
        "mode": mode,
        "tau_s": float(tau_s),
        "tau_k": float(tau_k),
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
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }
    return metrics, out

def calibrate(train_rows, mode):
    tau_s_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tau_k_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    if mode == "support_only":
        grid = [(ts, 1.0) for ts in tau_s_grid]
    else:
        grid = [(ts, tk) for ts in tau_s_grid for tk in tau_k_grid]

    best = None
    records = []

    for tau_s, tau_k in grid:
        m, _ = evaluate(train_rows, tau_s, tau_k, mode)

        # Reliability-oriented objective.
        objective = (
            m["macro_f1"]
            - 0.25 * m["false_verification_rate"]
            - 0.10 * m["false_refuted_rate"]
        )
        m["objective"] = float(objective)
        records.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})

        if best is None or m["objective"] > best["objective"]:
            best = m

    return best, records

def score_split(split, tokenizer, model, device, corpus_map):
    retrieval_path = PRED_DIR / f"bm25_top10_{split}.jsonl"
    cache_path = PRED_DIR / f"bm25_top10_sentence_nli_scores_{split}.jsonl"

    if cache_path.exists():
        print(f"Loading cached sentence NLI scores: {cache_path}")
        return read_jsonl(cache_path)

    examples = read_jsonl(retrieval_path)
    print(f"Loaded {split} retrieval examples: {len(examples)}")

    claims, premises, refs = expand_sentence_candidates(examples, corpus_map)
    print(f"Total sentence-level NLI pairs for {split}: {len(claims)}")

    ent, con, neu = score_pairs(claims, premises, tokenizer, model, device)

    for (ex_idx, cand_idx), e, c, n in zip(refs, ent, con, neu):
        examples[ex_idx]["sentence_candidates"][cand_idx]["nli_entailment"] = e
        examples[ex_idx]["sentence_candidates"][cand_idx]["nli_contradiction"] = c
        examples[ex_idx]["sentence_candidates"][cand_idx]["nli_neutral"] = n

    rows = build_rows_with_scores(examples)
    write_jsonl(cache_path, rows)
    print(f"Saved sentence scores: {cache_path}")

    return rows

def evidence_sentence_recall(rows):
    # Evaluate whether the top ranked sentence by max NLI score hits gold evidence.
    ks = [1, 3, 5, 10]
    counts = {k: 0 for k in ks}
    denom = 0

    for r in rows:
        gold_pairs = set()
        # gold sentence info is not directly in rows after retrieval file,
        # so sentence recall will be added later if needed from processed files.
        # Here we skip because train/dev NLI rows do not carry full gold evidence.
        pass

    return {}

def compact(m):
    keys = [
        "mode", "tau_s", "tau_k", "accuracy", "macro_f1",
        "verified_f1", "refuted_f1", "unsupported_f1",
        "num_predicted_verified", "false_verification_rate",
        "num_predicted_refuted", "false_refuted_rate", "objective"
    ]
    return {k: m.get(k) for k in keys}

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
    print("Loaded corpus docs:", len(corpus_map))

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)

    train_rows = score_split("train", tokenizer, model, device, corpus_map)
    dev_rows = score_split("dev", tokenizer, model, device, corpus_map)

    modes = ["support_only", "support_contradiction", "ours_no_abstain"]
    summary = []
    all_grid = []

    for mode in modes:
        print("\n" + "=" * 80)
        print("Calibrating:", mode)
        best, grid = calibrate(train_rows, mode)
        all_grid.extend([{**x, "mode": mode} for x in grid])

        print("Best train:")
        print(json.dumps(compact(best), indent=2))

        dev_metrics, dev_out = evaluate(dev_rows, best["tau_s"], best["tau_k"], mode)
        dev_metrics["selected_train_objective"] = best["objective"]
        dev_metrics["selected_train_macro_f1"] = best["macro_f1"]
        dev_metrics["selected_train_false_verification_rate"] = best["false_verification_rate"]

        print("Dev result:")
        print(json.dumps(compact(dev_metrics), indent=2))
        print("Dev confusion matrix labels:", LABELS)
        print(np.array(dev_metrics["confusion_matrix"]))

        pred_path = PRED_DIR / f"sentence_nli_calibrated_{mode}_dev.jsonl"
        metric_path = METRIC_DIR / f"sentence_nli_calibrated_{mode}_dev_metrics.json"

        write_jsonl(pred_path, dev_out)
        with metric_path.open("w", encoding="utf-8") as f:
            json.dump(dev_metrics, f, indent=2, ensure_ascii=False)

        summary.append({
            "mode": mode,
            "selected_tau_s": best["tau_s"],
            "selected_tau_k": best["tau_k"],
            "train_objective": best["objective"],
            "train_macro_f1": best["macro_f1"],
            "train_false_verification_rate": best["false_verification_rate"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_verified_f1": dev_metrics["verified_f1"],
            "dev_refuted_f1": dev_metrics["refuted_f1"],
            "dev_unsupported_f1": dev_metrics["unsupported_f1"],
            "dev_num_predicted_verified": dev_metrics["num_predicted_verified"],
            "dev_false_verification_rate": dev_metrics["false_verification_rate"],
            "dev_num_predicted_refuted": dev_metrics["num_predicted_refuted"],
            "dev_false_refuted_rate": dev_metrics["false_refuted_rate"],
        })

    summary_df = pd.DataFrame(summary)
    summary_path = METRIC_DIR / "sentence_nli_threshold_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_path = METRIC_DIR / "sentence_nli_threshold_grid_train.csv"
    pd.DataFrame(all_grid).to_csv(grid_path, index=False)

    print("\n" + "=" * 80)
    print("Final sentence-level NLI calibration summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved grid: {grid_path}")

if __name__ == "__main__":
    main()
