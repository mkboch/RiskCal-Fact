import os
import re
import gc
import json
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROC_DIR = Path("data/processed/fever")
OUT_DIR = Path("outputs/predictions/fever")
METRIC_DIR = Path("outputs/metrics/fever")
TABLE_DIR = Path("outputs/tables/fever")
FINAL_DIR = Path("outputs/final_report")

for d in [OUT_DIR, METRIC_DIR, TABLE_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABELS = ["verified", "refuted", "unsupported"]

# Use already-created full-dev splits and 1.2M sentence corpus.
TUNE_PATH = PROC_DIR / "full_dev_tune_claims.jsonl"
CAL_PATH = PROC_DIR / "full_dev_cal_claims.jsonl"
DEV_PATH = PROC_DIR / "full_dev_paper_dev_claims.jsonl"
CORPUS_PATH = PROC_DIR / "full_dev_sentence_corpus.jsonl"

TOPK = 10
BATCH_QUERY = 256
BATCH_NLI = 32
MAX_LENGTH = 256

MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "facebook/bart-large-mnli",
]

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def gold_sentence_ids(row):
    ids = set()
    for ev in row.get("gold_evidence", []):
        wiki = str(ev.get("wiki_url", "")).replace(" ", "_")
        try:
            sid = int(ev.get("sentence_id", -1))
        except Exception:
            sid = -1
        if wiki and sid >= 0:
            ids.add(f"{wiki}::{sid}")
    return ids

def ensure_splits_exist():
    if TUNE_PATH.exists() and CAL_PATH.exists() and DEV_PATH.exists():
        print("Using existing full-dev tune/cal/dev claim files.")
        return

    raise FileNotFoundError(
        "Missing full-dev tune/cal/dev files. The previous pipeline must at least finish split creation."
    )

def load_corpus():
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(f"Missing corpus: {CORPUS_PATH}")

    corpus = read_jsonl(CORPUS_PATH)
    print("Loaded corpus sentences:", len(corpus))
    return corpus

def tfidf_retrieve(split_name, claims, corpus):
    out_path = OUT_DIR / f"fast_full_dev_tfidf_top{TOPK}_{split_name}.jsonl"
    metric_path = METRIC_DIR / f"fast_full_dev_tfidf_top{TOPK}_{split_name}_metrics.json"

    if out_path.exists() and metric_path.exists():
        print("Loading cached TF-IDF retrieval:", out_path)
        return read_jsonl(out_path)

    print(f"Building TF-IDF vectorizer for split={split_name} over corpus size={len(corpus)}")
    corpus_texts = [(c["wiki_url"] + " " + c["text"]) for c in corpus]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=700000,
        dtype=np.float32,
        norm="l2",
    )

    X = vectorizer.fit_transform(corpus_texts)
    print("TF-IDF matrix:", X.shape, "nnz:", X.nnz)

    outputs = []
    denom = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}

    for start in tqdm(range(0, len(claims), BATCH_QUERY), desc=f"TF-IDF retrieve {split_name}"):
        batch = claims[start:start+BATCH_QUERY]
        q = vectorizer.transform([r["claim"] for r in batch])
        scores = q @ X.T

        for bi, r in enumerate(batch):
            row = scores.getrow(bi)
            if row.nnz == 0:
                top_idx = np.arange(min(TOPK, len(corpus)))
                top_scores = np.zeros(len(top_idx), dtype=np.float32)
            else:
                data = row.data
                idx = row.indices
                if len(data) > TOPK:
                    part = np.argpartition(data, -TOPK)[-TOPK:]
                    order = part[np.argsort(data[part])[::-1]]
                else:
                    order = np.argsort(data)[::-1]
                top_idx = idx[order]
                top_scores = data[order]

            retrieved = []
            for rank, (ci, score) in enumerate(zip(top_idx, top_scores), 1):
                item = corpus[int(ci)]
                retrieved.append({
                    "rank": rank,
                    "score": float(score),
                    "sent_id": item["sent_id"],
                    "wiki_url": item["wiki_url"],
                    "sentence_id": item["sentence_id"],
                    "text": item["text"],
                })

            g = gold_sentence_ids(r)
            if g:
                denom += 1
                pred_ids = [x["sent_id"] for x in retrieved]
                for k in hits:
                    if any(x in g for x in pred_ids[:k]):
                        hits[k] += 1

            rr = dict(r)
            rr["gold_sentence_ids"] = sorted(g)
            rr["retrieved"] = retrieved
            outputs.append(rr)

    write_jsonl(out_path, outputs)

    metrics = {
        "dataset": "fever",
        "split": split_name,
        "method": "tfidf_sentence_fast_full_dev",
        "topk": TOPK,
        "num_examples": len(claims),
        "num_with_gold": denom,
    }
    for k in hits:
        metrics[f"evidence_sentence_recall@{k}"] = hits[k] / denom if denom else None

    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Retrieval metrics:", json.dumps(metrics, indent=2))
    return outputs

def infer_mapping(model):
    id2label = model.config.id2label
    mapping = {}
    for idx, label in id2label.items():
        l = str(label).lower()
        if "entail" in l:
            mapping["entailment"] = int(idx)
        elif "contrad" in l:
            mapping["contradiction"] = int(idx)
        elif "neutral" in l:
            mapping["neutral"] = int(idx)
    if len(id2label) == 3 and ("entailment" not in mapping or "contradiction" not in mapping):
        mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}
    return mapping

def score_nli(model_name, split_name, rows, device):
    safe = model_name.replace("/", "__")
    out_path = OUT_DIR / f"fast_full_dev_nli_{safe}_{split_name}_scores.jsonl"

    if out_path.exists():
        print("Loading cached NLI:", out_path)
        return read_jsonl(out_path)

    print("Loading NLI model:", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    mapping = infer_mapping(model)
    print("mapping:", mapping)

    pair_claims, pair_premises, refs = [], [], []
    for i, r in enumerate(rows):
        for j, ev in enumerate(r["retrieved"][:TOPK]):
            pair_claims.append(r["claim"])
            pair_premises.append(ev["wiki_url"] + ". " + ev["text"])
            refs.append((i, j))

    ent, con, neu = [], [], []
    with torch.no_grad():
        for start in tqdm(range(0, len(pair_claims), BATCH_NLI), desc=f"NLI {split_name} {model_name}"):
            bc = pair_claims[start:start+BATCH_NLI]
            bp = pair_premises[start:start+BATCH_NLI]
            enc = tok(bp, bc, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).detach().cpu().numpy()
            for row in probs:
                ent.append(float(row[mapping["entailment"]]))
                con.append(float(row[mapping["contradiction"]]))
                neu.append(float(row[mapping["neutral"]]))

    full = json.loads(json.dumps(rows))
    for (i, j), e, c, n in zip(refs, ent, con, neu):
        full[i]["retrieved"][j]["entailment"] = e
        full[i]["retrieved"][j]["contradiction"] = c
        full[i]["retrieved"][j]["neutral"] = n

    outputs = []
    for r in full:
        cands = r["retrieved"]
        S = max([float(x.get("entailment", 0.0)) for x in cands], default=0.0)
        K = max([float(x.get("contradiction", 0.0)) for x in cands], default=0.0)
        N = max([float(x.get("neutral", 0.0)) for x in cands], default=1.0)

        outputs.append({
            "id": r["id"],
            "dataset": "fever",
            "split": split_name,
            "claim": r["claim"],
            "label": r["label"],
            "model": model_name,
            "S": S,
            "K": K,
            "N": N,
            "P": 1 if cands else 0,
            "gold_sentence_ids": r.get("gold_sentence_ids", []),
            "retrieved": cands,
        })

    write_jsonl(out_path, outputs)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return outputs

def predict(rule, r, params):
    S, K, P = float(r["S"]), float(r["K"]), int(r["P"])
    tau_s = float(params.get("tau_s", 0.5))
    tau_k = float(params.get("tau_k", 0.5))
    margin = float(params.get("margin", 0.0))

    if rule == "support_only":
        return "verified" if S >= tau_s and P == 1 else "unsupported"

    if rule == "margin_support_refute":
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        if S >= tau_s and (S - K) >= margin and P == 1:
            return "verified"
        return "unsupported"

    raise ValueError(rule)

def verified_conf(rule, r):
    return float(r["S"]) if rule == "support_only" else float(r["S"]) - float(r["K"])

def eval_preds(rows, preds):
    y_true = [r["label"] for r in rows]
    y_pred = preds

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    pred_verified = [i for i, p in enumerate(y_pred) if p == "verified"]
    fvr = sum(1 for i in pred_verified if y_true[i] != "verified") / len(pred_verified) if pred_verified else 0.0

    pred_refuted = [i for i, p in enumerate(y_pred) if p == "refuted"]
    frr = sum(1 for i in pred_refuted if y_true[i] != "refuted") / len(pred_refuted) if pred_refuted else 0.0

    abstained = [i for i, p in enumerate(y_pred) if p == "abstain"]
    accepted = [i for i, p in enumerate(y_pred) if p != "abstain"]
    accepted_acc = accuracy_score([y_true[i] for i in accepted], [y_pred[i] for i in accepted]) if accepted else 0.0

    return {
        "accuracy_all": float(acc),
        "macro_f1_all": float(macro),
        "verified_f1": float(per[0]),
        "refuted_f1": float(per[1]),
        "unsupported_f1": float(per[2]),
        "num_predicted_verified": len(pred_verified),
        "false_verification_rate": float(fvr),
        "num_predicted_refuted": len(pred_refuted),
        "false_refuted_rate": float(frr),
        "num_abstained": len(abstained),
        "coverage": float(len(accepted) / len(rows)),
        "accepted_accuracy": float(accepted_acc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS + ["abstain"]).tolist(),
        "confusion_matrix_labels": LABELS + ["abstain"],
    }

def tune_params(rows, rule):
    vals = np.round(np.arange(0.05, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.0, 0.81, 0.05), 2)

    if rule == "support_only":
        grid = [{"tau_s": ts} for ts in vals]
    else:
        grid = [{"tau_s": ts, "tau_k": tk, "margin": mg} for ts in vals for tk in vals for mg in margins]

    best = None
    records = []

    for params in grid:
        preds = [predict(rule, r, params) for r in rows]
        m = eval_preds(rows, preds)
        obj = m["macro_f1_all"] + 0.1 * m["accepted_accuracy"] - 0.5 * m["false_verification_rate"] - 0.2 * m["false_refuted_rate"]
        m["objective"] = float(obj)
        m.update({f"param_{k}": float(v) for k, v in params.items()})
        records.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})
        if best is None or obj > best["objective"]:
            best = m

    params = {k.replace("param_", ""): float(v) for k, v in best.items() if k.startswith("param_")}
    return params, best, records

def choose_threshold(cal_rows, cal_preds, cal_confs, alpha):
    idxs = [i for i, p in enumerate(cal_preds) if p == "verified"]

    if not idxs:
        return float("inf"), {
            "cal_base_verified": 0,
            "cal_retained_verified": 0,
            "cal_retained_fvr": 0.0,
            "cal_verified_retention": 0.0,
        }

    thresholds = sorted(set(cal_confs[i] for i in idxs))

    for t in thresholds:
        retained = [i for i in idxs if cal_confs[i] >= t]
        if not retained:
            continue
        fvr = sum(1 for i in retained if cal_rows[i]["label"] != "verified") / len(retained)
        if fvr <= alpha:
            return float(t), {
                "cal_base_verified": len(idxs),
                "cal_retained_verified": len(retained),
                "cal_retained_fvr": float(fvr),
                "cal_verified_retention": float(len(retained) / len(idxs)),
            }

    return float(max(thresholds) + 1e-6), {
        "cal_base_verified": len(idxs),
        "cal_retained_verified": 0,
        "cal_retained_fvr": 0.0,
        "cal_verified_retention": 0.0,
    }

def apply_gate(preds, confs, threshold):
    return ["abstain" if p == "verified" and c < threshold else p for p, c in zip(preds, confs)]

def run_calibration(model_name, tune_rows, cal_rows, dev_rows):
    summary = []
    grid_records = []

    for rule in ["support_only", "margin_support_refute"]:
        print("Calibrating:", model_name, rule)
        params, best, grid = tune_params(tune_rows, rule)

        for g in grid:
            g["model"] = model_name
            g["rule"] = rule
        grid_records.extend(grid)

        cal_preds = [predict(rule, r, params) for r in cal_rows]
        dev_preds = [predict(rule, r, params) for r in dev_rows]
        cal_confs = [verified_conf(rule, r) for r in cal_rows]
        dev_confs = [verified_conf(rule, r) for r in dev_rows]

        base_dev = eval_preds(dev_rows, dev_preds)

        for alpha in ALPHAS:
            thr, cal_info = choose_threshold(cal_rows, cal_preds, cal_confs, alpha)
            risk_preds = apply_gate(dev_preds, dev_confs, thr)
            m = eval_preds(dev_rows, risk_preds)
            row = {
                "model": model_name,
                "rule": rule,
                "alpha": float(alpha),
                "threshold": float(thr),
                **{f"param_{k}": float(v) for k, v in params.items()},
                **cal_info,
                "base_dev_macro_f1": base_dev["macro_f1_all"],
                "base_dev_false_verification_rate": base_dev["false_verification_rate"],
                **m,
            }
            summary.append(row)

    return summary, grid_records

def main():
    ensure_splits_exist()

    tune_claims = read_jsonl(TUNE_PATH)
    cal_claims = read_jsonl(CAL_PATH)
    dev_claims = read_jsonl(DEV_PATH)

    print("Tune claims:", len(tune_claims), Counter(r["label"] for r in tune_claims))
    print("Cal claims:", len(cal_claims), Counter(r["label"] for r in cal_claims))
    print("Dev claims:", len(dev_claims), Counter(r["label"] for r in dev_claims))

    corpus = load_corpus()

    tune_ret = tfidf_retrieve("tune", tune_claims, corpus)
    cal_ret = tfidf_retrieve("cal", cal_claims, corpus)
    dev_ret = tfidf_retrieve("paper_dev_full", dev_claims, corpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    all_summary = []
    all_grid = []

    for model_name in MODELS:
        tune_scores = score_nli(model_name, "tune", tune_ret, device)
        cal_scores = score_nli(model_name, "cal", cal_ret, device)
        dev_scores = score_nli(model_name, "paper_dev_full", dev_ret, device)

        summary, grid = run_calibration(model_name, tune_scores, cal_scores, dev_scores)
        all_summary.extend(summary)
        all_grid.extend(grid)

    summary_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in all_summary])
    summary_path = METRIC_DIR / "fever_fast_full_dev_risk_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_path = METRIC_DIR / "fever_fast_full_dev_tune_grid.csv"
    pd.DataFrame(all_grid).to_csv(grid_path, index=False)

    keep = [
        "model", "rule", "alpha", "macro_f1_all", "accuracy_all",
        "false_verification_rate", "false_refuted_rate",
        "num_predicted_verified", "num_predicted_refuted",
        "num_abstained", "coverage", "accepted_accuracy",
        "cal_retained_fvr", "cal_verified_retention",
        "base_dev_macro_f1", "base_dev_false_verification_rate",
    ]
    table = summary_df[[c for c in keep if c in summary_df.columns]].copy()
    table_path = TABLE_DIR / "table_fever_fast_full_dev_risk_calibration.csv"
    table.to_csv(table_path, index=False)

    status = {
        "dataset": "FEVER",
        "evaluation": "full paper_dev claims",
        "retrieval": "TF-IDF sentence retrieval over full_dev_sentence_corpus",
        "corpus_sentences": len(corpus),
        "tune_rows": len(tune_claims),
        "cal_rows": len(cal_claims),
        "dev_rows": len(dev_claims),
        "summary": str(summary_path),
        "table": str(table_path),
    }
    (FINAL_DIR / "fever_fast_full_dev_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("\n==== FEVER FAST FULL-DEV SUMMARY ====")
    print(table.to_string(index=False))
    print("\nSaved:", summary_path)
    print("Saved:", table_path)
    print("Saved:", grid_path)

if __name__ == "__main__":
    main()
