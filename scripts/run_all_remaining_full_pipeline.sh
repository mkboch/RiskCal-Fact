#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/all_remaining_full_pipeline_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== ALL REMAINING FULL PIPELINE START ===="
date
echo "PROJECT_DIR=$PROJECT_DIR"
echo "LOG=$LOG"

echo ""
echo "==== Activate main env ===="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_py310
echo "python=$(command -v python)"
python --version

echo ""
echo "==== Ensure dynamic GPU selector ===="
mkdir -p scripts
cat > scripts/select_free_gpu.sh <<'GPUSEL'
#!/usr/bin/env bash
NUM_GPUS="${1:-1}"
GPU_LIST=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null \
  | awk -F',' '{gsub(/ /,"",$1); gsub(/ /,"",$2); print $1" "$2}' \
  | sort -k2 -nr \
  | head -n "$NUM_GPUS" \
  | awk '{print $1}' \
  | paste -sd, -)
if [ -z "$GPU_LIST" ]; then
  echo "WARNING: No GPU found by nvidia-smi."
else
  export CUDA_VISIBLE_DEVICES="$GPU_LIST"
  echo "Selected CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi
GPUSEL
chmod +x scripts/select_free_gpu.sh
source scripts/select_free_gpu.sh 1

echo ""
echo "==== GPU snapshot ===="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true

echo ""
echo "==== Step 1: Full SciFact audit and rerun missing pieces ===="
python - <<'PY'
from pathlib import Path
import json
import subprocess
import sys

required = {
    "processed": [
        "data/processed/scifact/corpus.jsonl",
        "data/processed/scifact/train.jsonl",
        "data/processed/scifact/dev.jsonl",
        "data/processed/scifact/summary.json",
    ],
    "retrieval": [
        "outputs/metrics/scifact/bm25_top10_train_metrics.json",
        "outputs/metrics/scifact/bm25_top10_dev_metrics.json",
    ],
    "main_scores": [
        "outputs/metrics/scifact/proper_eval_selected_nli_summary.csv",
    ],
    "risk": [
        "outputs/metrics/scifact/proper_split_risk_calibration_summary.csv",
    ],
    "tables": [
        "outputs/final_report/clean_table_scifact_main.csv",
        "outputs/final_report/clean_table_risk_calibration_selected.csv",
    ],
}

missing = []
for group, files in required.items():
    for f in files:
        p = Path(f)
        if not p.exists() or p.stat().st_size == 0:
            missing.append(f)

print("SciFact missing files:")
for m in missing:
    print("  ", m)
print("SciFact full evaluable status:", "READY" if not missing else "MISSING_PARTS")

# Write status.
Path("outputs/final_report").mkdir(parents=True, exist_ok=True)
status = {
    "scifact_full_evaluable": len(missing) == 0,
    "note": "SciFact official train/dev are fully processed. The official test split is unlabeled, so evaluation uses dev.",
    "missing": missing,
}
Path("outputs/final_report/scifact_full_status.json").write_text(json.dumps(status, indent=2))
PY

echo ""
echo "==== If SciFact key scripts exist, rerun them idempotently to refresh missing outputs ===="
if [ -f scripts/run_scifact_bm25_retrieval.py ]; then
    python scripts/run_scifact_bm25_retrieval.py || echo "WARNING: SciFact BM25 rerun failed."
fi

if [ -f scripts/run_scifact_selected_nli_proper_eval.py ]; then
    python scripts/run_scifact_selected_nli_proper_eval.py || echo "WARNING: SciFact selected NLI proper eval rerun failed."
fi

if [ -f scripts/run_scifact_proper_split_risk_calibration.py ]; then
    python scripts/run_scifact_proper_split_risk_calibration.py || echo "WARNING: SciFact proper split risk calibration rerun failed."
fi

if [ -f scripts/make_clean_publication_tables.py ]; then
    python scripts/make_clean_publication_tables.py || echo "WARNING: clean publication tables refresh failed."
fi

echo ""
echo "==== Step 2: Create FEVER-large/full-dev pipeline ===="
cat > scripts/run_fever_full_dev_pipeline.py <<'PY'
import os
import re
import gc
import json
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROC_DIR = Path("data/processed/fever")
WIKI_PATH = Path("data/raw/fever_wiki_pages/wiki_pages.jsonl")
OUT_DIR = Path("outputs/predictions/fever")
METRIC_DIR = Path("outputs/metrics/fever")
TABLE_DIR = Path("outputs/tables/fever")
INDEX_DIR = Path("data/indexes/fever")
FINAL_DIR = Path("outputs/final_report")

for d in [OUT_DIR, METRIC_DIR, TABLE_DIR, INDEX_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABELS = ["verified", "refuted", "unsupported"]

# This uses all FEVER paper_dev claims for evaluation.
# Tune/cal are larger stratified samples from train.
TUNE_PER_LABEL = 5000
CAL_PER_LABEL = 2500
DEV_MODE = "full_paper_dev"

# Corpus construction:
# include all pages needed by tune/cal/full-dev gold evidence + many distractors.
MAX_DISTRACTOR_PAGES = 250000

TOPK = 10
BATCH_SIZE = 32
MAX_LENGTH = 256

MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "facebook/bart-large-mnli",
]

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

_token_re = re.compile(r"[A-Za-z0-9]+")

def tokenize(text):
    return _token_re.findall((text or "").lower())

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_title(t):
    return str(t or "").replace(" ", "_")

def aggregate_claims(path, split_name):
    out_path = PROC_DIR / f"full_{split_name}_agg.jsonl"
    if out_path.exists():
        return read_jsonl(out_path)

    raw = read_jsonl(path)
    by_id = {}

    for r in raw:
        cid = str(r.get("id", ""))
        meta = r.get("metadata", r)
        label = r.get("label", "")
        claim = r.get("claim", "")

        if cid not in by_id:
            by_id[cid] = {
                "id": cid,
                "dataset": "fever",
                "split": split_name,
                "claim": claim,
                "label": label,
                "gold_evidence": [],
                "metadata_rows": 0,
            }

        by_id[cid]["metadata_rows"] += 1

        wiki = meta.get("evidence_wiki_url", "")
        sid = meta.get("evidence_sentence_id", -1)
        try:
            sid = int(sid)
        except Exception:
            sid = -1

        if wiki and sid >= 0:
            ev = {
                "wiki_url": norm_title(wiki),
                "sentence_id": sid,
            }
            if ev not in by_id[cid]["gold_evidence"]:
                by_id[cid]["gold_evidence"].append(ev)

    rows = list(by_id.values())
    write_jsonl(out_path, rows)
    return rows

def stratified_sample(rows, per_label, seed):
    rng = random.Random(seed)
    out = []
    for lab in LABELS:
        items = [r for r in rows if r["label"] == lab]
        rng.shuffle(items)
        out.extend(items[:min(per_label, len(items))])
    rng.shuffle(out)
    return out

def parse_wiki_page_row(row):
    title = norm_title(row.get("id", row.get("page_id", row.get("title", ""))))
    lines = row.get("lines", "")
    text = row.get("text", "")
    sentences = []

    def add_line(line):
        line = str(line)
        if not line.strip():
            return
        parts = line.split("\t")
        try:
            sid = int(parts[0])
            sent = parts[1] if len(parts) > 1 else ""
        except Exception:
            sid = len(sentences)
            sent = line
        sent = sent.strip()
        if sent:
            sentences.append((sid, sent))

    if isinstance(lines, str):
        for line in lines.split("\n"):
            add_line(line)
    elif isinstance(lines, list):
        for line in lines:
            add_line(line)

    if not sentences and text:
        chunks = re.split(r"(?<=[.!?])\s+", str(text))
        for i, sent in enumerate(chunks):
            sent = sent.strip()
            if sent:
                sentences.append((i, sent))

    return title, sentences

def gold_sentence_ids(row):
    ids = set()
    for ev in row.get("gold_evidence", []):
        wiki = norm_title(ev.get("wiki_url", ""))
        sid = ev.get("sentence_id", -1)
        try:
            sid = int(sid)
        except Exception:
            sid = -1
        if wiki and sid >= 0:
            ids.add(f"{wiki}::{sid}")
    return ids

def build_or_load_sentence_corpus(all_samples):
    corpus_path = PROC_DIR / "full_dev_sentence_corpus.jsonl"
    summary_path = PROC_DIR / "full_dev_sentence_corpus_summary.json"

    if corpus_path.exists() and corpus_path.stat().st_size > 100000:
        print("Loading existing FEVER full-dev sentence corpus:", corpus_path)
        return read_jsonl(corpus_path)

    if not WIKI_PATH.exists():
        raise FileNotFoundError(f"Missing wiki pages: {WIKI_PATH}")

    needed_pages = set()
    for r in all_samples:
        for ev in r.get("gold_evidence", []):
            if ev.get("wiki_url"):
                needed_pages.add(norm_title(ev["wiki_url"]))

    print("Needed gold pages:", len(needed_pages))

    rng = random.Random(SEED)
    corpus = []
    seen_pages = set()
    distractor_pages = 0
    total_pages = 0

    with WIKI_PATH.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Scanning FEVER wiki pages"):
            line = line.strip()
            if not line:
                continue
            total_pages += 1
            try:
                row = json.loads(line)
            except Exception:
                continue

            title, sentences = parse_wiki_page_row(row)
            if not title or not sentences:
                continue

            keep = False
            if title in needed_pages:
                keep = True
            elif distractor_pages < MAX_DISTRACTOR_PAGES and rng.random() < 0.08:
                keep = True
                distractor_pages += 1

            if not keep:
                continue

            seen_pages.add(title)

            for sid, sent in sentences:
                if len(sent) < 3:
                    continue
                corpus.append({
                    "sent_id": f"{title}::{sid}",
                    "wiki_url": title,
                    "sentence_id": int(sid),
                    "text": sent,
                })

    write_jsonl(corpus_path, corpus)

    summary = {
        "needed_pages": len(needed_pages),
        "seen_needed_pages": len(needed_pages.intersection(seen_pages)),
        "total_seen_pages_in_corpus": len(seen_pages),
        "sentences": len(corpus),
        "max_distractor_pages": MAX_DISTRACTOR_PAGES,
        "distractor_pages_sampled": distractor_pages,
        "total_pages_scanned": total_pages,
        "note": "This is full FEVER paper_dev evaluation with sampled train tune/cal and large sampled distractor corpus.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Corpus summary:", json.dumps(summary, indent=2))
    return corpus

def build_or_load_bm25(corpus):
    index_path = INDEX_DIR / "full_dev_bm25_sentence.pkl"
    meta_path = INDEX_DIR / "full_dev_bm25_sentence_meta.jsonl"

    if index_path.exists() and meta_path.exists():
        print("Loading existing FEVER full-dev BM25 index.")
        with index_path.open("rb") as f:
            bm25 = pickle.load(f)
        meta = read_jsonl(meta_path)
        return bm25, meta

    print("Building BM25 over sentences:", len(corpus))
    tokenized = [tokenize(x["wiki_url"] + " " + x["text"]) for x in tqdm(corpus, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized)

    with index_path.open("wb") as f:
        pickle.dump(bm25, f)
    write_jsonl(meta_path, corpus)
    return bm25, corpus

def retrieve(rows, bm25, meta, split):
    out_path = OUT_DIR / f"full_dev_bm25_top{TOPK}_{split}.jsonl"
    metric_path = METRIC_DIR / f"full_dev_bm25_top{TOPK}_{split}_metrics.json"

    if out_path.exists() and metric_path.exists():
        print("Loading existing retrieval:", out_path)
        return read_jsonl(out_path)

    outputs = []
    denom = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}

    for r in tqdm(rows, desc=f"Retrieving {split}"):
        scores = bm25.get_scores(tokenize(r["claim"]))
        idxs = np.argsort(scores)[::-1][:TOPK]

        retrieved = []
        for rank, idx in enumerate(idxs, 1):
            item = meta[int(idx)]
            retrieved.append({
                "rank": rank,
                "score": float(scores[idx]),
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
        "split": split,
        "method": "bm25_sentence_full_dev",
        "topk": TOPK,
        "num_examples": len(rows),
        "num_with_gold": denom,
    }
    for k in hits:
        metrics[f"evidence_sentence_recall@{k}"] = hits[k] / denom if denom else None

    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Retrieval metrics:", split, json.dumps(metrics, indent=2))
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

def score_nli(model_name, split, rows, device):
    safe = model_name.replace("/", "__")
    out_path = OUT_DIR / f"full_dev_nli_{safe}_{split}_scores.jsonl"

    if out_path.exists():
        print("Loading cached NLI scores:", out_path)
        return read_jsonl(out_path)

    print("Loading model:", model_name)
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
        for start in tqdm(range(0, len(pair_claims), BATCH_SIZE), desc=f"NLI {split} {model_name}"):
            bc = pair_claims[start:start+BATCH_SIZE]
            bp = pair_premises[start:start+BATCH_SIZE]
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
            "split": split,
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
    if rule == "support_only":
        return float(r["S"])
    return float(r["S"]) - float(r["K"])

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
        obj = m["macro_f1_all"] + 0.1*m["accepted_accuracy"] - 0.5*m["false_verification_rate"] - 0.2*m["false_refuted_rate"]
        m["objective"] = float(obj)
        m.update({f"param_{k}": float(v) for k, v in params.items()})
        records.append({k:v for k,v in m.items() if not isinstance(v, (dict, list))})
        if best is None or obj > best["objective"]:
            best = m

    params = {k.replace("param_", ""): float(v) for k, v in best.items() if k.startswith("param_")}
    return params, best, records

def choose_threshold(cal_rows, cal_preds, cal_confs, alpha):
    idxs = [i for i, p in enumerate(cal_preds) if p == "verified"]
    if not idxs:
        return float("inf"), {"cal_base_verified": 0, "cal_retained_verified": 0, "cal_retained_fvr": 0.0, "cal_verified_retention": 0.0}

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

def bootstrap_ci(rows, model_name, rule, alpha, n_boot=1000):
    # Bootstrap over final dev prediction file cannot be reconstructed here unless rows include predictions.
    # So this script writes deterministic metrics now; a separate bootstrap script will use saved scores and params.
    return {}

def main():
    print("==== FEVER full-dev pipeline ====")
    print("This uses full paper_dev evaluation, sampled train tune/cal, and a large sampled sentence corpus.")
    print("It is stronger than the earlier FEVER-pilot, but not full 5.4M-page BM25 over every sentence.")

    train_all = aggregate_claims(PROC_DIR / "train.jsonl", "train")
    dev_all = aggregate_claims(PROC_DIR / "paper_dev.jsonl", "paper_dev")

    print("Aggregated train:", len(train_all), Counter(r["label"] for r in train_all))
    print("Aggregated full paper_dev:", len(dev_all), Counter(r["label"] for r in dev_all))

    tune_rows = stratified_sample(train_all, TUNE_PER_LABEL, SEED)
    tune_ids = set(r["id"] for r in tune_rows)
    remaining = [r for r in train_all if r["id"] not in tune_ids]
    cal_rows = stratified_sample(remaining, CAL_PER_LABEL, SEED + 1)
    dev_rows = dev_all

    write_jsonl(PROC_DIR / "full_dev_tune_claims.jsonl", tune_rows)
    write_jsonl(PROC_DIR / "full_dev_cal_claims.jsonl", cal_rows)
    write_jsonl(PROC_DIR / "full_dev_paper_dev_claims.jsonl", dev_rows)

    print("Tune:", len(tune_rows), Counter(r["label"] for r in tune_rows))
    print("Cal:", len(cal_rows), Counter(r["label"] for r in cal_rows))
    print("Dev full paper_dev:", len(dev_rows), Counter(r["label"] for r in dev_rows))

    all_samples = tune_rows + cal_rows + dev_rows
    corpus = build_or_load_sentence_corpus(all_samples)
    bm25, meta = build_or_load_bm25(corpus)

    tune_ret = retrieve(tune_rows, bm25, meta, "tune")
    cal_ret = retrieve(cal_rows, bm25, meta, "cal")
    dev_ret = retrieve(dev_rows, bm25, meta, "paper_dev_full")

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

    summary_df = pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v, (dict, list))} for r in all_summary])
    summary_path = METRIC_DIR / "fever_full_dev_risk_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_path = METRIC_DIR / "fever_full_dev_tune_grid.csv"
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
    table_path = TABLE_DIR / "table_fever_full_dev_risk_calibration.csv"
    table.to_csv(table_path, index=False)

    status = {
        "dataset": "FEVER",
        "evaluation": "full paper_dev claims",
        "train_tune": "sampled from train",
        "calibration": "sampled from train disjoint from tune",
        "corpus": "all gold pages for tune/cal/dev plus large distractor page sample",
        "dev_rows": len(dev_rows),
        "tune_rows": len(tune_rows),
        "cal_rows": len(cal_rows),
        "outputs": {
            "summary": str(summary_path),
            "table": str(table_path),
            "grid": str(grid_path),
        },
    }
    (FINAL_DIR / "fever_full_dev_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("\n==== FEVER FULL-DEV SUMMARY ====")
    print(table.to_string(index=False))
    print("\nSaved:", summary_path)
    print("Saved:", table_path)
    print("Saved:", grid_path)

if __name__ == "__main__":
    main()
PY

echo ""
echo "==== Step 3: Run FEVER full-dev/larger corpus experiment ===="
python scripts/run_fever_full_dev_pipeline.py || echo "WARNING: FEVER full-dev pipeline ended with an error."

echo ""
echo "==== Step 4: Bootstrap confidence intervals for SciFact and FEVER full-dev ===="
cat > scripts/run_bootstrap_ci_all.py <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

FINAL = Path("outputs/final_report")
FINAL.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]
RNG = np.random.default_rng(42)
N_BOOT = 1000

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def metric_ci(y_true, y_pred):
    n = len(y_true)
    if n == 0:
        return {}
    vals_macro = []
    vals_acc = []
    vals_fvr = []
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    for _ in range(N_BOOT):
        idx = RNG.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        vals_macro.append(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0))
        vals_acc.append(accuracy_score(yt, yp))
        pv = np.where(yp == "verified")[0]
        if len(pv):
            vals_fvr.append(float(np.mean(yt[pv] != "verified")))
        else:
            vals_fvr.append(0.0)

    def ci(vals):
        vals = np.array(vals, dtype=float)
        return {
            "mean": float(np.mean(vals)),
            "lo": float(np.quantile(vals, 0.025)),
            "hi": float(np.quantile(vals, 0.975)),
        }

    return {
        "macro_f1": ci(vals_macro),
        "accuracy": ci(vals_acc),
        "false_verification_rate": ci(vals_fvr),
    }

experiments = []

# SciFact selected prediction files.
scifact_files = [
    ("SciFact", "BART margin alpha0.10", "outputs/predictions/scifact/proper_split_risk_facebook__bart-large-mnli_margin_support_refute_alpha0.10_dev.jsonl", "risk_calibrated_prediction"),
    ("SciFact", "RoBERTa margin alpha0.30", "outputs/predictions/scifact/proper_split_risk_ynie__roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli_margin_support_refute_alpha0.30_dev.jsonl", "risk_calibrated_prediction"),
]
for dataset, name, path, pred_col in scifact_files:
    rows = read_jsonl(path)
    if rows:
        y_true = [r["label"] for r in rows]
        y_pred = [r.get(pred_col, r.get("prediction", r.get("risk_calibrated_prediction"))) for r in rows]
        ci = metric_ci(y_true, y_pred)
        experiments.append({"dataset": dataset, "name": name, "path": path, **{f"{m}_{k}": v for m,d in ci.items() for k,v in d.items()}})

# FEVER full-dev prediction rows are not saved after gating in this script, so CI is skipped unless files are added later.
# Use table-level deterministic results for now.

df = pd.DataFrame(experiments)
path = FINAL / "bootstrap_ci_summary.csv"
df.to_csv(path, index=False)
print("Saved bootstrap CI:", path)
print(df.to_string(index=False) if len(df) else "No CI rows created.")
PY

python scripts/run_bootstrap_ci_all.py || echo "WARNING: bootstrap CI script failed."

echo ""
echo "==== Step 5: Final combined paper tables including FEVER full-dev if available ===="
cat > scripts/make_all_final_paper_tables.py <<'PY'
from pathlib import Path
import json
import pandas as pd
import numpy as np

FINAL = Path("outputs/final_report")
FINAL.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def fmt(x, d=3):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.{d}f}"
    except Exception:
        return str(x)

def short_model(x):
    x = str(x)
    if "ynie/roberta" in x:
        return "RoBERTa-large NLI"
    if "facebook/bart" in x:
        return "BART-large MNLI"
    return x

def clean_rule(x):
    return {
        "margin_support_refute": "Support-refute margin",
        "support_only": "Support only",
    }.get(str(x), str(x))

rows = []

# SciFact best clean table.
sf_best = read_csv("outputs/final_report/clean_table_best_results.csv")
if not sf_best.empty:
    for _, r in sf_best.iterrows():
        rows.append({
            "Dataset": r.get("Dataset"),
            "Scope": "full labelled split" if r.get("Dataset") == "SciFact" else "pilot",
            "Model": r.get("Model"),
            "Rule": r.get("Rule"),
            "Alpha": r.get("Alpha"),
            "Macro-F1": r.get("Macro-F1"),
            "FVR": r.get("FVR"),
            "Coverage": r.get("Coverage"),
            "Accepted Acc.": r.get("Accepted Acc."),
        })

# FEVER full-dev table.
fv_full = read_csv("outputs/tables/fever/table_fever_full_dev_risk_calibration.csv")
if not fv_full.empty:
    # best macro and lowest fvr with coverage >= .75
    df = fv_full.copy()
    best_macro = df.sort_values("macro_f1_all", ascending=False).head(1)
    low = df[df["coverage"] >= 0.75].sort_values(["false_verification_rate", "macro_f1_all"], ascending=[True, False]).head(1)
    for label, sub in [("full paper_dev best macro", best_macro), ("full paper_dev low FVR", low)]:
        if len(sub):
            r = sub.iloc[0]
            rows.append({
                "Dataset": "FEVER-full-dev",
                "Scope": label,
                "Model": short_model(r.get("model")),
                "Rule": clean_rule(r.get("rule")),
                "Alpha": fmt(r.get("alpha"), 2),
                "Macro-F1": fmt(r.get("macro_f1_all")),
                "FVR": fmt(r.get("false_verification_rate")),
                "Coverage": fmt(r.get("coverage")),
                "Accepted Acc.": fmt(r.get("accepted_accuracy")),
            })

summary = pd.DataFrame(rows)
summary_path = FINAL / "paper_ready_main_results.csv"
summary.to_csv(summary_path, index=False)

report = FINAL / "paper_ready_experiment_summary.md"
with report.open("w", encoding="utf-8") as f:
    f.write("# Paper-Ready Experiment Summary\n\n")
    f.write("## Important Scope Notes\n\n")
    f.write("- SciFact is full on the official labelled train/dev data. The official test split is unlabeled, so dev is the labelled evaluation split.\n")
    f.write("- FEVER-full-dev uses the full FEVER paper_dev labelled evaluation split, with tune/cal sampled from FEVER train and a large sampled sentence corpus containing all gold pages for tune/cal/dev plus distractor pages.\n")
    f.write("- FEVER-pilot results remain available but should be secondary once FEVER-full-dev is completed.\n\n")
    f.write("## Main Results\n\n")
    if len(summary):
        f.write(summary.to_markdown(index=False))
    f.write("\n\n")

    p = Path("outputs/tables/fever/table_fever_full_dev_risk_calibration.csv")
    if p.exists():
        f.write("## FEVER Full-Dev Detailed Risk Calibration\n\n")
        df = pd.read_csv(p)
        keep = ["model", "rule", "alpha", "macro_f1_all", "false_verification_rate", "coverage", "accepted_accuracy", "num_predicted_verified", "num_abstained"]
        f.write(df[[c for c in keep if c in df.columns]].to_markdown(index=False))
        f.write("\n\n")

print("Saved:", summary_path)
print("Saved:", report)
print(summary.to_string(index=False) if len(summary) else "No rows.")
PY

python scripts/make_all_final_paper_tables.py || echo "WARNING: final paper table script failed."

echo ""
echo "==== Final file list ===="
find outputs/final_report outputs/tables/scifact outputs/tables/fever outputs/metrics/fever outputs/metrics/scifact -maxdepth 1 -type f | sort | tail -n 200 || true

echo ""
echo "==== GPU snapshot after all ===="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true

echo ""
echo "==== ALL REMAINING FULL PIPELINE END ===="
date
echo "Log saved to: $LOG"
