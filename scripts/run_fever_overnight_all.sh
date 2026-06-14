#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/fever_overnight_all_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FEVER OVERNIGHT ALL START ===="
date
echo "PROJECT_DIR=$PROJECT_DIR"
echo "LOG=$LOG"

echo ""
echo "==== GPU snapshot before run ===="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true

echo ""
echo "==== Ensure dynamic GPU selector exists ===="
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

echo ""
echo "==== Step 1: Export FEVER wiki_pages if needed using rcv_fever_loader ===="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_fever_loader

mkdir -p data/raw/fever_wiki_pages scripts

cat > scripts/export_fever_wiki_pages.py <<'PY'
import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/raw/fever_wiki_pages")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "wiki_pages.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

if OUT_PATH.exists() and OUT_PATH.stat().st_size > 1000000:
    print(f"Wiki pages already exported: {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    if SUMMARY_PATH.exists():
        print(SUMMARY_PATH.read_text()[:3000])
    raise SystemExit(0)

print("Loading fever/fever wiki_pages...")
ds = load_dataset("fever/fever", "wiki_pages", trust_remote_code=True)
print(ds)

summary = {"splits": {}}
total = 0

with OUT_PATH.open("w", encoding="utf-8") as fout:
    for split in ds.keys():
        n = len(ds[split])
        cols = ds[split].column_names
        summary["splits"][split] = {"rows": n, "columns": cols}
        print(f"Split={split}, rows={n}, columns={cols}")
        print("Example:", json.dumps(ds[split][0], ensure_ascii=False)[:2000])

        for row in ds[split]:
            item = dict(row)
            item["_split"] = split
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1

summary["total_rows"] = total
summary["output"] = str(OUT_PATH)

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Saved wiki pages:", OUT_PATH)
print("Summary:", json.dumps(summary, indent=2)[:5000])
PY

python scripts/export_fever_wiki_pages.py || echo "WARNING: wiki export failed or skipped."

echo ""
echo "==== Step 2: Run FEVER pilot retrieval + NLI + risk calibration using rcv_py310 ===="
conda activate rcv_py310
source scripts/select_free_gpu.sh 1

python --version
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

cat > scripts/run_fever_pilot_all.py <<'PY'
import os
import re
import gc
import json
import math
import pickle
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROJECT = Path(".")
PROC_DIR = Path("data/processed/fever")
WIKI_PATH = Path("data/raw/fever_wiki_pages/wiki_pages.jsonl")
OUT_DIR = Path("outputs/predictions/fever")
METRIC_DIR = Path("outputs/metrics/fever")
TABLE_DIR = Path("outputs/tables/fever")
INDEX_DIR = Path("data/indexes/fever")
for d in [OUT_DIR, METRIC_DIR, TABLE_DIR, INDEX_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABELS = ["verified", "refuted", "unsupported"]

# Keep pilot controlled but meaningful.
TRAIN_PER_LABEL = 600
CAL_PER_LABEL = 300
DEV_PER_LABEL = 600
MAX_DISTRACTOR_PAGES = 30000
TOPK = 10
MAX_LENGTH = 256
BATCH_SIZE = 32

MODELS = [
    "facebook/bart-large-mnli",
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
]

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

_token_re = re.compile(r"[A-Za-z0-9]+")

def tokenize(text):
    return _token_re.findall((text or "").lower())

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

def norm_title(t):
    return str(t or "").replace(" ", "_")

def aggregate_claims(path, split_name):
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
        eid = meta.get("evidence_id", -1)
        ann = meta.get("evidence_annotation_id", -1)

        try:
            sid_int = int(sid)
        except Exception:
            sid_int = -1

        if wiki and sid_int >= 0:
            ev = {
                "wiki_url": str(wiki),
                "sentence_id": sid_int,
                "evidence_id": eid,
                "annotation_id": ann,
            }
            if ev not in by_id[cid]["gold_evidence"]:
                by_id[cid]["gold_evidence"].append(ev)

    rows = list(by_id.values())
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
    # Try many possible schemas from FEVER wiki pages.
    title = row.get("id", row.get("page_id", row.get("title", row.get("wiki_url", ""))))
    title = str(title)
    title_norm = norm_title(title)

    lines = row.get("lines", None)
    text = row.get("text", None)

    sentences = []

    def add_line_line(line):
        line = str(line)
        if not line.strip():
            return
        # FEVER line format is often: sentence_id \t sentence text \t links...
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
            add_line_line(line)
    elif isinstance(lines, list):
        for line in lines:
            if isinstance(line, dict):
                sid = line.get("line_id", line.get("sentence_id", len(sentences)))
                sent = line.get("text", line.get("sentence", ""))
                try:
                    sid = int(sid)
                except Exception:
                    sid = len(sentences)
                sent = str(sent).strip()
                if sent:
                    sentences.append((sid, sent))
            else:
                add_line_line(line)

    if not sentences and text:
        if isinstance(text, list):
            for i, sent in enumerate(text):
                sent = str(sent).strip()
                if sent:
                    sentences.append((i, sent))
        else:
            # Fall back to rough sentence split.
            chunks = re.split(r"(?<=[.!?])\s+", str(text))
            for i, sent in enumerate(chunks):
                sent = sent.strip()
                if sent:
                    sentences.append((i, sent))

    return title_norm, sentences

def build_or_load_sentence_corpus(all_samples):
    corpus_path = PROC_DIR / "pilot_sentence_corpus.jsonl"
    summary_path = PROC_DIR / "pilot_sentence_corpus_summary.json"

    if corpus_path.exists() and corpus_path.stat().st_size > 100000:
        print(f"Loading existing pilot sentence corpus: {corpus_path}")
        return read_jsonl(corpus_path)

    if not WIKI_PATH.exists():
        raise FileNotFoundError(f"Missing wiki pages export: {WIKI_PATH}")

    needed_pages = set()
    for r in all_samples:
        for ev in r.get("gold_evidence", []):
            if ev.get("wiki_url"):
                needed_pages.add(norm_title(ev["wiki_url"]))

    print("Needed evidence pages:", len(needed_pages))

    rng = random.Random(SEED)
    corpus = []
    seen_pages = set()
    distractor_pages = 0
    total_pages = 0

    with WIKI_PATH.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading wiki pages"):
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
            elif distractor_pages < MAX_DISTRACTOR_PAGES and rng.random() < 0.02:
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
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("Pilot corpus summary:")
    print(json.dumps(summary, indent=2))
    print("Saved:", corpus_path)
    return corpus

def gold_sentence_ids(row):
    out = set()
    for ev in row.get("gold_evidence", []):
        wiki = norm_title(ev.get("wiki_url", ""))
        sid = ev.get("sentence_id", -1)
        try:
            sid = int(sid)
        except Exception:
            sid = -1
        if wiki and sid >= 0:
            out.add(f"{wiki}::{sid}")
    return out

def build_bm25(corpus):
    index_path = INDEX_DIR / "pilot_bm25_sentence.pkl"
    meta_path = INDEX_DIR / "pilot_bm25_sentence_meta.jsonl"

    if index_path.exists() and meta_path.exists():
        print("Loading existing BM25 sentence index.")
        with index_path.open("rb") as f:
            bm25 = pickle.load(f)
        meta = read_jsonl(meta_path)
        return bm25, meta

    print("Building BM25 sentence index over", len(corpus), "sentences")
    tokenized = [tokenize(x["wiki_url"] + " " + x["text"]) for x in tqdm(corpus, desc="Tokenizing sentence corpus")]
    bm25 = BM25Okapi(tokenized)

    with index_path.open("wb") as f:
        pickle.dump(bm25, f)
    write_jsonl(meta_path, corpus)
    return bm25, corpus

def retrieve(rows, bm25, meta, split):
    out_path = OUT_DIR / f"pilot_bm25_top{TOPK}_{split}.jsonl"
    if out_path.exists():
        print("Loading existing retrieval:", out_path)
        return read_jsonl(out_path)

    outputs = []
    denom = 0
    hits = {1: 0, 3: 0, 5: 0, 10: 0}

    for r in tqdm(rows, desc=f"BM25 retrieve {split}"):
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
        "method": "bm25_sentence",
        "topk": TOPK,
        "num_examples": len(rows),
        "num_with_gold": denom,
    }
    for k in hits:
        metrics[f"evidence_sentence_recall@{k}"] = hits[k] / denom if denom else None

    metric_path = METRIC_DIR / f"pilot_bm25_top{TOPK}_{split}_metrics.json"
    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Retrieval metrics", split, json.dumps(metrics, indent=2))
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
    if "entailment" not in mapping or "contradiction" not in mapping:
        raise RuntimeError(f"Cannot infer NLI mapping: {id2label}")
    return mapping

def score_nli(model_name, split, rows, device):
    out_path = OUT_DIR / f"pilot_nli_{model_name.replace('/', '__')}_{split}_scores.jsonl"
    if out_path.exists():
        print("Loading cached NLI scores:", out_path)
        return read_jsonl(out_path)

    print("Loading NLI model:", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    mapping = infer_mapping(model)
    print("id2label:", model.config.id2label)
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
                neu.append(float(row[mapping["neutral"]]) if "neutral" in mapping else 0.0)

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
            "S": float(S),
            "K": float(K),
            "N": float(N),
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

def choose_risk_threshold(cal_rows, cal_preds, cal_confs, alpha):
    idxs = [i for i,p in enumerate(cal_preds) if p == "verified"]
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
    return ["abstain" if p == "verified" and c < threshold else p for p,c in zip(preds, confs)]

def run_calibration(model_name, tune_rows, cal_rows, dev_rows):
    all_rows = []
    tune_grid_records = []

    for rule in ["support_only", "margin_support_refute"]:
        params, best, grid_records = tune_params(tune_rows, rule)
        for g in grid_records:
            g["model"] = model_name
            g["rule"] = rule
        tune_grid_records.extend(grid_records)

        tune_preds = [predict(rule, r, params) for r in tune_rows]
        cal_preds = [predict(rule, r, params) for r in cal_rows]
        dev_preds = [predict(rule, r, params) for r in dev_rows]

        cal_confs = [verified_conf(rule, r) for r in cal_rows]
        dev_confs = [verified_conf(rule, r) for r in dev_rows]

        base_dev = eval_preds(dev_rows, dev_preds)

        for alpha in ALPHAS:
            thr, cal_info = choose_risk_threshold(cal_rows, cal_preds, cal_confs, alpha)
            risk_preds = apply_gate(dev_preds, dev_confs, thr)
            m = eval_preds(dev_rows, risk_preds)
            row = {
                "model": model_name,
                "rule": rule,
                "alpha": alpha,
                "threshold": thr,
                **{f"param_{k}": v for k,v in params.items()},
                **cal_info,
                "base_dev_macro_f1": base_dev["macro_f1_all"],
                "base_dev_false_verification_rate": base_dev["false_verification_rate"],
                **m,
            }
            all_rows.append(row)

    return all_rows, tune_grid_records

def main():
    print("==== Load and aggregate FEVER claims ====")
    train_agg_path = PROC_DIR / "pilot_train_agg.jsonl"
    dev_agg_path = PROC_DIR / "pilot_paper_dev_agg.jsonl"

    if train_agg_path.exists() and dev_agg_path.exists():
        train_rows_all = read_jsonl(train_agg_path)
        dev_rows_all = read_jsonl(dev_agg_path)
    else:
        train_rows_all = aggregate_claims(PROC_DIR / "train.jsonl", "train")
        dev_rows_all = aggregate_claims(PROC_DIR / "paper_dev.jsonl", "paper_dev")
        write_jsonl(train_agg_path, train_rows_all)
        write_jsonl(dev_agg_path, dev_rows_all)

    print("Aggregated train:", len(train_rows_all), Counter(r["label"] for r in train_rows_all))
    print("Aggregated paper_dev:", len(dev_rows_all), Counter(r["label"] for r in dev_rows_all))

    train_tune = stratified_sample(train_rows_all, TRAIN_PER_LABEL, SEED)
    remaining_train = [r for r in train_rows_all if r["id"] not in set(x["id"] for x in train_tune)]
    train_cal = stratified_sample(remaining_train, CAL_PER_LABEL, SEED + 1)
    dev_sample = stratified_sample(dev_rows_all, DEV_PER_LABEL, SEED + 2)

    write_jsonl(PROC_DIR / "pilot_tune_claims.jsonl", train_tune)
    write_jsonl(PROC_DIR / "pilot_cal_claims.jsonl", train_cal)
    write_jsonl(PROC_DIR / "pilot_dev_claims.jsonl", dev_sample)

    print("Pilot tune:", len(train_tune), Counter(r["label"] for r in train_tune))
    print("Pilot cal:", len(train_cal), Counter(r["label"] for r in train_cal))
    print("Pilot dev:", len(dev_sample), Counter(r["label"] for r in dev_sample))

    all_samples = train_tune + train_cal + dev_sample

    print("==== Build/load pilot sentence corpus ====")
    corpus = build_or_load_sentence_corpus(all_samples)
    bm25, meta = build_bm25(corpus)

    print("==== Retrieval ====")
    tune_ret = retrieve(train_tune, bm25, meta, "tune")
    cal_ret = retrieve(train_cal, bm25, meta, "cal")
    dev_ret = retrieve(dev_sample, bm25, meta, "dev")

    print("==== NLI scoring ====")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    final_summary = []
    final_grids = []

    for model_name in MODELS:
        tune_scores = score_nli(model_name, "tune", tune_ret, device)
        cal_scores = score_nli(model_name, "cal", cal_ret, device)
        dev_scores = score_nli(model_name, "dev", dev_ret, device)

        rows, grids = run_calibration(model_name, tune_scores, cal_scores, dev_scores)
        final_summary.extend(rows)
        final_grids.extend(grids)

    summary_df = pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v, (dict, list))} for r in final_summary])
    summary_path = METRIC_DIR / "fever_pilot_risk_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_path = METRIC_DIR / "fever_pilot_tune_grid.csv"
    pd.DataFrame(final_grids).to_csv(grid_path, index=False)

    # Compact tables.
    retrieval_rows = []
    for split in ["tune", "cal", "dev"]:
        p = METRIC_DIR / f"pilot_bm25_top{TOPK}_{split}_metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            retrieval_rows.append(m)
    pd.DataFrame(retrieval_rows).to_csv(TABLE_DIR / "table_fever_pilot_retrieval.csv", index=False)

    keep = [
        "model", "rule", "alpha", "macro_f1_all", "false_verification_rate",
        "num_predicted_verified", "num_predicted_refuted", "num_abstained",
        "coverage", "accepted_accuracy", "cal_retained_fvr",
        "cal_verified_retention", "base_dev_macro_f1", "base_dev_false_verification_rate"
    ]
    table_df = summary_df[[c for c in keep if c in summary_df.columns]].copy()
    table_df.to_csv(TABLE_DIR / "table_fever_pilot_risk_calibration.csv", index=False)

    print("\n==== FEVER PILOT SUMMARY ====")
    print(table_df.to_string(index=False))
    print("\nSaved summary:", summary_path)
    print("Saved grid:", grid_path)
    print("Saved tables:")
    for p in sorted(TABLE_DIR.glob("table_fever_pilot*.csv")):
        print(" ", p, p.stat().st_size, "bytes")

if __name__ == "__main__":
    main()
PY

python scripts/run_fever_pilot_all.py || echo "WARNING: FEVER pilot script ended with an error."

echo ""
echo "==== Final outputs ===="
find data/processed/fever outputs/metrics/fever outputs/tables/fever outputs/predictions/fever -maxdepth 1 -type f -printf "%p %k KB\n" | sort | tail -n 120 || true

echo ""
echo "==== GPU snapshot after run ===="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader,nounits || true

echo ""
echo "==== FEVER OVERNIGHT ALL END ===="
date
echo "Log saved to: $LOG"
