#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/fever_oracle_raw_fixed_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FEVER ORACLE RAW FIXED START ===="
date

mkdir -p outputs/final_report outputs/metrics/review_hardening outputs/tables/review_hardening outputs/latex_tables outputs/predictions/review_hardening scripts

echo ""
echo "==== Stage 1: Build oracle inputs from raw HuggingFace FEVER ===="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_fever_loader
echo "loader python=$(command -v python)"
python --version

cat > scripts/build_fever_oracle_raw_inputs.py <<'PY'
import json
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

SEED = 42
random.seed(SEED)

OUT = Path("outputs/predictions/review_hardening")
MET = Path("outputs/metrics/review_hardening")
OUT.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_title(x):
    return str(x).replace(" ", "_")

def label_norm(x):
    s = str(x).strip().lower()
    if s in {"supports", "support", "supported", "verified", "1"}:
        return "verified"
    if s in {"refutes", "refute", "refuted", "0"}:
        return "refuted"
    if s in {"not enough info", "nei", "unsupported", "2"}:
        return "unsupported"
    return s

def traverse_evidence(obj):
    pairs = []

    if obj is None:
        return pairs

    if isinstance(obj, dict):
        page = (
            obj.get("wiki_url")
            or obj.get("wikipedia_url")
            or obj.get("page")
            or obj.get("title")
            or obj.get("doc_id")
        )
        sid = (
            obj.get("sentence_id")
            or obj.get("line_num")
            or obj.get("line")
            or obj.get("sent_id")
        )

        if page is not None and sid is not None:
            try:
                pairs.append((norm_title(page), int(sid)))
            except Exception:
                pass

        for v in obj.values():
            pairs.extend(traverse_evidence(v))

    elif isinstance(obj, (list, tuple)):
        # Common FEVER raw evidence item:
        # [annotation_id, evidence_id, wiki_url, sentence_id]
        if len(obj) >= 4 and isinstance(obj[2], str):
            try:
                pairs.append((norm_title(obj[2]), int(obj[3])))
            except Exception:
                pass

        for v in obj:
            pairs.extend(traverse_evidence(v))

    return pairs

def get_evidence_pairs(row):
    for key in ["evidence", "gold_evidence", "evidences"]:
        if key in row:
            pairs = traverse_evidence(row[key])
            if pairs:
                seen = set()
                out = []
                for p in pairs:
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                return out
    return []

def load_corpus_map():
    # Prefer full-dev corpus because it contains many more FEVER gold pages.
    candidates = [
        Path("data/processed/fever/full_dev_sentence_corpus.jsonl"),
        Path("data/processed/fever/pilot_sentence_corpus.jsonl"),
    ]

    corpus_file = None
    for c in candidates:
        if c.exists():
            corpus_file = c
            break

    if corpus_file is None:
        raise FileNotFoundError("No FEVER sentence corpus found.")

    rows = read_jsonl(corpus_file)
    m = {}

    for r in rows:
        text = r.get("text") or r.get("sentence") or ""
        page = r.get("wiki_url") or r.get("wikipedia_url") or r.get("page") or r.get("title") or r.get("doc_id") or ""
        sid = r.get("sentence_id") or r.get("line_num") or r.get("line") or None

        if "sent_id" in r:
            sid_str = str(r["sent_id"])
            if "::" in sid_str:
                page2, sid2 = sid_str.rsplit("::", 1)
                try:
                    m[(norm_title(page2), int(sid2))] = text
                except Exception:
                    pass

        if page and sid is not None:
            try:
                m[(norm_title(page), int(sid))] = text
            except Exception:
                pass

    print(f"Using corpus: {corpus_file}")
    print(f"Corpus lookup entries: {len(m)}")
    return m, corpus_file

def row_to_example(row, corpus_map, source_split, idx):
    claim = row.get("claim", "")
    label = label_norm(row.get("label", row.get("claim_label", "")))
    pairs = get_evidence_pairs(row)

    ev_texts = []
    found = 0
    seen_txt = set()

    for p in pairs:
        txt = corpus_map.get(p)
        if txt:
            found += 1
            if txt not in seen_txt:
                seen_txt.add(txt)
                ev_texts.append(txt)

    ev_texts = ev_texts[:5]

    return {
        "id": str(row.get("id", f"{source_split}_{idx}")),
        "source_split": source_split,
        "claim": claim,
        "label": label,
        "evidence_units": ev_texts,
        "P": 1 if ev_texts else 0,
        "num_gold_pairs": len(pairs),
        "num_gold_pairs_found": found,
        "num_oracle_units": len(ev_texts),
    }

def make_balanced_splits(examples):
    usable = []
    for ex in examples:
        # Keep unsupported even with no evidence; require oracle evidence for verified/refuted.
        if ex["label"] == "unsupported":
            usable.append(ex)
        elif ex["label"] in {"verified", "refuted"} and ex["P"] == 1:
            usable.append(ex)

    by = defaultdict(list)
    for ex in usable:
        by[ex["label"]].append(ex)

    for lab in LABELS:
        random.shuffle(by[lab])
        print(f"Usable {lab}: {len(by[lab])}")

    targets = {"tune": 300, "cal": 150, "dev": 300}
    splits = {"tune": [], "cal": [], "dev": []}

    for lab in LABELS:
        needed = sum(targets.values())
        if len(by[lab]) < needed:
            print(f"WARNING: only {len(by[lab])} usable examples for {lab}; requested {needed}.")
        start = 0
        for split, n in targets.items():
            chunk = by[lab][start:start+n]
            for ex in chunk:
                ex = dict(ex)
                ex["split"] = split
                splits[split].append(ex)
            start += n

    for split in splits:
        random.shuffle(splits[split])

    return splits, usable

def main():
    corpus_map, corpus_file = load_corpus_map()

    print("Loading raw FEVER from HuggingFace...")
    ds = load_dataset("fever/fever", "v1.0", trust_remote_code=True)
    print(ds)

    source_splits = []
    for name in ["paper_dev", "labelled_dev", "train"]:
        if name in ds:
            source_splits.append(name)

    print("Using source splits:", source_splits)

    raw_examples = []
    raw_stats = []

    for split in source_splits:
        n = len(ds[split])
        print(f"Processing raw split {split}: {n}")
        for i, row in enumerate(ds[split]):
            ex = row_to_example(row, corpus_map, split, i)
            if ex["label"] in LABELS and ex["claim"]:
                raw_examples.append(ex)

        # lightweight split stats after adding all rows
        part = [e for e in raw_examples if e["source_split"] == split]
        raw_stats.append({
            "source_split": split,
            "rows": len(part),
            "verified": sum(1 for e in part if e["label"] == "verified"),
            "refuted": sum(1 for e in part if e["label"] == "refuted"),
            "unsupported": sum(1 for e in part if e["label"] == "unsupported"),
            "verified_or_refuted_with_found_evidence": sum(1 for e in part if e["label"] in {"verified", "refuted"} and e["P"] == 1),
            "mean_gold_pairs": sum(e["num_gold_pairs"] for e in part) / len(part) if part else 0,
            "mean_found_pairs": sum(e["num_gold_pairs_found"] for e in part) / len(part) if part else 0,
        })

    splits, usable = make_balanced_splits(raw_examples)

    for split, rows in splits.items():
        write_jsonl(OUT / f"fever_oracle_raw_{split}_inputs.jsonl", rows)

    stats = {
        "corpus_file": str(corpus_file),
        "raw_stats": raw_stats,
        "usable_total": len(usable),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "label_counts": {
            k: {lab: sum(1 for e in v if e["label"] == lab) for lab in LABELS}
            for k, v in splits.items()
        }
    }

    (MET / "fever_oracle_raw_input_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Saved oracle raw input stats:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()
PY

python scripts/build_fever_oracle_raw_inputs.py

echo ""
echo "==== Stage 2: Score oracle inputs with main environment ===="
conda activate rcv_py310
echo "main python=$(command -v python)"
python --version
source scripts/select_free_gpu.sh 1 || true

cat > scripts/score_fever_oracle_raw_inputs.py <<'PY'
import json
import gc
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INP = Path("outputs/predictions/review_hardening")
MET = Path("outputs/metrics/review_hardening")
TAB = Path("outputs/tables/review_hardening")
FINAL = Path("outputs/final_report")
TEX = Path("outputs/latex_tables")
for d in [INP, MET, TAB, FINAL, TEX]:
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]
MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def infer_mapping(model):
    id2label = model.config.id2label
    mapping = {}
    for idx, lab in id2label.items():
        s = str(lab).lower()
        if "entail" in s:
            mapping["entailment"] = int(idx)
        elif "contrad" in s:
            mapping["contradiction"] = int(idx)
        elif "neutral" in s:
            mapping["neutral"] = int(idx)
    if set(mapping.keys()) != {"entailment", "contradiction", "neutral"}:
        mapping = {"contradiction": 2, "neutral": 1, "entailment": 0}
    return mapping

def score_all(all_inputs):
    out_path = INP / "fever_oracle_raw_scores.jsonl"
    if out_path.exists():
        print("Loading cached oracle raw scores.")
        return read_jsonl(out_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Loading model:", MODEL_NAME)

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()
    mapping = infer_mapping(model)
    print("NLI mapping:", mapping)

    pairs = []
    refs = []

    for i, r in enumerate(all_inputs):
        for j, ev in enumerate(r.get("evidence_units", [])):
            pairs.append((ev, r["claim"]))
            refs.append((i, j))

    scores_by_i = defaultdict(list)

    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), 32), desc="Oracle raw NLI"):
            batch = pairs[start:start+32]
            premises = [x[0] for x in batch]
            claims = [x[1] for x in batch]
            enc = tok(premises, claims, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).detach().cpu().numpy()

            for prob, ref in zip(probs, refs[start:start+32]):
                scores_by_i[ref[0]].append({
                    "entailment": float(prob[mapping["entailment"]]),
                    "contradiction": float(prob[mapping["contradiction"]]),
                    "neutral": float(prob[mapping["neutral"]]),
                })

    scored = []
    for i, r in enumerate(all_inputs):
        unit_scores = scores_by_i.get(i, [])
        scored.append({
            "id": r["id"],
            "split": r["split"],
            "claim": r["claim"],
            "label": r["label"],
            "P": int(r.get("P", 0)),
            "num_oracle_units": int(r.get("num_oracle_units", 0)),
            "S": max([u["entailment"] for u in unit_scores], default=0.0),
            "K": max([u["contradiction"] for u in unit_scores], default=0.0),
            "N": max([u["neutral"] for u in unit_scores], default=1.0),
            "unit_scores": unit_scores,
        })

    write_jsonl(out_path, scored)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return scored

def eval_preds(rows, preds):
    y = [r["label"] for r in rows]
    macro = f1_score(y, preds, labels=LABELS, average="macro", zero_division=0)
    acc = accuracy_score(y, preds)
    pv = [i for i, p in enumerate(preds) if p == "verified"]
    fvr = sum(1 for i in pv if y[i] != "verified") / len(pv) if pv else 0.0
    accepted = [i for i, p in enumerate(preds) if p != "abstain"]
    cov = len(accepted) / len(rows) if rows else 0.0
    acc_accepted = accuracy_score([y[i] for i in accepted], [preds[i] for i in accepted]) if accepted else 0.0
    return {
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "false_verification_rate": float(fvr),
        "coverage": float(cov),
        "accepted_accuracy": float(acc_accepted),
        "num_predicted_verified": len(pv),
        "num_abstained": len(rows) - len(accepted),
    }

def predict(r, rule, tau_s, tau_k, margin, risk_thr=None):
    S = float(r["S"])
    K = float(r["K"])
    P = int(r["P"])
    conf = S - K

    if rule == "S":
        pred = "verified" if S >= tau_s else "unsupported"
    else:
        if K >= tau_k and (K - S) >= margin:
            pred = "refuted"
        elif S >= tau_s and (S - K) >= margin and (rule == "S+K" or P == 1):
            pred = "verified"
        else:
            pred = "unsupported"

        if rule == "S+K+P+rho" and pred == "verified" and risk_thr is not None and conf < risk_thr:
            pred = "abstain"

    return pred, conf

def tune(rows, rule):
    vals = [0.30, 0.40, 0.50, 0.60, 0.70]
    margins = [0.00, 0.10, 0.20, 0.30]
    best = None

    if rule == "S":
        grid = [(ts, 0.5, 0.0) for ts in vals]
    else:
        grid = [(ts, tk, m) for ts in vals for tk in vals for m in margins]

    for ts, tk, m in grid:
        preds = [predict(r, rule, ts, tk, m)[0] for r in rows]
        met = eval_preds(rows, preds)
        obj = met["macro_f1"] - 0.25 * met["false_verification_rate"]
        if best is None or obj > best["obj"]:
            best = {"obj": obj, "tau_s": ts, "tau_k": tk, "margin": m}
    return best

def risk_threshold(cal_rows, tau_s, tau_k, margin, alpha):
    pc = [predict(r, "S+K+P", tau_s, tau_k, margin) for r in cal_rows]
    idx = [i for i, (p, c) in enumerate(pc) if p == "verified"]
    if not idx:
        return 999.0

    thresholds = sorted(set(pc[i][1] for i in idx))
    for thr in thresholds:
        kept = [i for i in idx if pc[i][1] >= thr]
        fvr = sum(1 for i in kept if cal_rows[i]["label"] != "verified") / len(kept) if kept else 0.0
        if fvr <= alpha:
            return float(thr)

    return float(max(thresholds) + 1e-6)

def main():
    tune_in = read_jsonl(INP / "fever_oracle_raw_tune_inputs.jsonl")
    cal_in = read_jsonl(INP / "fever_oracle_raw_cal_inputs.jsonl")
    dev_in = read_jsonl(INP / "fever_oracle_raw_dev_inputs.jsonl")

    print("Input counts:", len(tune_in), len(cal_in), len(dev_in))
    all_inputs = tune_in + cal_in + dev_in

    if not all_inputs:
        raise RuntimeError("No oracle raw inputs found.")

    scored = score_all(all_inputs)
    tune_rows = [r for r in scored if r["split"] == "tune"]
    cal_rows = [r for r in scored if r["split"] == "cal"]
    dev_rows = [r for r in scored if r["split"] == "dev"]

    rows = []

    for rule in ["S", "S+K", "S+K+P"]:
        best = tune(tune_rows, rule)
        preds = [predict(r, rule, best["tau_s"], best["tau_k"], best["margin"])[0] for r in dev_rows]
        met = eval_preds(dev_rows, preds)
        rows.append({
            "Setting": "FEVER raw oracle diagnostic",
            "Model": "RoBERTa-large NLI",
            "Rule": rule,
            "Alpha": "",
            "tau_s": best["tau_s"],
            "tau_k": best["tau_k"],
            "margin": best["margin"],
            **met,
        })

    best = tune(tune_rows, "S+K+P")
    for alpha in [0.05, 0.10, 0.20, 0.30]:
        thr = risk_threshold(cal_rows, best["tau_s"], best["tau_k"], best["margin"], alpha)
        preds = [
            predict(r, "S+K+P+rho", best["tau_s"], best["tau_k"], best["margin"], risk_thr=thr)[0]
            for r in dev_rows
        ]
        met = eval_preds(dev_rows, preds)
        rows.append({
            "Setting": "FEVER raw oracle diagnostic",
            "Model": "RoBERTa-large NLI",
            "Rule": "S+K+P+rho",
            "Alpha": alpha,
            "risk_threshold": thr,
            "tau_s": best["tau_s"],
            "tau_k": best["tau_k"],
            "margin": best["margin"],
            **met,
        })

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table_fever_oracle_raw_diagnostic.csv", index=False)
    df.to_csv(MET / "fever_oracle_raw_diagnostic_results.csv", index=False)

    tex_df = df.copy()
    for c in ["macro_f1", "accuracy", "false_verification_rate", "coverage", "accepted_accuracy"]:
        tex_df[c] = tex_df[c].map(lambda x: f"{float(x):.3f}")
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{FEVER oracle diagnostic using raw gold evidence identifiers and available sentence-corpus text.}\n"
    tex += "\\label{tab:fever_oracle_raw_diagnostic}\n"
    tex += tex_df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_fever_oracle_raw_diagnostic.tex").write_text(tex, encoding="utf-8")

    report = FINAL / "fever_oracle_raw_diagnostic_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# FEVER Raw Oracle Diagnostic\n\n")
        f.write("This diagnostic uses raw HuggingFace FEVER gold evidence identifiers and available sentence-corpus text to isolate verifier behavior from retrieval errors.\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Interpretation\n\n")
        f.write("This diagnostic should be compared against retrieval-grounded FEVER results. Higher oracle performance indicates that retrieval is a limiting factor; similar oracle and retrieval-grounded performance indicates that NLI/verifier scoring is also a limiting factor.\n")

    print("\nFEVER raw oracle diagnostic results:")
    print(df.to_string(index=False))

    status = {
        "status": "completed",
        "outputs": [
            "outputs/tables/review_hardening/table_fever_oracle_raw_diagnostic.csv",
            "outputs/metrics/review_hardening/fever_oracle_raw_diagnostic_results.csv",
            "outputs/latex_tables/table_fever_oracle_raw_diagnostic.tex",
            "outputs/final_report/fever_oracle_raw_diagnostic_report.md",
        ]
    }
    (FINAL / "fever_oracle_raw_diagnostic_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("\n==== FEVER RAW ORACLE DIAGNOSTIC COMPLETE ====")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
PY

python scripts/score_fever_oracle_raw_inputs.py

echo ""
echo "==== Final FEVER raw-oracle files ===="
find outputs/final_report outputs/tables/review_hardening outputs/metrics/review_hardening outputs/latex_tables outputs/predictions/review_hardening \
  -maxdepth 1 -type f | grep -E "fever_oracle_raw" | sort || true

echo ""
echo "==== FEVER ORACLE RAW FIXED END ===="
date
echo "Log saved to: $LOG"
