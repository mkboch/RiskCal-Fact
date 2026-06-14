#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/fever_oracle_pilot_fast_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FEVER ORACLE PILOT FAST START ===="
date

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_py310
echo "python=$(command -v python)"
python --version

source scripts/select_free_gpu.sh 1 || true

mkdir -p outputs/final_report outputs/metrics/review_hardening outputs/tables/review_hardening outputs/latex_tables outputs/predictions/review_hardening scripts

cat > scripts/run_fever_oracle_pilot_fast.py <<'PY'
import json
import random
import gc
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

OUT_METRIC = Path("outputs/metrics/review_hardening")
OUT_TABLE = Path("outputs/tables/review_hardening")
OUT_PRED = Path("outputs/predictions/review_hardening")
OUT_FINAL = Path("outputs/final_report")
OUT_TEX = Path("outputs/latex_tables")
for d in [OUT_METRIC, OUT_TABLE, OUT_PRED, OUT_FINAL, OUT_TEX]:
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]
MODEL_NAME = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"

def read_jsonl(path):
    p = Path(path)
    rows = []
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
    if s in {"supports", "support", "supported", "verified"}:
        return "verified"
    if s in {"refutes", "refute", "refuted"}:
        return "refuted"
    if s in {"not enough info", "nei", "unsupported"}:
        return "unsupported"
    return s

def traverse_evidence(obj):
    """Flexible extractor for FEVER evidence structures."""
    out = []

    if obj is None:
        return out

    if isinstance(obj, dict):
        page = obj.get("wiki_url") or obj.get("page") or obj.get("title") or obj.get("doc_id")
        sid = obj.get("sentence_id") or obj.get("line_num") or obj.get("line") or obj.get("sent_id")
        if page is not None and sid is not None:
            try:
                out.append((norm_title(page), int(sid)))
            except Exception:
                pass
        for v in obj.values():
            out.extend(traverse_evidence(v))

    elif isinstance(obj, list):
        # FEVER raw evidence items are often [annotation_id, evidence_id, wiki_url, sentence_id]
        if len(obj) >= 4 and isinstance(obj[2], str):
            try:
                out.append((norm_title(obj[2]), int(obj[3])))
            except Exception:
                pass
        for v in obj:
            out.extend(traverse_evidence(v))

    return out

def get_evidence_pairs(row):
    for key in ["gold_evidence", "evidence", "evidences"]:
        if key in row:
            pairs = traverse_evidence(row[key])
            if pairs:
                # deduplicate
                seen = set()
                clean = []
                for p in pairs:
                    if p not in seen:
                        seen.add(p)
                        clean.append(p)
                return clean
    return []

def load_corpus_map():
    candidates = [
        Path("data/processed/fever/pilot_sentence_corpus.jsonl"),
        Path("data/processed/fever/full_dev_sentence_corpus.jsonl"),
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
        page = r.get("wiki_url") or r.get("page") or r.get("title") or r.get("doc_id") or ""
        sid = r.get("sentence_id") or r.get("line_num") or r.get("line") or None

        if "sent_id" in r:
            m[str(r["sent_id"])] = text
            # Also parse sent_id if possible.
            if "::" in str(r["sent_id"]):
                page2, sid2 = str(r["sent_id"]).rsplit("::", 1)
                m[(norm_title(page2), int(sid2))] = text

        if page != "" and sid is not None:
            try:
                m[(norm_title(page), int(sid))] = text
            except Exception:
                pass

    print(f"Using corpus: {corpus_file}")
    print(f"Corpus lookup entries: {len(m)}")
    return m, corpus_file

def load_claim_pool():
    candidates = [
        Path("data/processed/fever/paper_dev.jsonl"),
        Path("data/processed/fever/labelled_dev.jsonl"),
        Path("data/processed/fever/train.jsonl"),
    ]
    rows = []
    for p in candidates:
        part = read_jsonl(p)
        if part:
            print(f"Loaded {len(part)} claims from {p}")
            rows.extend(part)

    clean = []
    for r in rows:
        lab = label_norm(r.get("label", r.get("claim_label", "")))
        claim = r.get("claim", "")
        if lab in LABELS and claim:
            clean.append({**r, "label": lab, "claim": claim})

    return clean

def balanced_sample(rows, n_per_label):
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)

    out = []
    for lab in LABELS:
        random.shuffle(by[lab])
        out.extend(by[lab][:n_per_label])
    random.shuffle(out)
    return out

def make_oracle_examples(rows, corpus_map, split):
    examples = []
    found_claims = 0
    total_gold_ids = 0
    found_gold_ids = 0

    for i, r in enumerate(rows):
        pairs = get_evidence_pairs(r)
        ev_texts = []
        used = set()

        for p in pairs:
            total_gold_ids += 1
            txt = corpus_map.get(p)
            if txt:
                found_gold_ids += 1
                if p not in used:
                    used.add(p)
                    ev_texts.append(txt)

        ev_texts = ev_texts[:5]
        if ev_texts:
            found_claims += 1

        # Unsupported claims often have no gold evidence; keep them as unsupported with no evidence.
        examples.append({
            "id": r.get("id", f"{split}_{i}"),
            "split": split,
            "claim": r["claim"],
            "label": r["label"],
            "evidence_units": ev_texts,
            "P": 1 if ev_texts else 0,
            "num_gold_pairs": len(pairs),
            "num_oracle_units": len(ev_texts),
        })

    stats = {
        "split": split,
        "claims": len(rows),
        "claims_with_oracle_evidence": found_claims,
        "claims_with_oracle_evidence_rate": found_claims / len(rows) if rows else 0.0,
        "gold_evidence_ids_total": total_gold_ids,
        "gold_evidence_ids_found": found_gold_ids,
        "gold_evidence_id_found_rate": found_gold_ids / total_gold_ids if total_gold_ids else 0.0,
    }

    return examples, stats

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
        mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}
    return mapping

def score_examples(rows):
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
    for i, r in enumerate(rows):
        for j, ev in enumerate(r["evidence_units"]):
            pairs.append((ev, r["claim"]))
            refs.append((i, j))

    scores_by_i = defaultdict(list)

    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), 32), desc="Oracle NLI scoring"):
            batch = pairs[start:start+32]
            if not batch:
                continue
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

    out = []
    for i, r in enumerate(rows):
        unit_scores = scores_by_i.get(i, [])
        out.append({
            "id": r["id"],
            "split": r["split"],
            "claim": r["claim"],
            "label": r["label"],
            "P": r["P"],
            "num_oracle_units": r["num_oracle_units"],
            "S": max([u["entailment"] for u in unit_scores], default=0.0),
            "K": max([u["contradiction"] for u in unit_scores], default=0.0),
            "N": max([u["neutral"] for u in unit_scores], default=1.0),
            "unit_scores": unit_scores,
        })

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out

def eval_preds(rows, preds):
    y = [r["label"] for r in rows]
    macro = f1_score(y, preds, labels=LABELS, average="macro", zero_division=0)
    acc = accuracy_score(y, preds)
    pred_verified = [i for i, p in enumerate(preds) if p == "verified"]
    fvr = sum(1 for i in pred_verified if y[i] != "verified") / len(pred_verified) if pred_verified else 0.0
    accepted = [i for i, p in enumerate(preds) if p != "abstain"]
    cov = len(accepted) / len(rows)
    acc_accept = accuracy_score([y[i] for i in accepted], [preds[i] for i in accepted]) if accepted else 0.0
    return {
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "false_verification_rate": float(fvr),
        "coverage": float(cov),
        "accepted_accuracy": float(acc_accept),
        "num_predicted_verified": len(pred_verified),
        "num_abstained": len(rows) - len(accepted),
    }

def predict(r, rule, tau_s, tau_k, margin, risk_thr=None):
    S = float(r["S"])
    K = float(r["K"])
    P = int(r["P"])
    conf = S - K

    if rule == "S":
        pred = "verified" if S >= tau_s else "unsupported"
    elif rule in {"S+K", "S+K+P", "S+K+P+rho"}:
        if K >= tau_k and (K - S) >= margin:
            pred = "refuted"
        elif S >= tau_s and (S - K) >= margin and (rule == "S+K" or P == 1):
            pred = "verified"
        else:
            pred = "unsupported"

        if rule == "S+K+P+rho" and pred == "verified" and risk_thr is not None and conf < risk_thr:
            pred = "abstain"
    else:
        raise ValueError(rule)

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
    corpus_map, corpus_file = load_corpus_map()
    claims = load_claim_pool()
    print("Clean labelled claims:", len(claims))

    # Keep this fast but reviewer-useful: balanced FEVER-pilot oracle.
    tune_raw = balanced_sample(claims, 300)
    cal_raw = balanced_sample([r for r in claims if r not in tune_raw], 150)
    dev_raw = balanced_sample([r for r in claims if r not in tune_raw and r not in cal_raw], 300)

    tune_ex, st1 = make_oracle_examples(tune_raw, corpus_map, "tune")
    cal_ex, st2 = make_oracle_examples(cal_raw, corpus_map, "cal")
    dev_ex, st3 = make_oracle_examples(dev_raw, corpus_map, "dev")

    write_jsonl(OUT_PRED / "fever_oracle_pilot_tune_inputs.jsonl", tune_ex)
    write_jsonl(OUT_PRED / "fever_oracle_pilot_cal_inputs.jsonl", cal_ex)
    write_jsonl(OUT_PRED / "fever_oracle_pilot_dev_inputs.jsonl", dev_ex)

    stats_df = pd.DataFrame([st1, st2, st3])
    stats_df.to_csv(OUT_METRIC / "fever_oracle_pilot_evidence_recovery.csv", index=False)
    print("\nOracle evidence recovery:")
    print(stats_df.to_string(index=False))

    all_scored = score_examples(tune_ex + cal_ex + dev_ex)
    tune_scores = [r for r in all_scored if r["split"] == "tune"]
    cal_scores = [r for r in all_scored if r["split"] == "cal"]
    dev_scores = [r for r in all_scored if r["split"] == "dev"]

    write_jsonl(OUT_PRED / "fever_oracle_pilot_scores.jsonl", all_scored)

    rows = []
    for rule in ["S", "S+K", "S+K+P"]:
        best = tune(tune_scores, rule)
        preds = [predict(r, rule, best["tau_s"], best["tau_k"], best["margin"])[0] for r in dev_scores]
        met = eval_preds(dev_scores, preds)
        rows.append({
            "Setting": "FEVER oracle pilot",
            "Model": "RoBERTa-large NLI",
            "Rule": rule,
            "Alpha": "",
            "tau_s": best["tau_s"],
            "tau_k": best["tau_k"],
            "margin": best["margin"],
            **met,
        })

    best = tune(tune_scores, "S+K+P")
    for alpha in [0.05, 0.10, 0.20, 0.30]:
        thr = risk_threshold(cal_scores, best["tau_s"], best["tau_k"], best["margin"], alpha)
        preds = [
            predict(r, "S+K+P+rho", best["tau_s"], best["tau_k"], best["margin"], risk_thr=thr)[0]
            for r in dev_scores
        ]
        met = eval_preds(dev_scores, preds)
        rows.append({
            "Setting": "FEVER oracle pilot",
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
    df.to_csv(OUT_TABLE / "table_fever_oracle_pilot.csv", index=False)
    df.to_csv(OUT_METRIC / "fever_oracle_pilot_results.csv", index=False)

    tex_df = df.copy()
    for c in ["macro_f1", "accuracy", "false_verification_rate", "coverage", "accepted_accuracy"]:
        tex_df[c] = tex_df[c].map(lambda x: f"{float(x):.3f}")
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{FEVER oracle-pilot diagnostic using gold evidence sentences to isolate verifier behavior from retrieval errors.}\n"
    tex += "\\label{tab:fever_oracle_pilot}\n"
    tex += tex_df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (OUT_TEX / "table_fever_oracle_pilot.tex").write_text(tex, encoding="utf-8")

    report = OUT_FINAL / "fever_oracle_pilot_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# FEVER Oracle-Pilot Diagnostic\n\n")
        f.write("This diagnostic uses gold FEVER evidence sentences rather than retrieved sentences to isolate verifier behavior from retrieval errors.\n\n")
        f.write("## Evidence Recovery\n\n")
        f.write(stats_df.to_markdown(index=False))
        f.write("\n\n## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Interpretation\n\n")
        f.write("This table should be compared against FEVER retrieval-grounded results. If oracle evidence improves macro-F1 or reduces FVR, then retrieval is a major bottleneck. If oracle evidence does not improve substantially, then verifier/NLI scoring is also a bottleneck.\n")

    print("\nFEVER oracle pilot results:")
    print(df.to_string(index=False))

    status = {
        "status": "completed",
        "corpus_file": str(corpus_file),
        "outputs": [
            "outputs/tables/review_hardening/table_fever_oracle_pilot.csv",
            "outputs/metrics/review_hardening/fever_oracle_pilot_results.csv",
            "outputs/metrics/review_hardening/fever_oracle_pilot_evidence_recovery.csv",
            "outputs/latex_tables/table_fever_oracle_pilot.tex",
            "outputs/final_report/fever_oracle_pilot_report.md",
        ],
    }
    (OUT_FINAL / "fever_oracle_pilot_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("\n==== FEVER ORACLE PILOT COMPLETE ====")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
PY

python scripts/run_fever_oracle_pilot_fast.py

echo ""
echo "==== Final FEVER oracle-pilot files ===="
find outputs/final_report outputs/tables/review_hardening outputs/metrics/review_hardening outputs/latex_tables outputs/predictions/review_hardening \
  -maxdepth 1 -type f | grep -E "fever_oracle_pilot" | sort || true

echo ""
echo "==== FEVER ORACLE PILOT FAST END ===="
date
echo "Log saved to: $LOG"
