import json
import gc
from pathlib import Path
from collections import defaultdict

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
    mapping = {}
    for idx, lab in model.config.id2label.items():
        s = str(lab).lower()
        if "entail" in s:
            mapping["entailment"] = int(idx)
        elif "contrad" in s:
            mapping["contradiction"] = int(idx)
        elif "neutral" in s:
            mapping["neutral"] = int(idx)
    if set(mapping) != {"entailment", "contradiction", "neutral"}:
        mapping = {"entailment": 0, "neutral": 1, "contradiction": 2}
    return mapping

def score_all(all_inputs):
    out_path = INP / "fever_oracle_flat_scores.jsonl"
    if out_path.exists():
        print("Loading cached scores:", out_path)
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

    print("NLI pairs:", len(pairs))
    scores_by_i = defaultdict(list)

    with torch.no_grad():
        for start in tqdm(range(0, len(pairs), 32), desc="Oracle flat NLI"):
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
        us = scores_by_i.get(i, [])
        scored.append({
            "id": r["id"],
            "split": r["split"],
            "claim": r["claim"],
            "label": r["label"],
            "P": int(r.get("P", 0)),
            "num_oracle_units": int(r.get("num_oracle_units", 0)),
            "S": max([u["entailment"] for u in us], default=0.0),
            "K": max([u["contradiction"] for u in us], default=0.0),
            "N": max([u["neutral"] for u in us], default=1.0),
            "unit_scores": us,
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
    cov = len(accepted) / len(rows)
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
    tune_in = read_jsonl(INP / "fever_oracle_flat_tune_inputs.jsonl")
    cal_in = read_jsonl(INP / "fever_oracle_flat_cal_inputs.jsonl")
    dev_in = read_jsonl(INP / "fever_oracle_flat_dev_inputs.jsonl")

    print("Input counts:", len(tune_in), len(cal_in), len(dev_in))
    all_inputs = tune_in + cal_in + dev_in
    scored = score_all(all_inputs)

    tune_rows = [r for r in scored if r["split"] == "tune"]
    cal_rows = [r for r in scored if r["split"] == "cal"]
    dev_rows = [r for r in scored if r["split"] == "dev"]

    rows = []
    for rule in ["S", "S+K", "S+K+P"]:
        best = tune(tune_rows, rule)
        preds = [predict(r, rule, best["tau_s"], best["tau_k"], best["margin"])[0] for r in dev_rows]
        rows.append({
            "Setting": "FEVER oracle diagnostic",
            "Model": "RoBERTa-large NLI",
            "Rule": rule,
            "Alpha": "",
            "tau_s": best["tau_s"],
            "tau_k": best["tau_k"],
            "margin": best["margin"],
            **eval_preds(dev_rows, preds),
        })

    best = tune(tune_rows, "S+K+P")
    for alpha in [0.05, 0.10, 0.20, 0.30]:
        thr = risk_threshold(cal_rows, best["tau_s"], best["tau_k"], best["margin"], alpha)
        preds = [
            predict(r, "S+K+P+rho", best["tau_s"], best["tau_k"], best["margin"], risk_thr=thr)[0]
            for r in dev_rows
        ]
        rows.append({
            "Setting": "FEVER oracle diagnostic",
            "Model": "RoBERTa-large NLI",
            "Rule": "S+K+P+rho",
            "Alpha": alpha,
            "risk_threshold": thr,
            "tau_s": best["tau_s"],
            "tau_k": best["tau_k"],
            "margin": best["margin"],
            **eval_preds(dev_rows, preds),
        })

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "table_fever_oracle_flat_diagnostic.csv", index=False)
    df.to_csv(MET / "fever_oracle_flat_diagnostic_results.csv", index=False)

    tex_df = df.copy()
    for c in ["macro_f1", "accuracy", "false_verification_rate", "coverage", "accepted_accuracy"]:
        tex_df[c] = tex_df[c].map(lambda x: f"{float(x):.3f}")
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{FEVER oracle diagnostic using gold evidence sentence identifiers to isolate verifier behavior from retrieval errors.}\n"
    tex += "\\label{tab:fever_oracle_diagnostic}\n"
    tex += tex_df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_fever_oracle_diagnostic.tex").write_text(tex, encoding="utf-8")

    report = FINAL / "fever_oracle_flat_diagnostic_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# FEVER Oracle Diagnostic\n\n")
        f.write("This diagnostic uses flat HuggingFace FEVER gold evidence columns, `evidence_wiki_url` and `evidence_sentence_id`, and maps them to available sentence-corpus text.\n\n")
        f.write("## Results\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Interpretation\n\n")
        f.write("This diagnostic isolates verifier behavior from retrieval errors for a balanced oracle subset. It should be interpreted alongside the retrieval-grounded FEVER results.\n")

    print("\nFEVER oracle diagnostic results:")
    print(df.to_string(index=False))

    status = {
        "status": "completed",
        "outputs": [
            "outputs/tables/review_hardening/table_fever_oracle_flat_diagnostic.csv",
            "outputs/metrics/review_hardening/fever_oracle_flat_diagnostic_results.csv",
            "outputs/latex_tables/table_fever_oracle_diagnostic.tex",
            "outputs/final_report/fever_oracle_flat_diagnostic_report.md",
        ],
    }
    (FINAL / "fever_oracle_flat_diagnostic_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("\n==== FEVER ORACLE FLAT DIAGNOSTIC COMPLETE ====")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
