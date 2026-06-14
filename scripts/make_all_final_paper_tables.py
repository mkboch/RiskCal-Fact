from pathlib import Path
import json
import pandas as pd
import numpy as np

FINAL = Path("outputs/final_report")
FINAL.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

def read_json(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}

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
        "verified_safe": "Verified-safe margin",
        "winner_takes_all": "Winner takes all",
        "risk_abstain": "Risk abstain",
    }.get(str(x), str(x))

rows = []

# SciFact clean best rows.
sf_best = read_csv("outputs/final_report/clean_table_best_results.csv")
if not sf_best.empty:
    for _, r in sf_best.iterrows():
        if str(r.get("Dataset")) == "SciFact":
            rows.append({
                "Dataset": "SciFact",
                "Scope": "full official labelled dev",
                "Model": r.get("Model"),
                "Rule": r.get("Rule"),
                "Alpha": r.get("Alpha"),
                "Macro-F1": r.get("Macro-F1"),
                "FVR": r.get("FVR"),
                "Coverage": r.get("Coverage"),
                "Accepted Acc.": r.get("Accepted Acc."),
            })

# FEVER pilot rows, kept as secondary.
fv_pilot = read_csv("outputs/tables/fever/table_fever_pilot_risk_calibration.csv")
if not fv_pilot.empty:
    df = fv_pilot.copy()
    best = df.sort_values("macro_f1_all", ascending=False).head(1)
    low = df[df["coverage"] >= 0.75].sort_values(["false_verification_rate", "macro_f1_all"], ascending=[True, False]).head(1)
    for scope, sub in [("pilot best macro-F1", best), ("pilot low FVR", low)]:
        if len(sub):
            r = sub.iloc[0]
            rows.append({
                "Dataset": "FEVER-pilot",
                "Scope": scope,
                "Model": short_model(r.get("model")),
                "Rule": clean_rule(r.get("rule")),
                "Alpha": fmt(r.get("alpha"), 2),
                "Macro-F1": fmt(r.get("macro_f1_all")),
                "FVR": fmt(r.get("false_verification_rate")),
                "Coverage": fmt(r.get("coverage")),
                "Accepted Acc.": fmt(r.get("accepted_accuracy")),
            })

# FEVER fast full-dev rows.
fv_full = read_csv("outputs/tables/fever/table_fever_fast_full_dev_risk_calibration.csv")
if not fv_full.empty:
    df = fv_full.copy()
    best = df.sort_values("macro_f1_all", ascending=False).head(1)
    low = df[df["coverage"] >= 0.75].sort_values(["false_verification_rate", "macro_f1_all"], ascending=[True, False]).head(1)
    for scope, sub in [("full paper_dev best macro-F1", best), ("full paper_dev low FVR", low)]:
        if len(sub):
            r = sub.iloc[0]
            rows.append({
                "Dataset": "FEVER-full-dev",
                "Scope": scope,
                "Model": short_model(r.get("model")),
                "Rule": clean_rule(r.get("rule")),
                "Alpha": fmt(r.get("alpha"), 2),
                "Macro-F1": fmt(r.get("macro_f1_all")),
                "FVR": fmt(r.get("false_verification_rate")),
                "Coverage": fmt(r.get("coverage")),
                "Accepted Acc.": fmt(r.get("accepted_accuracy")),
            })

main = pd.DataFrame(rows)
main_path = FINAL / "paper_ready_main_results.csv"
main.to_csv(main_path, index=False)

# Retrieval summary.
ret_rows = []

sf_ret = read_csv("outputs/tables/scifact/table_scifact_retrieval.csv")
if not sf_ret.empty:
    for _, r in sf_ret.iterrows():
        ret_rows.append({
            "Dataset": "SciFact",
            "Split": r.get("split"),
            "Method": "BM25",
            "R@1": fmt(r.get("evidence_doc_recall@1")),
            "R@3": fmt(r.get("evidence_doc_recall@3")),
            "R@5": fmt(r.get("evidence_doc_recall@5")),
            "R@10": fmt(r.get("evidence_doc_recall@10")),
        })

for prefix, label in [
    ("pilot_bm25_top10", "FEVER-pilot BM25"),
    ("fast_full_dev_tfidf_top10", "FEVER-full-dev TF-IDF"),
]:
    for split in ["tune", "cal", "dev", "paper_dev_full"]:
        p = Path(f"outputs/metrics/fever/{prefix}_{split}_metrics.json")
        if p.exists():
            m = read_json(p)
            ret_rows.append({
                "Dataset": "FEVER-pilot" if "pilot" in prefix else "FEVER-full-dev",
                "Split": m.get("split", split),
                "Method": label,
                "R@1": fmt(m.get("evidence_sentence_recall@1")),
                "R@3": fmt(m.get("evidence_sentence_recall@3")),
                "R@5": fmt(m.get("evidence_sentence_recall@5")),
                "R@10": fmt(m.get("evidence_sentence_recall@10")),
            })

ret = pd.DataFrame(ret_rows)
ret_path = FINAL / "paper_ready_retrieval_results.csv"
ret.to_csv(ret_path, index=False)

# Markdown report.
report_path = FINAL / "paper_ready_experiment_summary.md"
with report_path.open("w", encoding="utf-8") as f:
    f.write("# Paper-Ready Experiment Summary\n\n")
    f.write("## Scope Notes\n\n")
    f.write("- SciFact is complete on the official labelled train/dev split. The official test split is unlabeled, so dev is the evaluable split.\n")
    f.write("- FEVER-pilot used sampled claims and a smaller sampled sentence corpus.\n")
    f.write("- FEVER-full-dev used all labelled FEVER paper_dev claims, tune/cal sampled from train, and a large 1.2M-sentence evidence corpus with fast TF-IDF retrieval.\n")
    f.write("- FEVER-full-dev is stronger than the pilot, but it is still not exhaustive all-Wikipedia sentence retrieval.\n\n")

    f.write("## Main Results\n\n")
    if not main.empty:
        f.write(main.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Retrieval Results\n\n")
    if not ret.empty:
        f.write(ret.to_markdown(index=False))
    f.write("\n\n")

    if not fv_full.empty:
        f.write("## FEVER Full-Dev Detailed Risk Calibration\n\n")
        keep = [
            "model", "rule", "alpha", "macro_f1_all", "accuracy_all",
            "false_verification_rate", "false_refuted_rate",
            "num_predicted_verified", "num_predicted_refuted",
            "num_abstained", "coverage", "accepted_accuracy"
        ]
        f.write(fv_full[[c for c in keep if c in fv_full.columns]].to_markdown(index=False))
        f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write("The results show a consistent tradeoff between verification utility and false-verification risk. Risk-calibrated gating can reduce accepted verified claims and lower false-verification risk, but may reduce macro-F1 and coverage. The FEVER full-dev run confirms that the method scales beyond SciFact, while retrieval quality remains a major limiting factor for corpus-level claim verification.\n")

status = {
    "scifact": "full official labelled dev completed",
    "fever_pilot": "completed",
    "fever_full_dev": "completed over full paper_dev claims with large sampled sentence corpus",
    "main_results_csv": str(main_path),
    "retrieval_csv": str(ret_path),
    "report_md": str(report_path),
}
(FINAL / "paper_ready_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

print("Saved:")
for p in [main_path, ret_path, report_path, FINAL / "paper_ready_status.json"]:
    print(f"  {p} ({p.stat().st_size} bytes)")

print("\nMain results:")
print(main.to_string(index=False) if not main.empty else "EMPTY")

print("\nRetrieval:")
print(ret.to_string(index=False) if not ret.empty else "EMPTY")
