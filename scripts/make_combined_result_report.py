from pathlib import Path
import json
import pandas as pd

OUT_DIR = Path("outputs/final_report")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def safe_read_csv(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    print(f"Missing: {p}")
    return pd.DataFrame()

rows = []

# -------------------------
# SciFact retrieval
# -------------------------
for split in ["train", "dev"]:
    p = Path(f"outputs/metrics/scifact/bm25_top10_{split}_metrics.json")
    if p.exists():
        m = read_json(p)
        rows.append({
            "dataset": "SciFact",
            "experiment": "retrieval",
            "split": split,
            "method": "BM25 sentence/doc retrieval",
            "alpha": "",
            "macro_f1": "",
            "accuracy": "",
            "false_verification_rate": "",
            "coverage": "",
            "accepted_accuracy": "",
            "retrieval_recall@1": m.get("evidence_doc_recall@1"),
            "retrieval_recall@3": m.get("evidence_doc_recall@3"),
            "retrieval_recall@5": m.get("evidence_doc_recall@5"),
            "retrieval_recall@10": m.get("evidence_doc_recall@10"),
        })

# -------------------------
# SciFact main verification
# -------------------------
sf_main = safe_read_csv("outputs/tables/scifact/table_scifact_main_verification.csv")
if not sf_main.empty:
    for _, r in sf_main.iterrows():
        rows.append({
            "dataset": "SciFact",
            "experiment": "verification",
            "split": "dev",
            "method": f"{r.get('model')} | {r.get('rule')}",
            "alpha": "",
            "macro_f1": r.get("macro_f1_all"),
            "accuracy": r.get("accuracy_all"),
            "false_verification_rate": r.get("false_verification_rate"),
            "coverage": r.get("coverage"),
            "accepted_accuracy": r.get("accepted_accuracy"),
            "retrieval_recall@1": "",
            "retrieval_recall@3": "",
            "retrieval_recall@5": "",
            "retrieval_recall@10": "",
        })

# -------------------------
# SciFact risk calibration
# -------------------------
sf_risk = safe_read_csv("outputs/tables/scifact/table_scifact_risk_calibration.csv")
if not sf_risk.empty:
    for _, r in sf_risk.iterrows():
        rows.append({
            "dataset": "SciFact",
            "experiment": "risk_calibration",
            "split": "dev",
            "method": f"{r.get('model')} | {r.get('rule')}",
            "alpha": r.get("alpha"),
            "macro_f1": r.get("macro_f1_all"),
            "accuracy": "",
            "false_verification_rate": r.get("false_verification_rate"),
            "coverage": r.get("coverage"),
            "accepted_accuracy": r.get("accepted_accuracy"),
            "retrieval_recall@1": "",
            "retrieval_recall@3": "",
            "retrieval_recall@5": "",
            "retrieval_recall@10": "",
        })

# -------------------------
# FEVER pilot retrieval
# -------------------------
fever_ret = safe_read_csv("outputs/tables/fever/table_fever_pilot_retrieval.csv")
if not fever_ret.empty:
    for _, r in fever_ret.iterrows():
        rows.append({
            "dataset": "FEVER-pilot",
            "experiment": "retrieval",
            "split": r.get("split"),
            "method": "BM25 sentence retrieval",
            "alpha": "",
            "macro_f1": "",
            "accuracy": "",
            "false_verification_rate": "",
            "coverage": "",
            "accepted_accuracy": "",
            "retrieval_recall@1": r.get("evidence_sentence_recall@1"),
            "retrieval_recall@3": r.get("evidence_sentence_recall@3"),
            "retrieval_recall@5": r.get("evidence_sentence_recall@5"),
            "retrieval_recall@10": r.get("evidence_sentence_recall@10"),
        })

# -------------------------
# FEVER pilot risk calibration
# -------------------------
fever_risk = safe_read_csv("outputs/tables/fever/table_fever_pilot_risk_calibration.csv")
if not fever_risk.empty:
    for _, r in fever_risk.iterrows():
        rows.append({
            "dataset": "FEVER-pilot",
            "experiment": "risk_calibration",
            "split": "dev",
            "method": f"{r.get('model')} | {r.get('rule')}",
            "alpha": r.get("alpha"),
            "macro_f1": r.get("macro_f1_all"),
            "accuracy": "",
            "false_verification_rate": r.get("false_verification_rate"),
            "coverage": r.get("coverage"),
            "accepted_accuracy": r.get("accepted_accuracy"),
            "retrieval_recall@1": "",
            "retrieval_recall@3": "",
            "retrieval_recall@5": "",
            "retrieval_recall@10": "",
        })

combined = pd.DataFrame(rows)
combined_path = OUT_DIR / "combined_all_results_long.csv"
combined.to_csv(combined_path, index=False)

# Best rows by dataset/experiment.
best_rows = []

for dataset in combined["dataset"].dropna().unique():
    sub = combined[(combined["dataset"] == dataset) & (combined["experiment"].isin(["verification", "risk_calibration"]))].copy()
    if not sub.empty:
        sub["macro_f1_num"] = pd.to_numeric(sub["macro_f1"], errors="coerce")
        sub["fvr_num"] = pd.to_numeric(sub["false_verification_rate"], errors="coerce")
        sub["coverage_num"] = pd.to_numeric(sub["coverage"], errors="coerce")

        best_macro = sub.sort_values("macro_f1_num", ascending=False).head(1)
        if len(best_macro):
            rr = best_macro.iloc[0].to_dict()
            rr["selection"] = "best_macro_f1"
            best_rows.append(rr)

        low_fvr = sub[sub["coverage_num"] >= 0.75].sort_values(["fvr_num", "macro_f1_num"], ascending=[True, False]).head(1)
        if len(low_fvr):
            rr = low_fvr.iloc[0].to_dict()
            rr["selection"] = "lowest_fvr_with_coverage_ge_0.75"
            best_rows.append(rr)

best_df = pd.DataFrame(best_rows)
best_path = OUT_DIR / "combined_best_results.csv"
best_df.to_csv(best_path, index=False)

# Markdown report.
report_path = OUT_DIR / "combined_result_report.md"

with report_path.open("w", encoding="utf-8") as f:
    f.write("# Combined Experimental Result Report\n\n")

    f.write("## Dataset Status\n\n")
    f.write("- SciFact: official dataset downloaded, processed, retrieved, scored, calibrated, and tabled.\n")
    f.write("- FEVER: v1.0 claims and wiki pages downloaded; pilot experiment completed with sampled tune/cal/dev splits and sampled sentence corpus.\n\n")

    f.write("## Retrieval Summary\n\n")
    retrieval = combined[combined["experiment"] == "retrieval"].copy()
    if not retrieval.empty:
        f.write(retrieval.to_markdown(index=False))
        f.write("\n\n")

    f.write("## Best Result Summary\n\n")
    if not best_df.empty:
        cols = [
            "dataset", "selection", "experiment", "method", "alpha",
            "macro_f1", "false_verification_rate", "coverage", "accepted_accuracy"
        ]
        cols = [c for c in cols if c in best_df.columns]
        f.write(best_df[cols].to_markdown(index=False))
        f.write("\n\n")

    f.write("## SciFact Main Verification Top 10\n\n")
    if not sf_main.empty:
        cols = [
            "model", "rule", "accuracy_all", "macro_f1_all",
            "verified_f1", "refuted_f1", "unsupported_f1",
            "false_verification_rate", "coverage"
        ]
        cols = [c for c in cols if c in sf_main.columns]
        f.write(sf_main[cols].head(10).to_markdown(index=False))
        f.write("\n\n")

    f.write("## SciFact Risk Calibration\n\n")
    if not sf_risk.empty:
        cols = [
            "model", "rule", "alpha", "macro_f1_all",
            "false_verification_rate", "num_predicted_verified",
            "num_abstained", "coverage", "accepted_accuracy"
        ]
        cols = [c for c in cols if c in sf_risk.columns]
        f.write(sf_risk[cols].to_markdown(index=False))
        f.write("\n\n")

    f.write("## FEVER Pilot Risk Calibration\n\n")
    if not fever_risk.empty:
        cols = [
            "model", "rule", "alpha", "macro_f1_all",
            "false_verification_rate", "num_predicted_verified",
            "num_predicted_refuted", "num_abstained",
            "coverage", "accepted_accuracy"
        ]
        cols = [c for c in cols if c in fever_risk.columns]
        f.write(fever_risk[cols].to_markdown(index=False))
        f.write("\n\n")

    f.write("## Interpretation Notes\n\n")
    f.write("1. SciFact shows a clear performance-risk tradeoff: the highest macro-F1 setting has higher false-verification risk, while the lower-risk setting sacrifices macro-F1.\n")
    f.write("2. FEVER pilot confirms that the risk-calibrated framework runs on a larger evidence corpus and public benchmark.\n")
    f.write("3. FEVER results should be called pilot results unless the experiment is scaled to the full corpus and full split.\n")
    f.write("4. The risk calibration should be presented as empirical risk calibration under split/exchangeability assumptions, not as an unconditional guarantee.\n")

print("Saved combined outputs:")
for p in sorted(OUT_DIR.glob("*")):
    print(f"  {p} ({p.stat().st_size} bytes)")

print("\nBest results:")
if not best_df.empty:
    print(best_df[[
        "dataset", "selection", "experiment", "method", "alpha",
        "macro_f1", "false_verification_rate", "coverage", "accepted_accuracy"
    ]].to_string(index=False))

print("\nReport preview:")
print(report_path.read_text(encoding="utf-8")[:6000])
