from pathlib import Path
import json
import pandas as pd

METRIC_DIR = Path("outputs/metrics/scifact")
TABLE_DIR = Path("outputs/tables/scifact")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

rows = []

# BM25 retrieval.
for split in ["train", "dev"]:
    p = METRIC_DIR / f"bm25_top10_{split}_metrics.json"
    if p.exists():
        m = read_json(p)
        rows.append({
            "table": "retrieval",
            "dataset": "SciFact",
            "split": split,
            "method": "BM25 top-10",
            "metric": "evidence_doc_recall@1",
            "value": m.get("evidence_doc_recall@1"),
        })
        rows.append({
            "table": "retrieval",
            "dataset": "SciFact",
            "split": split,
            "method": "BM25 top-10",
            "metric": "evidence_doc_recall@3",
            "value": m.get("evidence_doc_recall@3"),
        })
        rows.append({
            "table": "retrieval",
            "dataset": "SciFact",
            "split": split,
            "method": "BM25 top-10",
            "metric": "evidence_doc_recall@5",
            "value": m.get("evidence_doc_recall@5"),
        })
        rows.append({
            "table": "retrieval",
            "dataset": "SciFact",
            "split": split,
            "method": "BM25 top-10",
            "metric": "evidence_doc_recall@10",
            "value": m.get("evidence_doc_recall@10"),
        })

# Proper NLI eval.
proper_summary = METRIC_DIR / "proper_eval_selected_nli_summary.csv"
if proper_summary.exists():
    df = pd.read_csv(proper_summary)
    for _, r in df.iterrows():
        rows.append({
            "table": "verification",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']}",
            "metric": "macro_f1",
            "value": r["macro_f1_all"],
        })
        rows.append({
            "table": "verification",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']}",
            "metric": "accuracy",
            "value": r["accuracy_all"],
        })
        rows.append({
            "table": "verification",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']}",
            "metric": "false_verification_rate",
            "value": r["false_verification_rate"],
        })
        rows.append({
            "table": "verification",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']}",
            "metric": "coverage",
            "value": r["coverage"],
        })

# Proper split risk calibration.
risk_summary = METRIC_DIR / "proper_split_risk_calibration_summary.csv"
if risk_summary.exists():
    df = pd.read_csv(risk_summary)
    for _, r in df.iterrows():
        rows.append({
            "table": "risk_calibration",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']} | alpha={r['alpha']}",
            "metric": "macro_f1",
            "value": r["macro_f1_all"],
        })
        rows.append({
            "table": "risk_calibration",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']} | alpha={r['alpha']}",
            "metric": "false_verification_rate",
            "value": r["false_verification_rate"],
        })
        rows.append({
            "table": "risk_calibration",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']} | alpha={r['alpha']}",
            "metric": "coverage",
            "value": r["coverage"],
        })
        rows.append({
            "table": "risk_calibration",
            "dataset": "SciFact",
            "split": "dev",
            "method": f"{r['model']} | {r['rule']} | alpha={r['alpha']}",
            "metric": "accepted_accuracy",
            "value": r["accepted_accuracy"],
        })

long_df = pd.DataFrame(rows)
long_path = TABLE_DIR / "scifact_all_metrics_long.csv"
long_df.to_csv(long_path, index=False)

# Retrieval compact.
retrieval_df = long_df[long_df["table"] == "retrieval"].pivot_table(
    index=["dataset", "split", "method"],
    columns="metric",
    values="value"
).reset_index()
retrieval_path = TABLE_DIR / "table_scifact_retrieval.csv"
retrieval_df.to_csv(retrieval_path, index=False)

# Main verification compact.
if proper_summary.exists():
    main_df = pd.read_csv(proper_summary)
    keep = [
        "model", "rule", "accuracy_all", "macro_f1_all", "verified_f1",
        "refuted_f1", "unsupported_f1", "false_verification_rate",
        "false_refuted_rate", "coverage", "accepted_accuracy",
        "num_predicted_verified", "num_predicted_refuted", "num_abstained"
    ]
    main_df = main_df[[c for c in keep if c in main_df.columns]].copy()
    main_df = main_df.sort_values(["macro_f1_all", "false_verification_rate"], ascending=[False, True])
    main_path = TABLE_DIR / "table_scifact_main_verification.csv"
    main_df.to_csv(main_path, index=False)

# Risk calibration compact.
if risk_summary.exists():
    risk_df = pd.read_csv(risk_summary)
    keep = [
        "model", "rule", "alpha", "macro_f1_all", "false_verification_rate",
        "num_predicted_verified", "num_abstained", "coverage",
        "accepted_accuracy", "cal_retained_fvr", "cal_verified_retention",
        "dev_base_macro_f1", "dev_base_false_verification_rate"
    ]
    risk_df = risk_df[[c for c in keep if c in risk_df.columns]].copy()
    risk_df = risk_df.sort_values(["model", "rule", "alpha"])
    risk_path = TABLE_DIR / "table_scifact_risk_calibration.csv"
    risk_df.to_csv(risk_path, index=False)

print("Saved:")
for p in sorted(TABLE_DIR.glob("*.csv")):
    print(f"  {p} ({p.stat().st_size} bytes)")

print("\nRetrieval table:")
print(retrieval_df.to_string(index=False))

if proper_summary.exists():
    print("\nMain verification top rows:")
    print(main_df.head(10).to_string(index=False))

if risk_summary.exists():
    print("\nRisk calibration selected rows:")
    selected = risk_df[
        ((risk_df["model"].str.contains("ynie")) & (risk_df["rule"] == "margin_support_refute")) |
        ((risk_df["model"].str.contains("facebook")) & (risk_df["rule"] == "margin_support_refute"))
    ]
    print(selected.to_string(index=False))
