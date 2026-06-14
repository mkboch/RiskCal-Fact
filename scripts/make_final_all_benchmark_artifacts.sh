#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/make_final_all_benchmark_artifacts_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FINAL ALL-BENCHMARK ARTIFACTS START ===="
date
echo "PROJECT_DIR=$PROJECT_DIR"
echo "LOG=$LOG"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_py310
python --version

mkdir -p scripts outputs/final_report outputs/figures outputs/latex_tables

cat > scripts/make_final_all_benchmark_artifacts.py <<'PY'
from pathlib import Path
import json
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FINAL = Path("outputs/final_report")
FIG = Path("outputs/figures")
TEX = Path("outputs/latex_tables")
for d in [FINAL, FIG, TEX]:
    d.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p)
    return pd.DataFrame()

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
        "verified_safe": "Verified-safe",
        "winner_takes_all": "Winner-takes-all",
        "risk_abstain": "Risk abstain",
    }.get(str(x), str(x))

def clean_dataset(x):
    return {
        "vitaminc": "VitaminC",
        "pubhealth": "PubHealth",
        "climate_fever": "Climate-FEVER",
    }.get(str(x), str(x))

all_best_rows = []
all_risk_rows = []

# 1. Existing SciFact + FEVER-pilot clean best.
clean_best = read_csv("outputs/final_report/clean_table_best_results.csv")
if not clean_best.empty:
    for _, r in clean_best.iterrows():
        all_best_rows.append({
            "Dataset": r.get("Dataset"),
            "Scope": r.get("Selection", ""),
            "Model": r.get("Model"),
            "Rule": r.get("Rule"),
            "Alpha": r.get("Alpha", ""),
            "Macro-F1": r.get("Macro-F1"),
            "FVR": r.get("FVR"),
            "Coverage": r.get("Coverage"),
            "Accepted Acc.": r.get("Accepted Acc."),
        })

# 2. FEVER full-dev.
fv_full = read_csv("outputs/tables/fever/table_fever_fast_full_dev_risk_calibration.csv")
if not fv_full.empty:
    tmp = fv_full.copy()
    tmp["Dataset"] = "FEVER-full-dev"
    for _, r in tmp.iterrows():
        all_risk_rows.append({
            "Dataset": "FEVER-full-dev",
            "Model": short_model(r.get("model")),
            "Rule": clean_rule(r.get("rule")),
            "Alpha": fmt(r.get("alpha"), 2),
            "Macro-F1": float(r.get("macro_f1_all")),
            "Accuracy": float(r.get("accuracy_all")),
            "FVR": float(r.get("false_verification_rate")),
            "FRR": float(r.get("false_refuted_rate")),
            "Coverage": float(r.get("coverage")),
            "Accepted Acc.": float(r.get("accepted_accuracy")),
            "Verified": int(r.get("num_predicted_verified")),
            "Refuted": int(r.get("num_predicted_refuted")),
            "Abstain": int(r.get("num_abstained")),
        })

    best = tmp.sort_values("macro_f1_all", ascending=False).head(1)
    low = tmp[tmp["coverage"] >= 0.75].sort_values(["false_verification_rate", "macro_f1_all"], ascending=[True, False]).head(1)

    for scope, sub in [("best macro-F1", best), ("lowest FVR, coverage >= 0.75", low)]:
        if len(sub):
            r = sub.iloc[0]
            all_best_rows.append({
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

# 3. Evidence-given.
eg_best = read_csv("outputs/tables/evidence_given/table_evidence_given_best_results.csv")
if not eg_best.empty:
    for _, r in eg_best.iterrows():
        all_best_rows.append({
            "Dataset": clean_dataset(r.get("dataset")),
            "Scope": str(r.get("selection")).replace("_", " "),
            "Model": short_model(r.get("model")),
            "Rule": clean_rule(r.get("rule")),
            "Alpha": fmt(r.get("alpha"), 2),
            "Macro-F1": fmt(r.get("macro_f1")),
            "FVR": fmt(r.get("fvr")),
            "Coverage": fmt(r.get("coverage")),
            "Accepted Acc.": fmt(r.get("accepted_accuracy")),
        })

eg_full = read_csv("outputs/tables/evidence_given/table_evidence_given_risk_calibration.csv")
if not eg_full.empty:
    for _, r in eg_full.iterrows():
        all_risk_rows.append({
            "Dataset": clean_dataset(r.get("dataset")),
            "Model": short_model(r.get("model")),
            "Rule": clean_rule(r.get("rule")),
            "Alpha": fmt(r.get("alpha"), 2),
            "Macro-F1": float(r.get("macro_f1_all")),
            "Accuracy": float(r.get("accuracy_all")),
            "FVR": float(r.get("false_verification_rate")),
            "FRR": float(r.get("false_refuted_rate")),
            "Coverage": float(r.get("coverage")),
            "Accepted Acc.": float(r.get("accepted_accuracy")),
            "Verified": int(r.get("num_predicted_verified")),
            "Refuted": int(r.get("num_predicted_refuted")),
            "Abstain": int(r.get("num_abstained")),
        })

best_df = pd.DataFrame(all_best_rows)
risk_df = pd.DataFrame(all_risk_rows)

best_path = FINAL / "all_benchmark_best_results.csv"
risk_path = FINAL / "all_benchmark_risk_calibration_long.csv"
best_df.to_csv(best_path, index=False)
risk_df.to_csv(risk_path, index=False)

# Selected risk rows for paper: alpha 0.05, 0.10, 0.30 where available.
selected = risk_df[risk_df["Alpha"].isin(["0.05", "0.10", "0.30"])].copy()
selected_path = FINAL / "all_benchmark_selected_risk_rows.csv"
selected.to_csv(selected_path, index=False)

# Dataset-level status.
status = {
    "completed_benchmark_settings": [
        "SciFact full official labelled dev",
        "FEVER-pilot",
        "FEVER-full-dev with full paper_dev and large sampled sentence corpus",
        "VitaminC evidence-given",
        "PubHealth evidence-given",
        "Climate-FEVER evidence-given",
    ],
    "remaining_major_dataset": "FEVEROUS structured evidence",
    "notes": [
        "SciFact official test is unlabeled; dev is used for labelled evaluation.",
        "FEVER-full-dev uses all paper_dev claims but not exhaustive all-Wikipedia retrieval.",
        "Evidence-given datasets test support/refute/risk calibration independent of corpus retrieval.",
    ],
}
status_path = FINAL / "all_benchmark_status.json"
status_path.write_text(json.dumps(status, indent=2), encoding="utf-8")

# LaTeX tables.
def to_latex_table(df, path, caption, label, max_rows=None):
    if max_rows is not None:
        df = df.head(max_rows)
    tex = df.to_latex(index=False, escape=True, longtable=False)
    tex = tex.replace("\\begin{tabular}", f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
    Path(path).write_text(tex, encoding="utf-8")

paper_cols = ["Dataset", "Scope", "Model", "Rule", "Alpha", "Macro-F1", "FVR", "Coverage", "Accepted Acc."]
to_latex_table(
    best_df[paper_cols],
    TEX / "table_all_benchmark_best_results.tex",
    "Best and low-risk operating points across benchmark settings.",
    "tab:all_benchmark_best_results",
)

risk_cols = ["Dataset", "Model", "Rule", "Alpha", "Macro-F1", "FVR", "Coverage", "Accepted Acc.", "Verified", "Abstain"]
if not selected.empty:
    to_latex_table(
        selected[risk_cols],
        TEX / "table_all_benchmark_selected_risk.tex",
        "Selected risk-calibrated operating points across datasets.",
        "tab:all_benchmark_selected_risk",
    )

# Figures: risk vs coverage and macro-F1 vs FVR.
if not risk_df.empty:
    plot_df = risk_df.copy()

    plt.figure(figsize=(8, 5))
    for ds, sub in plot_df.groupby("Dataset"):
        plt.plot(sub["Coverage"], sub["FVR"], marker="o", linestyle="", label=ds)
    plt.xlabel("Coverage")
    plt.ylabel("False-verification rate")
    plt.title("Risk-coverage tradeoff across benchmark settings")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig1 = FIG / "fig_risk_coverage_all_benchmarks.png"
    plt.savefig(fig1, dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    for ds, sub in plot_df.groupby("Dataset"):
        plt.plot(sub["FVR"], sub["Macro-F1"], marker="o", linestyle="", label=ds)
    plt.xlabel("False-verification rate")
    plt.ylabel("Macro-F1")
    plt.title("Utility-risk tradeoff across benchmark settings")
    plt.legend(fontsize=7)
    plt.tight_layout()
    fig2 = FIG / "fig_utility_risk_all_benchmarks.png"
    plt.savefig(fig2, dpi=300)
    plt.close()
else:
    fig1 = fig2 = None

# Markdown report.
report_path = FINAL / "all_benchmark_final_report.md"
with report_path.open("w", encoding="utf-8") as f:
    f.write("# Final All-Benchmark Experimental Report\n\n")

    f.write("## Completed Benchmark Settings\n\n")
    for item in status["completed_benchmark_settings"]:
        f.write(f"- {item}\n")
    f.write("\n")

    f.write("## Scope Notes\n\n")
    for note in status["notes"]:
        f.write(f"- {note}\n")
    f.write("\n")

    f.write("## Best and Low-Risk Operating Points\n\n")
    f.write(best_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Selected Risk Calibration Rows\n\n")
    if not selected.empty:
        f.write(selected.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Key Interpretation for Paper\n\n")
    f.write(
        "Across six benchmark settings, the method exposes the tradeoff between verification utility "
        "and false-verification risk. Evidence-given benchmarks isolate the verifier and risk gate, "
        "whereas SciFact and FEVER include retrieval and therefore also measure provenance/evidence acquisition. "
        "Low-risk operating points reduce false verification, often at the cost of coverage and macro-F1. "
        "This supports the paper's methodological claim that verification should be reported as a calibrated "
        "risk-controlled decision rather than a single accuracy value.\n"
    )

print("Saved:")
for p in [best_path, risk_path, selected_path, status_path, report_path]:
    print(f"  {p} ({p.stat().st_size} bytes)")
for p in [TEX / "table_all_benchmark_best_results.tex", TEX / "table_all_benchmark_selected_risk.tex"]:
    if p.exists():
        print(f"  {p} ({p.stat().st_size} bytes)")
for p in [FIG / "fig_risk_coverage_all_benchmarks.png", FIG / "fig_utility_risk_all_benchmarks.png"]:
    if p and p.exists():
        print(f"  {p} ({p.stat().st_size} bytes)")

print("\nBest results:")
print(best_df.to_string(index=False))

print("\nSelected risk rows:")
print(selected.to_string(index=False) if not selected.empty else "EMPTY")
PY

python scripts/make_final_all_benchmark_artifacts.py

echo ""
echo "==== Show final files ===="
find outputs/final_report outputs/latex_tables outputs/figures -maxdepth 1 -type f | sort | tail -n 120

echo ""
echo "==== FINAL ALL-BENCHMARK ARTIFACTS END ===="
date
echo "Log saved to: $LOG"
