#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/final_prewrite_review_tables_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FINAL PREWRITE REVIEW TABLES START ===="
date

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_py310
echo "python=$(command -v python)"
python --version

mkdir -p outputs/final_report outputs/tables/review_hardening outputs/latex_tables scripts

cat > scripts/make_final_prewrite_review_tables.py <<'PY'
import json
from pathlib import Path
import pandas as pd
import numpy as np

FINAL = Path("outputs/final_report")
TABLE = Path("outputs/tables/review_hardening")
METRIC = Path("outputs/metrics/review_hardening")
TEX = Path("outputs/latex_tables")

for d in [FINAL, TABLE, METRIC, TEX]:
    d.mkdir(parents=True, exist_ok=True)

def short_model(x):
    x = str(x)
    if "RoBERTa" in x or "ynie/roberta" in x:
        return "RoBERTa-large NLI"
    if "BART" in x or "facebook/bart" in x:
        return "BART-large MNLI"
    return x

def fmt_float(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return ""

def main():
    fast_ablation_path = TABLE / "table_fast_ablation.csv"
    skrho_path = TABLE / "table_explicit_s_k_rho_ablation.csv"

    fast = pd.read_csv(fast_ablation_path)
    skrho = pd.read_csv(skrho_path)

    # Normalize fast ablation table.
    fast_rows = []
    for _, r in fast.iterrows():
        dataset = r["Dataset"]
        rule = r["Rule"]
        if rule not in ["S", "S+K", "S+K+P+rho"]:
            continue

        fast_rows.append({
            "Dataset": dataset,
            "Model": short_model(r["Model"]),
            "Rule": rule,
            "Alpha": r.get("Alpha", ""),
            "Macro-F1": float(r["Macro-F1"]),
            "FVR": float(r["FVR"]),
            "Coverage": float(r["Coverage"]),
            "Accepted Acc.": float(r["Accepted Acc."]),
        })

    # S+K+rho explicit table.
    skrho_rows = []
    for _, r in skrho.iterrows():
        skrho_rows.append({
            "Dataset": r["Dataset"],
            "Model": short_model(r["Model"]),
            "Rule": "S+K+rho",
            "Alpha": r["Alpha"],
            "Macro-F1": float(r["Macro-F1"]),
            "FVR": float(r["FVR"]),
            "Coverage": float(r["Coverage"]),
            "Accepted Acc.": float(r["Accepted Acc."]),
        })

    unified = pd.DataFrame(fast_rows + skrho_rows)

    # Keep order.
    dataset_order = {
        "fever_full_dev": 0,
        "vitaminc": 1,
        "pubhealth": 2,
        "climate_fever": 3,
    }
    rule_order = {
        "S": 0,
        "S+K": 1,
        "S+K+rho": 2,
        "S+K+P+rho": 3,
    }

    unified["dataset_order"] = unified["Dataset"].map(dataset_order).fillna(99)
    unified["rule_order"] = unified["Rule"].map(rule_order).fillna(99)
    unified = unified.sort_values(["dataset_order", "rule_order"]).drop(columns=["dataset_order", "rule_order"])

    unified.to_csv(TABLE / "table_unified_ablation_final.csv", index=False)

    tex_df = unified.copy()
    for c in ["Macro-F1", "FVR", "Coverage", "Accepted Acc."]:
        tex_df[c] = tex_df[c].map(fmt_float)

    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Unified ablation of support, contradiction, calibrated risk gating, and modular provenance gating. Adding risk gating generally reduces false-verification risk at the cost of coverage or macro-F1.}\n"
    tex += "\\label{tab:unified_ablation_final}\n"
    tex += tex_df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_unified_ablation_final.tex").write_text(tex, encoding="utf-8")

    # Oracle note.
    oracle_path = TABLE / "table_fever_oracle_flat_diagnostic.csv"
    oracle = pd.read_csv(oracle_path) if oracle_path.exists() else pd.DataFrame()

    # Risk-coverage curve summary from all risk rows.
    all_ablation_path = METRIC / "fast_ablation_all.csv"
    all_ablation = pd.read_csv(all_ablation_path)

    risk = all_ablation[all_ablation["rule"] == "S+K+P+rho"].copy()
    curve_rows = []

    for ds in ["fever_full_dev", "vitaminc", "pubhealth", "climate_fever"]:
        sub = risk[risk["dataset"] == ds].copy()
        if sub.empty:
            continue
        if any(sub["model"].astype(str).str.contains("ynie/roberta")):
            sub = sub[sub["model"].astype(str).str.contains("ynie/roberta")]
        sub = sub.sort_values("alpha")

        curve_rows.append({
            "Dataset": ds,
            "Alpha range": f"{sub['alpha'].min():.2f}-{sub['alpha'].max():.2f}",
            "FVR min": sub["false_verification_rate"].min(),
            "FVR max": sub["false_verification_rate"].max(),
            "Coverage min": sub["coverage"].min(),
            "Coverage max": sub["coverage"].max(),
            "Macro-F1 min": sub["macro_f1"].min(),
            "Macro-F1 max": sub["macro_f1"].max(),
            "Visual interpretation": (
                "Risk gating exposes a coverage/utility tradeoff; lower alpha generally lowers accepted verified risk but may reduce coverage."
            )
        })

    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(TABLE / "table_risk_coverage_curve_summary.csv", index=False)

    tex_curve = curve_df.copy()
    for c in ["FVR min", "FVR max", "Coverage min", "Coverage max", "Macro-F1 min", "Macro-F1 max"]:
        tex_curve[c] = tex_curve[c].map(fmt_float)
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Summary of risk-coverage curve behavior across representative benchmarks.}\n"
    tex += "\\label{tab:risk_coverage_curve_summary}\n"
    tex += tex_curve.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_risk_coverage_curve_summary.tex").write_text(tex, encoding="utf-8")

    # Write final prewriting note.
    note_path = FINAL / "final_prewrite_review_resolution.md"
    with note_path.open("w", encoding="utf-8") as f:
        f.write("# Final Prewriting Review Resolution\n\n")

        f.write("## 1. Unified Ablation Interpretation\n\n")
        f.write("The unified ablation table now reports S, S+K, S+K+rho, and S+K+P+rho with macro-F1, FVR, coverage, and accepted accuracy. This resolves the apparent inconsistency between the higher S+K macro-F1 values and lower S+K+rho macro-F1 values. The drop in macro-F1 after adding rho is expected: risk gating is a selective-verification operation that can reduce FVR by abstaining from uncertain verified decisions, often at the cost of coverage and utility.\n\n")
        f.write(unified.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 2. Oracle FVR Explanation\n\n")
        f.write("The FEVER oracle diagnostic is an upper-bound diagnostic, not a deployable setting. It uses gold evidence identifiers provided by FEVER annotation to isolate verifier behavior from retrieval errors. The oracle split is balanced across verified, refuted, and unsupported labels. The stable oracle FVR of approximately 0.022 across S, S+K, S+K+P, and S+K+P+rho indicates that, with gold evidence, the NLI model rarely assigns high verified support to non-verified claims. Therefore FVR is already near its floor under S-only, while contradiction modeling mainly improves the separation of refuted and unsupported decisions, raising macro-F1 from 0.522 to 0.916.\n\n")
        if not oracle.empty:
            f.write(oracle.to_markdown(index=False))
            f.write("\n\n")

        f.write("## 3. Macro-F1 and Abstention Convention\n\n")
        f.write("The paper should state the following convention: macro-F1 is computed over the prediction outputs produced by each operating point, while coverage is reported separately as the fraction of examples that are not abstained. Accepted accuracy is computed only over non-abstained examples. Therefore, macro-F1, coverage, FVR, and accepted accuracy must be interpreted jointly. In selective-verification settings, a lower macro-F1 at a lower alpha can be an expected consequence of abstaining from uncertain verified decisions.\n\n")

        f.write("## 4. PubHealth Coverage Drop Explanation\n\n")
        f.write("PubHealth has a much lower coverage at alpha=0.10 than VitaminC or Climate-FEVER. This should be described separately from high FVR. The low PubHealth coverage suggests that calibrated risk estimates are high for many PubHealth claims, likely because the NLI signal is noisy over longer multi-unit evidence. PubHealth dev examples have about 4.5 evidence units on average and high mean support scores, which may make isolated evidence units appear supportive even when the final claim label is not verified. This motivates the limitation that domain-specific, multi-sentence evidence aggregation is needed for public-health verification.\n\n")

        f.write("## 5. Risk-Coverage Curve Description\n\n")
        f.write("The risk-coverage and utility-risk curves should be described as operating-point curves rather than monotonic guarantees. Across representative benchmarks, changing alpha moves the verifier along a utility-risk-coverage frontier. VitaminC shows a relatively smooth tradeoff: stricter alpha reduces FVR while coverage decreases moderately. Climate-FEVER shows a strong low-risk operating point, including zero FVR at strict alpha with non-trivial coverage. PubHealth is the outlier: strict risk gating causes a large coverage drop, consistent with noisier evidence aggregation. FEVER-full-dev reflects retrieval limitations, with risk gating reducing coverage without always improving FVR as strongly as in evidence-given settings.\n\n")
        f.write(curve_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## 6. Calibration Versus Conformal Prediction\n\n")
        f.write("The related work and theory sections should explicitly distinguish this calibration from conformal prediction. Suggested wording: Unlike split conformal prediction, which provides distribution-free marginal coverage guarantees, our calibration procedure estimates false-verification risk on held-out calibration data and selects operating points under a dataset-exchangeability assumption. This is a weaker guarantee but matches the goal of comparing utility-risk-coverage behavior across verification settings.\n\n")

        f.write("## 7. Final Writing Decisions\n\n")
        f.write("- Use the title: Risk-Calibrated Evidence-Constrained Verification of Factual Claims.\n")
        f.write("- Present S, K, and rho as the empirically active core.\n")
        f.write("- Present P and R as modular domain-specific gates.\n")
        f.write("- Use FEVEROUS to motivate P, not to claim P improves NLI accuracy.\n")
        f.write("- Use the FEVER oracle diagnostic as an upper-bound verifier diagnostic, not a deployable result.\n")
        f.write("- Keep contribution/finding separation: methodological contributions are the framework, S/K separation, and operating-point analysis; empirical findings are ablation, oracle, risk curves, and FEVEROUS characterization.\n")

    status = {
        "status": "completed",
        "outputs": [
            "outputs/tables/review_hardening/table_unified_ablation_final.csv",
            "outputs/latex_tables/table_unified_ablation_final.tex",
            "outputs/tables/review_hardening/table_risk_coverage_curve_summary.csv",
            "outputs/latex_tables/table_risk_coverage_curve_summary.tex",
            "outputs/final_report/final_prewrite_review_resolution.md",
        ],
    }
    (FINAL / "final_prewrite_review_resolution_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("Unified ablation final:")
    print(unified.to_string(index=False))
    print("\nRisk-coverage curve summary:")
    print(curve_df.to_string(index=False))
    print("\nSaved:", note_path)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
PY

python scripts/make_final_prewrite_review_tables.py

echo ""
echo "==== Final prewrite review files ===="
find outputs/final_report outputs/tables/review_hardening outputs/latex_tables \
  -maxdepth 1 -type f | grep -E "final_prewrite|unified_ablation_final|risk_coverage_curve_summary" | sort || true

echo ""
echo "==== FINAL PREWRITE REVIEW TABLES END ===="
date
echo "Log saved to: $LOG"
