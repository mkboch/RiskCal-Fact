#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/final_adversarial_review_fixes_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FINAL ADVERSARIAL REVIEW FIXES START ===="
date

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_py310
echo "python=$(command -v python)"
python --version

mkdir -p outputs/final_report outputs/tables/review_hardening outputs/latex_tables scripts

cat > scripts/final_adversarial_review_fixes.py <<'PY'
import json
from pathlib import Path
import pandas as pd

FINAL = Path("outputs/final_report")
TABLE = Path("outputs/tables/review_hardening")
METRIC = Path("outputs/metrics/review_hardening")
TEX = Path("outputs/latex_tables")
for d in [FINAL, TABLE, METRIC, TEX]:
    d.mkdir(parents=True, exist_ok=True)

def load_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"Missing: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)

def fmt(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def main():
    # Extract Climate-FEVER strict count from available detailed summary.
    candidates = [
        METRIC / "fast_ablation_all.csv",
        METRIC / "evidence_given_risk_calibration_summary.csv",
        Path("outputs/metrics/evidence_given/evidence_given_risk_calibration_summary.csv"),
    ]

    rows = []
    for p in candidates:
        df = load_csv(p)
        if df.empty:
            continue

        cols = {c.lower(): c for c in df.columns}
        # Flexible dataset/rule/alpha filters.
        dataset_col = cols.get("dataset") or cols.get("Dataset".lower())
        rule_col = cols.get("rule") or cols.get("Rule".lower())
        alpha_col = cols.get("alpha") or cols.get("Alpha".lower())

        if dataset_col is None or rule_col is None or alpha_col is None:
            continue

        sub = df[
            (df[dataset_col].astype(str).str.lower() == "climate_fever") &
            (df[rule_col].astype(str).str.replace("ρ", "rho").str.lower().isin(["s+k+p+rho", "s+k+rho"])) &
            (df[alpha_col].astype(float).round(2) == 0.05)
        ].copy()

        if sub.empty:
            continue

        sub["source_file"] = str(p)
        rows.append(sub)

    if rows:
        climate = pd.concat(rows, ignore_index=True)
        # Prefer RoBERTa if model column exists.
        model_cols = [c for c in climate.columns if c.lower() == "model"]
        if model_cols:
            mcol = model_cols[0]
            rb = climate[climate[mcol].astype(str).str.contains("roberta|RoBERTa|ynie", case=False, regex=True)]
            if not rb.empty:
                climate = rb
        climate = climate.head(10)
    else:
        climate = pd.DataFrame()

    climate.to_csv(TABLE / "table_climate_fever_strict_verified_count.csv", index=False)

    # Make a compact human-readable table from whatever columns exist.
    compact_rows = []
    if not climate.empty:
        for _, r in climate.iterrows():
            compact_rows.append({
                "Dataset": r.get("dataset", r.get("Dataset", "climate_fever")),
                "Model": r.get("model", r.get("Model", "")),
                "Rule": r.get("rule", r.get("Rule", "")),
                "Alpha": r.get("alpha", r.get("Alpha", "")),
                "Macro-F1": r.get("macro_f1", r.get("Macro-F1", "")),
                "FVR": r.get("false_verification_rate", r.get("FVR", "")),
                "Coverage": r.get("coverage", r.get("Coverage", "")),
                "Num predicted verified": r.get("num_predicted_verified", r.get("num_verified_predictions", "")),
                "Num abstained": r.get("num_abstained", ""),
                "Source": r.get("source_file", ""),
            })
    compact = pd.DataFrame(compact_rows)
    compact.to_csv(TABLE / "table_climate_fever_strict_verified_count_compact.csv", index=False)

    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Climate-FEVER strict risk-gated operating point, including the number of predicted verified claims used to interpret zero false-verification rate.}\n"
    tex += "\\label{tab:climate_fever_strict_count}\n"
    if compact.empty:
        tex += "\\begin{tabular}{l}No matching row found.\\end{tabular}\n"
    else:
        tex_df = compact.copy()
        for c in ["Alpha", "Macro-F1", "FVR", "Coverage"]:
            if c in tex_df:
                tex_df[c] = tex_df[c].map(fmt)
        tex += tex_df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_climate_fever_strict_count.tex").write_text(tex, encoding="utf-8")

    note = FINAL / "final_adversarial_review_fixes.md"
    with note.open("w", encoding="utf-8") as f:
        f.write("# Final Adversarial Review Fixes\n\n")

        f.write("## 1. VitaminC S+K versus best utility setting\n\n")
        f.write("The manuscript should not present the VitaminC S+K ablation value and the VitaminC best utility value as if they are the same setting. The S+K ablation row is an ungated full-coverage diagnostic row. The evidence-given best utility setting is selected from the risk-calibrated operating points under the full experimental protocol. Add this table note: `S+K rows are ungated full-coverage ablations; best utility rows are selected operating points from the risk-calibrated protocol and are therefore not required to match the ungated ablation values.`\n\n")

        f.write("## 2. Macro-F1 and abstention convention\n\n")
        f.write("Use this precise convention in the paper: `Macro-F1 is computed over the original task labels, with abstentions treated as non-matching predictions rather than as a fourth task label. Thus, stricter risk thresholds can mechanically depress macro-F1 by increasing abstentions. Coverage and accepted accuracy are therefore reported alongside macro-F1, and cross-alpha macro-F1 comparisons should be interpreted as selective operating-point comparisons rather than pure classifier-quality comparisons.`\n\n")

        f.write("## 3. Oracle S+K interpretation\n\n")
        f.write("Add this sentence to the oracle section: `The S+K improvement from macro-F1 0.522 to 0.916 reflects improved discrimination between refuted and unsupported claims, not a reduction in false verifications. This is consistent with the stable oracle FVR and reflects the asymmetric role of contradiction signals in three-class verification.`\n\n")

        f.write("## 4. Risk-coverage curve wording\n\n")
        f.write("Avoid describing the risk-coverage curves as smooth continuous curves. Use: `Across the evaluated alpha grid, the curves expose the operating-point frontier available to a practitioner; they should not be interpreted as guarantees of monotonic improvement under distribution shift.` State that alpha is evaluated over a discrete grid, e.g., 0.05 to 0.30.\n\n")

        f.write("## 5. Climate-FEVER zero-FVR count\n\n")
        if compact.empty:
            f.write("No matching Climate-FEVER strict row was found in the searched CSV files. Check detailed result files before writing the final count.\n\n")
        else:
            f.write("Use the following extracted strict Climate-FEVER row to report the number of predicted verified claims for the zero-FVR setting:\n\n")
            f.write(compact.to_markdown(index=False))
            f.write("\n\n")

        f.write("## 6. S/K correlation paragraph for Method section\n\n")
        f.write("Insert this in the Method section, not only in Limitations: `In our implementation, S(c) and K(c) are derived from the same NLI model and are therefore not statistically independent. We separate them operationally rather than statistically: the verifier can require a high support score while independently rejecting claims whose contradiction score exceeds a threshold. This asymmetric thresholding is a design choice, not a claim of independence.`\n\n")

        f.write("## 7. FEVEROUS abstract wording\n\n")
        f.write("Use this abstract-safe wording: `A structured-provenance analysis of FEVEROUS shows that table-only and mixed sentence-table evidence induces evidence graphs roughly 27--30 times larger than sentence-only evidence, motivating provenance-aware extensions.`\n\n")

        f.write("## 8. R(c) limitation wording\n\n")
        f.write("Use this wording in Limitations: `R(c) is designed as a domain-specific constraint gate. No benchmark-specific symbolic rules were defined for the current experiments, so R(c) does not affect the reported results. Future work can integrate domain rules through this gate without modifying the S,K,rho core.`\n\n")

    status = {
        "status": "completed",
        "outputs": [
            "outputs/final_report/final_adversarial_review_fixes.md",
            "outputs/tables/review_hardening/table_climate_fever_strict_verified_count.csv",
            "outputs/tables/review_hardening/table_climate_fever_strict_verified_count_compact.csv",
            "outputs/latex_tables/table_climate_fever_strict_count.tex",
        ],
    }
    (FINAL / "final_adversarial_review_fixes_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("Climate-FEVER strict compact count:")
    if compact.empty:
        print("No matching row found.")
    else:
        print(compact.to_string(index=False))

    print("\nSaved final note:", note)
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
PY

python scripts/final_adversarial_review_fixes.py

echo ""
echo "==== Final adversarial review fix files ===="
find outputs/final_report outputs/tables/review_hardening outputs/latex_tables \
  -maxdepth 1 -type f | grep -E "final_adversarial|climate_fever_strict" | sort || true

echo ""
echo "==== FINAL ADVERSARIAL REVIEW FIXES END ===="
date
echo "Log saved to: $LOG"
