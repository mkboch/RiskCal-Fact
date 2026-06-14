#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/finalize_feverous_from_saved_csvs_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FINALIZE FEVEROUS FROM SAVED CSVs START ===="
date

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_fever_loader
echo "python=$(command -v python)"
python --version

echo ""
echo "==== Install/verify Jinja2 and tabulate ===="
python - <<'PY'
missing = []
for pkg in ["jinja2", "tabulate"]:
    try:
        __import__(pkg)
        print(pkg, "OK")
    except Exception:
        print(pkg, "MISSING")
        missing.append(pkg)
raise SystemExit(7 if missing else 0)
PY

STATUS=$?
if [ "$STATUS" != "0" ]; then
    python -m pip install --no-cache-dir jinja2 tabulate
fi

echo ""
echo "==== Generate missing FEVEROUS report, metrics, and latex ===="
cat > scripts/finalize_feverous_from_saved_csvs.py <<'PY'
import json
from pathlib import Path
import pandas as pd

TABLE = Path("outputs/tables/feverous")
METRIC = Path("outputs/metrics/feverous")
FINAL = Path("outputs/final_report")
TEX = Path("outputs/latex_tables")
FIG = Path("outputs/figures")

for d in [TABLE, METRIC, FINAL, TEX, FIG]:
    d.mkdir(parents=True, exist_ok=True)

split_path = TABLE / "table_feverous_split_summary.csv"
mod_path = TABLE / "table_feverous_by_modality.csv"
chal_path = TABLE / "table_feverous_by_challenge.csv"
compact_path = TABLE / "table_feverous_structured_provenance_compact.csv"

split_df = pd.read_csv(split_path)
modality_df = pd.read_csv(mod_path)
challenge_df = pd.read_csv(chal_path)
compact_df = pd.read_csv(compact_path)

def write_latex_simple(df, path, caption, label):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].map(lambda x: f"{float(x):.3f}")
    body = out.to_latex(index=False, escape=True)
    text = "\\begin{table}[t]\n\\centering\n\\small\n"
    text += f"\\caption{{{caption}}}\n"
    text += f"\\label{{{label}}}\n"
    text += body
    text += "\\end{table}\n"
    Path(path).write_text(text, encoding="utf-8")

write_latex_simple(
    split_df,
    TEX / "table_feverous_split_summary.tex",
    "FEVEROUS structured-provenance split summary.",
    "tab:feverous_split_summary",
)

keep_cols = [
    "evidence_modality", "n", "verified", "refuted", "unsupported", "unlabeled",
    "provenance_complete_rate", "mean_content_ids", "mean_context_ids",
    "mean_unique_pages", "mean_graph_nodes", "mean_graph_edges"
]
modality_for_tex = modality_df[[c for c in keep_cols if c in modality_df.columns]].copy()

write_latex_simple(
    modality_for_tex,
    TEX / "table_feverous_structured_provenance.tex",
    "FEVEROUS evidence modality and provenance-graph characteristics.",
    "tab:feverous_structured_provenance",
)

labelled_rows = int(modality_df["n"].sum()) if "n" in modality_df.columns else 0

metrics = {
    "dataset": "FEVEROUS",
    "purpose": "structured provenance and evidence graph stress test",
    "scope_note": (
        "The Hugging Face FEVEROUS examples expose evidence IDs and context IDs. "
        "This experiment characterizes provenance structure, evidence modality, and graph complexity, "
        "rather than text-NLI accuracy over resolved table cells."
    ),
    "labelled_rows": labelled_rows,
    "split_summary": split_df.to_dict(orient="records"),
    "modality_summary": modality_df.to_dict(orient="records"),
    "challenge_summary_rows": len(challenge_df),
}
metric_path = METRIC / "feverous_structured_provenance_metrics.json"
metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

report_path = FINAL / "feverous_structured_provenance_report.md"
with report_path.open("w", encoding="utf-8") as f:
    f.write("# FEVEROUS Structured-Provenance Experiment\n\n")
    f.write("This experiment characterizes FEVEROUS as a structured provenance stress test for the provenance-completeness term P(c), evidence graph construction, and sentence/table/cell evidence modality.\n\n")

    f.write("## Scope Note\n\n")
    f.write("The Hugging Face FEVEROUS examples expose evidence and context identifiers. Therefore, this run reports provenance structure and graph characteristics rather than resolved table-cell-text NLI accuracy.\n\n")

    f.write("## Split Summary\n\n")
    f.write(split_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Evidence Modality Summary\n\n")
    f.write(modality_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Challenge Summary\n\n")
    f.write(challenge_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Paper Interpretation\n\n")
    f.write(
        "FEVEROUS motivates the explicit P(c) term in the verification rule. "
        "Evidence may be available as structured identifiers and graph links, even when plain text evidence is not directly available. "
        "This supports separating support S(c), contradiction K(c), rule consistency R(c), provenance completeness P(c), and calibrated risk rho(c). "
        "In the manuscript, FEVEROUS should be reported as a structured-provenance characterization rather than as another text-only NLI benchmark.\n"
    )

project_status = {
    "paper_experiment_status": "ready_for_manuscript_drafting",
    "completed": [
        "SciFact full labelled evaluation",
        "FEVER pilot evaluation",
        "FEVER full paper_dev evaluation with large sampled evidence corpus",
        "VitaminC evidence-given verification",
        "PubHealth evidence-given verification",
        "Climate-FEVER evidence-given verification",
        "FEVEROUS structured-provenance characterization",
        "Unified risk-coverage figures",
        "Unified LaTeX result tables",
    ],
    "core_equation": "verified iff S(c) >= tau_s and K(c) < tau_k and R(c)=1 and P(c)=1 and rho(c)<=alpha",
}
status_path = FINAL / "project_ready_for_paper_status.json"
status_path.write_text(json.dumps(project_status, indent=2), encoding="utf-8")

print("Saved:")
for p in [
    metric_path,
    report_path,
    TEX / "table_feverous_split_summary.tex",
    TEX / "table_feverous_structured_provenance.tex",
    status_path,
]:
    print(f"  {p} ({p.stat().st_size} bytes)")

print("")
print("FEVEROUS modality summary:")
print(modality_df.to_string(index=False))
PY

python scripts/finalize_feverous_from_saved_csvs.py

echo ""
echo "==== Confirm FEVEROUS success markers ===="
test -s outputs/final_report/feverous_structured_provenance_report.md && echo "OK report exists" || echo "MISSING report"
test -s outputs/tables/feverous/table_feverous_split_summary.csv && echo "OK split table exists" || echo "MISSING split table"
test -s outputs/metrics/feverous/feverous_structured_provenance_metrics.json && echo "OK metrics exists" || echo "MISSING metrics"
test -s outputs/latex_tables/table_feverous_structured_provenance.tex && echo "OK latex table exists" || echo "MISSING latex table"

echo ""
echo "==== Final FEVEROUS files ===="
find data/processed/feverous outputs/tables/feverous outputs/metrics/feverous outputs/final_report outputs/latex_tables \
  -maxdepth 1 -type f | grep -E "feverous|project_ready" | sort || true

echo ""
echo "==== FINALIZE FEVEROUS FROM SAVED CSVs END ===="
date
echo "Log saved to: $LOG"
