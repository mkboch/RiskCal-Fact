#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/install_matplotlib_and_rerun_feverous_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== INSTALL MATPLOTLIB + RERUN FEVEROUS START ===="
date

echo ""
echo "==== Activate loader env ===="
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_fever_loader
echo "python=$(command -v python)"
python --version

echo ""
echo "==== Install matplotlib if missing ===="
python - <<'PY'
try:
    import matplotlib
    print("matplotlib already installed:", matplotlib.__version__)
except Exception as e:
    print("matplotlib missing:", repr(e))
    raise SystemExit(7)
PY

STATUS=$?
if [ "$STATUS" != "0" ]; then
    python -m pip install --no-cache-dir matplotlib pandas numpy tabulate
fi

echo ""
echo "==== Verify imports ===="
python - <<'PY'
import matplotlib
import pandas
import numpy
import datasets
print("matplotlib:", matplotlib.__version__)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("datasets:", datasets.__version__)
PY

echo ""
echo "==== Rerun fixed FEVEROUS provenance script ===="
bash scripts/run_feverous_structured_provenance_fixed.sh || echo "WARNING: fixed FEVEROUS script returned nonzero status."

echo ""
echo "==== Check FEVEROUS outputs ===="
find data/processed/feverous outputs/tables/feverous outputs/metrics/feverous outputs/final_report outputs/latex_tables outputs/figures \
  -maxdepth 1 -type f | grep -E "feverous|project_ready" | sort || true

echo ""
echo "==== Confirm FEVEROUS success markers ===="
test -s outputs/final_report/feverous_structured_provenance_report.md && echo "OK report exists" || echo "MISSING report"
test -s outputs/tables/feverous/table_feverous_split_summary.csv && echo "OK split table exists" || echo "MISSING split table"
test -s outputs/metrics/feverous/feverous_structured_provenance_metrics.json && echo "OK metrics exists" || echo "MISSING metrics"
test -s outputs/latex_tables/table_feverous_structured_provenance.tex && echo "OK latex table exists" || echo "MISSING latex table"

echo ""
echo "==== INSTALL MATPLOTLIB + RERUN FEVEROUS END ===="
date
echo "Log saved to: $LOG"
