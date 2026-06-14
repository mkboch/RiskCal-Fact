#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/feverous_now_selfcontained_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FEVEROUS NOW SELF-CONTAINED START ===="
date

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_fever_loader
echo "python=$(command -v python)"
python --version

mkdir -p data/processed/feverous outputs/metrics/feverous outputs/tables/feverous outputs/final_report outputs/latex_tables outputs/figures scripts

echo ""
echo "==== Verify required imports ===="
python - <<'PY'
import matplotlib, pandas, numpy, datasets, tabulate
print("matplotlib:", matplotlib.__version__)
print("pandas:", pandas.__version__)
print("numpy:", numpy.__version__)
print("datasets:", datasets.__version__)
print("tabulate OK")
PY

echo ""
echo "==== Write fixed FEVEROUS provenance Python script ===="
cat > scripts/run_feverous_structured_provenance_fixed.py <<'PY'
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

PROC = Path("data/processed/feverous")
METRIC = Path("outputs/metrics/feverous")
TABLE = Path("outputs/tables/feverous")
FINAL = Path("outputs/final_report")
TEX = Path("outputs/latex_tables")
FIG = Path("outputs/figures")

for d in [PROC, METRIC, TABLE, FINAL, TEX, FIG]:
    d.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def label_to_text(raw):
    # Observed HF FEVEROUS: train/validation labels are numeric; test label is -1.
    try:
        x = int(raw)
        return {1: "verified", 0: "refuted", 2: "unsupported", -1: "unlabeled"}.get(x, str(raw))
    except Exception:
        s = str(raw).strip().lower().replace("_", " ").replace("-", " ")
        if s in {"supports", "support", "supported"}:
            return "verified"
        if s in {"refutes", "refute", "refuted"}:
            return "refuted"
        if s in {"not enough info", "nei"}:
            return "unsupported"
        return s

def evidence_type(x):
    x = str(x)
    if "_header_cell_" in x or "_cell_" in x:
        return "table_cell"
    if "_sentence_" in x:
        return "sentence"
    if "_title" in x:
        return "title"
    if "_section_" in x:
        return "section"
    return "other"

def page_from_id(x):
    x = str(x)
    for pat in ["_header_cell_", "_sentence_", "_cell_", "_section_", "_title"]:
        if pat in x:
            return x.split(pat)[0]
    return x

def flatten_evidence(evidence):
    content_ids = []
    context_ids = []
    for ev in evidence or []:
        if not isinstance(ev, dict):
            continue
        content = ev.get("content") or []
        context = ev.get("context") or []

        for c in content:
            content_ids.append(str(c))

        for ctx in context:
            if isinstance(ctx, list):
                for c in ctx:
                    context_ids.append(str(c))
            else:
                context_ids.append(str(ctx))

    return content_ids, context_ids

def summarize_example(ex, split):
    content_ids, context_ids = flatten_evidence(ex.get("evidence", []))

    content_types = [evidence_type(x) for x in content_ids]
    context_types = [evidence_type(x) for x in context_ids]
    pages = sorted(set(page_from_id(x) for x in content_ids + context_ids if x))

    has_sentence = any(t == "sentence" for t in content_types)
    has_cell = any(t == "table_cell" for t in content_types)

    if has_sentence and has_cell:
        modality = "mixed_sentence_table"
    elif has_cell:
        modality = "table_only"
    elif has_sentence:
        modality = "sentence_only"
    elif len(content_ids) == 0:
        modality = "no_evidence"
    else:
        modality = "other"

    graph_nodes = 1 + len(set(content_ids)) + len(set(context_ids))
    graph_edges = len(content_ids) + sum(
        1 for c in content_ids for ctx in context_ids
        if page_from_id(c) == page_from_id(ctx)
    )

    provenance_complete = int(len(content_ids) > 0 and all(str(x).strip() for x in content_ids))

    return {
        "id": str(ex.get("id")),
        "split": split,
        "claim": str(ex.get("claim", "")),
        "label": label_to_text(ex.get("label")),
        "raw_label": ex.get("label"),
        "challenge": str(ex.get("challenge", "")),
        "expected_challenge": str(ex.get("expected_challenge", "")),
        "evidence_modality": modality,
        "provenance_complete": provenance_complete,
        "num_evidence_bundles": len(ex.get("evidence", []) or []),
        "num_content_ids": len(content_ids),
        "num_context_ids": len(context_ids),
        "num_unique_pages": len(pages),
        "num_sentence_ids": sum(1 for t in content_types if t == "sentence"),
        "num_table_cell_ids": sum(1 for t in content_types if t == "table_cell"),
        "graph_num_nodes": graph_nodes,
        "graph_num_edges": graph_edges,
        "content_type_counts": dict(Counter(content_types)),
        "context_type_counts": dict(Counter(context_types)),
        "pages": pages,
        "content_ids": content_ids,
        "context_ids": context_ids,
    }

def group_summary(rows, group_col):
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(group_col, "blank") or "blank"].append(r)

    out = []
    for k, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        labels = Counter(r["label"] for r in items)
        out.append({
            group_col: k,
            "n": len(items),
            "verified": labels.get("verified", 0),
            "refuted": labels.get("refuted", 0),
            "unsupported": labels.get("unsupported", 0),
            "unlabeled": labels.get("unlabeled", 0),
            "provenance_complete_rate": float(np.mean([r["provenance_complete"] for r in items])),
            "mean_content_ids": float(np.mean([r["num_content_ids"] for r in items])),
            "mean_context_ids": float(np.mean([r["num_context_ids"] for r in items])),
            "mean_unique_pages": float(np.mean([r["num_unique_pages"] for r in items])),
            "mean_graph_nodes": float(np.mean([r["graph_num_nodes"] for r in items])),
            "mean_graph_edges": float(np.mean([r["graph_num_edges"] for r in items])),
        })

    return pd.DataFrame(out)

def write_latex(df, path, caption, label):
    out = df.copy()
    for c in out.columns:
        if out[c].dtype.kind in "fc":
            out[c] = out[c].map(lambda x: f"{float(x):.3f}")
    tex = out.to_latex(index=False, escape=True)
    tex = tex.replace(
        "\\begin{tabular}",
        f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}"
    )
    Path(path).write_text(tex, encoding="utf-8")

print("Loading fever/feverous...")
ds = load_dataset("fever/feverous", "default", trust_remote_code=True)
print(ds)

all_rows = []
split_rows = []

for split in ds.keys():
    print("Processing split:", split)
    rows = [summarize_example(ex, split) for ex in ds[split]]
    write_jsonl(PROC / f"feverous_{split}_structured.jsonl", rows)
    all_rows.extend(rows)

    lab = Counter(r["label"] for r in rows)
    mod = Counter(r["evidence_modality"] for r in rows)

    split_rows.append({
        "split": split,
        "rows": len(rows),
        "verified": lab.get("verified", 0),
        "refuted": lab.get("refuted", 0),
        "unsupported": lab.get("unsupported", 0),
        "unlabeled": lab.get("unlabeled", 0),
        "sentence_only": mod.get("sentence_only", 0),
        "table_only": mod.get("table_only", 0),
        "mixed_sentence_table": mod.get("mixed_sentence_table", 0),
        "no_evidence": mod.get("no_evidence", 0),
        "provenance_complete_rate": float(np.mean([r["provenance_complete"] for r in rows])),
        "mean_content_ids": float(np.mean([r["num_content_ids"] for r in rows])),
        "mean_context_ids": float(np.mean([r["num_context_ids"] for r in rows])),
        "mean_unique_pages": float(np.mean([r["num_unique_pages"] for r in rows])),
        "mean_graph_nodes": float(np.mean([r["graph_num_nodes"] for r in rows])),
        "mean_graph_edges": float(np.mean([r["graph_num_edges"] for r in rows])),
    })

labelled = [r for r in all_rows if r["label"] != "unlabeled"]

split_df = pd.DataFrame(split_rows)
modality_df = group_summary(labelled, "evidence_modality")
challenge_df = group_summary(labelled, "challenge")

split_path = TABLE / "table_feverous_split_summary.csv"
mod_path = TABLE / "table_feverous_by_modality.csv"
chal_path = TABLE / "table_feverous_by_challenge.csv"
compact_path = TABLE / "table_feverous_structured_provenance_compact.csv"

split_df.to_csv(split_path, index=False)
modality_df.to_csv(mod_path, index=False)
challenge_df.to_csv(chal_path, index=False)
modality_df.to_csv(compact_path, index=False)

write_latex(
    split_df,
    TEX / "table_feverous_split_summary.tex",
    "FEVEROUS structured-provenance split summary.",
    "tab:feverous_split_summary",
)
write_latex(
    modality_df,
    TEX / "table_feverous_structured_provenance.tex",
    "FEVEROUS evidence modality and provenance-graph characteristics.",
    "tab:feverous_structured_provenance",
)

plt.figure(figsize=(8, 5))
plt.bar(modality_df["evidence_modality"], modality_df["n"])
plt.xticks(rotation=25, ha="right")
plt.xlabel("Evidence modality")
plt.ylabel("Number of labelled claims")
plt.title("FEVEROUS evidence modalities")
plt.tight_layout()
plt.savefig(FIG / "fig_feverous_evidence_modalities.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 5))
plt.bar(modality_df["evidence_modality"], modality_df["mean_graph_edges"])
plt.xticks(rotation=25, ha="right")
plt.xlabel("Evidence modality")
plt.ylabel("Mean graph edges")
plt.title("FEVEROUS evidence graph complexity")
plt.tight_layout()
plt.savefig(FIG / "fig_feverous_graph_complexity.png", dpi=300)
plt.close()

metrics = {
    "dataset": "FEVEROUS",
    "purpose": "structured provenance and evidence graph stress test",
    "scope_note": "HF FEVEROUS examples expose evidence IDs and context IDs. This evaluates provenance structure, not resolved table-cell-text NLI.",
    "labelled_rows": len(labelled),
    "split_summary": split_rows,
    "modality_counts": dict(Counter(r["evidence_modality"] for r in labelled)),
    "provenance_complete_rate": float(np.mean([r["provenance_complete"] for r in labelled])),
    "mean_graph_nodes": float(np.mean([r["graph_num_nodes"] for r in labelled])),
    "mean_graph_edges": float(np.mean([r["graph_num_edges"] for r in labelled])),
}
metric_path = METRIC / "feverous_structured_provenance_metrics.json"
metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

report_path = FINAL / "feverous_structured_provenance_report.md"
with report_path.open("w", encoding="utf-8") as f:
    f.write("# FEVEROUS Structured-Provenance Experiment\n\n")
    f.write("This experiment characterizes FEVEROUS as a structured provenance stress test for P(c), evidence graph completeness, and table/sentence evidence modality.\n\n")
    f.write("## Scope Note\n\n")
    f.write("The Hugging Face examples expose evidence and context identifiers. We therefore report provenance and graph characteristics rather than text-NLI accuracy over resolved table cells.\n\n")
    f.write("## Split Summary\n\n")
    f.write(split_df.to_markdown(index=False))
    f.write("\n\n## Evidence Modality Summary\n\n")
    f.write(modality_df.to_markdown(index=False))
    f.write("\n\n## Challenge Summary\n\n")
    f.write(challenge_df.to_markdown(index=False))
    f.write("\n\n## Paper Interpretation\n\n")
    f.write("FEVEROUS motivates the explicit P(c) term in the verification rule: evidence can be present as structured identifiers and graph links even when plain text evidence is not directly available. This supports separating support S(c), contradiction K(c), rule consistency R(c), provenance completeness P(c), and calibrated risk rho(c).\n")

project_status = {
    "paper_experiment_status": "ready_for_manuscript_drafting",
    "completed": [
        "SciFact",
        "FEVER-pilot",
        "FEVER-full-dev",
        "VitaminC",
        "PubHealth",
        "Climate-FEVER",
        "FEVEROUS structured provenance",
    ],
    "core_equation": "verified iff S(c) >= tau_s and K(c) < tau_k and R(c)=1 and P(c)=1 and rho(c)<=alpha",
}
(FINAL / "project_ready_for_paper_status.json").write_text(json.dumps(project_status, indent=2), encoding="utf-8")

print("\nSaved FEVEROUS files:")
for p in [
    split_path, mod_path, chal_path, compact_path, metric_path, report_path,
    TEX / "table_feverous_split_summary.tex",
    TEX / "table_feverous_structured_provenance.tex",
    FIG / "fig_feverous_evidence_modalities.png",
    FIG / "fig_feverous_graph_complexity.png",
    FINAL / "project_ready_for_paper_status.json",
]:
    print(f"  {p} ({Path(p).stat().st_size} bytes)")

print("\nSplit summary:")
print(split_df.to_string(index=False))

print("\nModality summary:")
print(modality_df.to_string(index=False))
PY

echo ""
echo "==== Run fixed FEVEROUS provenance script ===="
python scripts/run_feverous_structured_provenance_fixed.py

echo ""
echo "==== Confirm FEVEROUS success markers ===="
test -s outputs/final_report/feverous_structured_provenance_report.md && echo "OK report exists" || echo "MISSING report"
test -s outputs/tables/feverous/table_feverous_split_summary.csv && echo "OK split table exists" || echo "MISSING split table"
test -s outputs/metrics/feverous/feverous_structured_provenance_metrics.json && echo "OK metrics exists" || echo "MISSING metrics"
test -s outputs/latex_tables/table_feverous_structured_provenance.tex && echo "OK latex table exists" || echo "MISSING latex table"

echo ""
echo "==== Final FEVEROUS files ===="
find data/processed/feverous outputs/tables/feverous outputs/metrics/feverous outputs/final_report outputs/latex_tables outputs/figures \
  -maxdepth 1 -type f | grep -E "feverous|project_ready" | sort || true

echo ""
echo "==== FEVEROUS NOW SELF-CONTAINED END ===="
date
echo "Log saved to: $LOG"
