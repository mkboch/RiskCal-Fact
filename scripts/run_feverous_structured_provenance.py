import json
import re
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd
import numpy as np
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

def label_to_text(split, raw_label):
    try:
        feat = split.features.get("label", None)
        if hasattr(feat, "int2str") and isinstance(raw_label, int) and raw_label >= 0:
            s = feat.int2str(raw_label)
        else:
            s = str(raw_label)
    except Exception:
        s = str(raw_label)

    t = s.strip().lower().replace("_", " ").replace("-", " ")
    if t in {"supports", "support", "supported", "1"}:
        return "verified"
    if t in {"refutes", "refute", "refuted", "0"}:
        return "refuted"
    if t in {"not enough info", "nei", "2"}:
        return "unsupported"
    if t in {"-1", "unknown"}:
        return "unlabeled"
    return t

def evidence_type_from_id(x):
    x = str(x)
    if "_cell_" in x or "_header_cell_" in x:
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
    for pat in ["_sentence_", "_cell_", "_header_cell_", "_section_", "_title"]:
        if pat in x:
            return x.split(pat)[0]
    return x.split("_")[0] if "_" in x else x

def flatten_evidence(evidence):
    content_ids = []
    context_ids = []
    bundles = []

    for bundle_idx, ev in enumerate(evidence or []):
        content = ev.get("content", []) if isinstance(ev, dict) else []
        context = ev.get("context", []) if isinstance(ev, dict) else []

        if content is None:
            content = []
        if context is None:
            context = []

        bundle = {
            "bundle_index": bundle_idx,
            "content": list(content),
            "context": context,
        }
        bundles.append(bundle)

        for c in content:
            content_ids.append(str(c))

        for ctx in context:
            if isinstance(ctx, list):
                for c in ctx:
                    context_ids.append(str(c))
            else:
                context_ids.append(str(ctx))

    return content_ids, context_ids, bundles

def summarize_row(ex, split_name, split_obj):
    raw_label = ex.get("label")
    label = label_to_text(split_obj, raw_label)

    evidence = ex.get("evidence", []) or []
    content_ids, context_ids, bundles = flatten_evidence(evidence)

    content_types = [evidence_type_from_id(x) for x in content_ids]
    context_types = [evidence_type_from_id(x) for x in context_ids]
    all_ids = content_ids + context_ids

    pages = sorted(set(page_from_id(x) for x in all_ids if x))
    content_pages = sorted(set(page_from_id(x) for x in content_ids if x))

    has_sentence = any(t == "sentence" for t in content_types)
    has_cell = any(t == "table_cell" for t in content_types)

    if has_sentence and has_cell:
        evidence_modality = "mixed_sentence_table"
    elif has_cell:
        evidence_modality = "table_only"
    elif has_sentence:
        evidence_modality = "sentence_only"
    elif len(content_ids) == 0:
        evidence_modality = "no_evidence"
    else:
        evidence_modality = "other"

    provenance_complete = int(
        len(content_ids) > 0
        and len(context_ids) > 0
        and all(str(x).strip() for x in content_ids)
    )

    graph_nodes = set(["claim"])
    graph_edges = []

    for c in content_ids:
        graph_nodes.add("evidence:" + c)
        graph_edges.append(("claim", "evidence:" + c, "supported_by"))

    for c in context_ids:
        graph_nodes.add("context:" + c)

    for c in content_ids:
        cpage = page_from_id(c)
        for ctx in context_ids:
            if page_from_id(ctx) == cpage:
                graph_edges.append(("evidence:" + c, "context:" + ctx, "has_context"))

    row = {
        "id": str(ex.get("id", "")),
        "dataset": "feverous",
        "split": split_name,
        "claim": str(ex.get("claim", "")),
        "label": label,
        "raw_label": raw_label,
        "challenge": ex.get("challenge", ""),
        "expected_challenge": ex.get("expected_challenge", ""),
        "num_evidence_bundles": len(evidence),
        "num_content_ids": len(content_ids),
        "num_context_ids": len(context_ids),
        "num_unique_pages": len(pages),
        "num_content_pages": len(content_pages),
        "evidence_modality": evidence_modality,
        "provenance_complete": provenance_complete,
        "graph_num_nodes": len(graph_nodes),
        "graph_num_edges": len(graph_edges),
        "content_type_counts": dict(Counter(content_types)),
        "context_type_counts": dict(Counter(context_types)),
        "content_ids": content_ids,
        "context_ids": context_ids,
        "pages": pages,
        "evidence_bundles": bundles,
    }
    return row

def group_summary(rows, group_col):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[r.get(group_col, "")].append(r)

    for k, items in sorted(groups.items(), key=lambda x: str(x[0])):
        labels = Counter(r["label"] for r in items)
        out.append({
            group_col: k if k != "" else "blank",
            "n": len(items),
            "verified": labels.get("verified", 0),
            "refuted": labels.get("refuted", 0),
            "unsupported": labels.get("unsupported", 0),
            "unlabeled": labels.get("unlabeled", 0),
            "provenance_complete_rate": np.mean([r["provenance_complete"] for r in items]) if items else 0,
            "mean_content_ids": np.mean([r["num_content_ids"] for r in items]) if items else 0,
            "mean_context_ids": np.mean([r["num_context_ids"] for r in items]) if items else 0,
            "mean_graph_nodes": np.mean([r["graph_num_nodes"] for r in items]) if items else 0,
            "mean_graph_edges": np.mean([r["graph_num_edges"] for r in items]) if items else 0,
            "mean_unique_pages": np.mean([r["num_unique_pages"] for r in items]) if items else 0,
        })

    return pd.DataFrame(out)

def make_latex(df, path, caption, label):
    tex = df.to_latex(index=False, escape=True)
    tex = tex.replace("\\begin{tabular}", f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\begin{{tabular}}")
    Path(path).write_text(tex, encoding="utf-8")

def main():
    print("Loading FEVEROUS from Hugging Face: fever/feverous")
    ds = load_dataset("fever/feverous", "default", trust_remote_code=True)
    print(ds)

    all_rows = []
    split_summaries = []

    for split_name in ds.keys():
        rows = []
        for ex in ds[split_name]:
            rows.append(summarize_row(ex, split_name, ds[split_name]))

        write_jsonl(PROC / f"feverous_{split_name}_structured.jsonl", rows)
        all_rows.extend(rows)

        labels = Counter(r["label"] for r in rows)
        modalities = Counter(r["evidence_modality"] for r in rows)

        split_summaries.append({
            "split": split_name,
            "rows": len(rows),
            "verified": labels.get("verified", 0),
            "refuted": labels.get("refuted", 0),
            "unsupported": labels.get("unsupported", 0),
            "unlabeled": labels.get("unlabeled", 0),
            "sentence_only": modalities.get("sentence_only", 0),
            "table_only": modalities.get("table_only", 0),
            "mixed_sentence_table": modalities.get("mixed_sentence_table", 0),
            "no_evidence": modalities.get("no_evidence", 0),
            "provenance_complete_rate": np.mean([r["provenance_complete"] for r in rows]) if rows else 0,
            "mean_content_ids": np.mean([r["num_content_ids"] for r in rows]) if rows else 0,
            "mean_context_ids": np.mean([r["num_context_ids"] for r in rows]) if rows else 0,
            "mean_graph_nodes": np.mean([r["graph_num_nodes"] for r in rows]) if rows else 0,
            "mean_graph_edges": np.mean([r["graph_num_edges"] for r in rows]) if rows else 0,
            "mean_unique_pages": np.mean([r["num_unique_pages"] for r in rows]) if rows else 0,
        })

    split_df = pd.DataFrame(split_summaries)
    split_path = TABLE / "table_feverous_split_summary.csv"
    split_df.to_csv(split_path, index=False)

    labelled = [r for r in all_rows if r["label"] != "unlabeled"]
    modality_df = group_summary(labelled, "evidence_modality")
    modality_path = TABLE / "table_feverous_by_modality.csv"
    modality_df.to_csv(modality_path, index=False)

    challenge_df = group_summary(labelled, "challenge")
    challenge_path = TABLE / "table_feverous_by_challenge.csv"
    challenge_df.to_csv(challenge_path, index=False)

    # Compact paper table.
    compact = modality_df.copy()
    compact = compact.sort_values("n", ascending=False)
    compact_path = TABLE / "table_feverous_structured_provenance_compact.csv"
    compact.to_csv(compact_path, index=False)

    # LaTeX tables.
    split_tex = split_df.copy()
    for c in split_tex.columns:
        if "rate" in c or "mean" in c:
            split_tex[c] = split_tex[c].map(lambda x: f"{float(x):.3f}")

    make_latex(
        split_tex,
        TEX / "table_feverous_split_summary.tex",
        "FEVEROUS structured-provenance split summary.",
        "tab:feverous_split_summary",
    )

    compact_tex_cols = [
        "evidence_modality", "n", "verified", "refuted", "unsupported",
        "provenance_complete_rate", "mean_content_ids", "mean_context_ids",
        "mean_graph_nodes", "mean_graph_edges", "mean_unique_pages"
    ]
    compact_tex = compact[compact_tex_cols].copy()
    for c in compact_tex.columns:
        if "rate" in c or "mean" in c:
            compact_tex[c] = compact_tex[c].map(lambda x: f"{float(x):.3f}")

    make_latex(
        compact_tex,
        TEX / "table_feverous_structured_provenance.tex",
        "FEVEROUS evidence modality and provenance-graph characteristics on labelled splits.",
        "tab:feverous_structured_provenance",
    )

    # Figures.
    plot_df = compact.copy()
    if len(plot_df):
        plt.figure(figsize=(8, 5))
        plt.bar(plot_df["evidence_modality"], plot_df["n"])
        plt.xlabel("Evidence modality")
        plt.ylabel("Number of labelled claims")
        plt.title("FEVEROUS structured evidence modalities")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / "fig_feverous_evidence_modalities.png", dpi=300)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.bar(plot_df["evidence_modality"], plot_df["mean_graph_edges"])
        plt.xlabel("Evidence modality")
        plt.ylabel("Mean graph edges")
        plt.title("FEVEROUS evidence graph complexity")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(FIG / "fig_feverous_graph_complexity.png", dpi=300)
        plt.close()

    # Machine-readable metrics.
    metrics = {
        "dataset": "FEVEROUS",
        "purpose": "structured evidence and provenance completeness stress test",
        "important_note": (
            "The Hugging Face version exposes evidence identifiers and contexts, not necessarily resolved "
            "table-cell text. Therefore this experiment evaluates provenance structure and evidence-graph "
            "completeness, not text NLI accuracy."
        ),
        "split_summary": split_summaries,
        "labelled_rows": len(labelled),
        "evidence_modality_counts_labelled": dict(Counter(r["evidence_modality"] for r in labelled)),
        "provenance_complete_rate_labelled": float(np.mean([r["provenance_complete"] for r in labelled])) if labelled else 0,
        "mean_graph_nodes_labelled": float(np.mean([r["graph_num_nodes"] for r in labelled])) if labelled else 0,
        "mean_graph_edges_labelled": float(np.mean([r["graph_num_edges"] for r in labelled])) if labelled else 0,
        "outputs": {
            "split_summary": str(split_path),
            "by_modality": str(modality_path),
            "by_challenge": str(challenge_path),
            "compact": str(compact_path),
        },
    }
    metric_path = METRIC / "feverous_structured_provenance_metrics.json"
    metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # Markdown report.
    report_path = FINAL / "feverous_structured_provenance_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# FEVEROUS Structured-Provenance Experiment\n\n")
        f.write("## Purpose\n\n")
        f.write(
            "This experiment evaluates the structured provenance component of the verifier. "
            "Unlike SciFact, FEVER, VitaminC, PubHealth, and Climate-FEVER, FEVEROUS contains "
            "structured sentence/table/cell evidence identifiers. Therefore it is used as a "
            "stress test for evidence graph construction and provenance completeness P(c).\n\n"
        )

        f.write("## Important Scope Note\n\n")
        f.write(
            "The available Hugging Face version exposes evidence identifiers and context identifiers. "
            "It does not directly provide resolved table-cell text in the normalized examples used here. "
            "Therefore this result should be reported as a structured-provenance characterization, "
            "not as a text-NLI verification benchmark.\n\n"
        )

        f.write("## Split Summary\n\n")
        f.write(split_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Evidence Modality and Graph Complexity\n\n")
        f.write(compact.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Challenge Breakdown\n\n")
        f.write(challenge_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Paper Interpretation\n\n")
        f.write(
            "FEVEROUS shows why a verifier needs an explicit provenance term P(c). "
            "Evidence may be complete as identifiers and graph links even when natural-language "
            "evidence text is not directly available. This supports separating evidence support S(c), "
            "contradiction K(c), symbolic/provenance consistency R(c), P(c), and calibrated risk rho(c). "
            "The result also motivates future extensions that resolve table cells into text before NLI scoring.\n"
        )

    print("\nSaved FEVEROUS outputs:")
    for p in [
        split_path,
        modality_path,
        challenge_path,
        compact_path,
        metric_path,
        report_path,
        TEX / "table_feverous_split_summary.tex",
        TEX / "table_feverous_structured_provenance.tex",
        FIG / "fig_feverous_evidence_modalities.png",
        FIG / "fig_feverous_graph_complexity.png",
    ]:
        if Path(p).exists():
            print(f"  {p} ({Path(p).stat().st_size} bytes)")

    print("\nSplit summary:")
    print(split_df.to_string(index=False))

    print("\nModality summary:")
    print(compact.to_string(index=False))

if __name__ == "__main__":
    main()
