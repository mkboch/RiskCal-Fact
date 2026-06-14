from pathlib import Path
import json
import pandas as pd
import numpy as np

OUT = Path("outputs/final_report")
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    p = Path(path)
    if p.exists():
        return pd.read_csv(p)
    print("Missing:", p)
    return pd.DataFrame()

def fmt(x, digits=3):
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)

def short_model(x):
    x = str(x)
    if "ynie/roberta-large" in x:
        return "RoBERTa-large NLI"
    if "facebook/bart-large-mnli" in x:
        return "BART-large MNLI"
    if "MoritzLaurer" in x:
        return "DeBERTa-v3 MNLI-FEVER-ANLI"
    if "cross-encoder" in x:
        return "DeBERTa-v3 NLI"
    return x

def clean_rule(x):
    x = str(x)
    return {
        "margin_support_refute": "Support-refute margin",
        "verified_safe": "Verified-safe margin",
        "support_only": "Support only",
        "winner_takes_all": "Winner takes all",
        "risk_abstain": "Risk abstain",
    }.get(x, x)

def md_escape(x):
    return str(x).replace("|", "\\|")

def latex_escape(x):
    x = str(x)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for a, b in repl.items():
        x = x.replace(a, b)
    return x

def df_to_latex_booktabs(df, caption, label, path):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\caption{" + latex_escape(caption) + r"}")
    lines.append(r"\label{" + label + r"}")
    lines.append(r"\begin{tabular}{" + "l" * len(df.columns) + r"}")
    lines.append(r"\toprule")
    lines.append(" & ".join(latex_escape(c) for c in df.columns) + r" \\")
    lines.append(r"\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(latex_escape(row[c]) for c in df.columns) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    Path(path).write_text("\n".join(lines), encoding="utf-8")

# ---------------------------
# Retrieval table
# ---------------------------
sf_ret = read_csv("outputs/tables/scifact/table_scifact_retrieval.csv")
fv_ret = read_csv("outputs/tables/fever/table_fever_pilot_retrieval.csv")

ret_rows = []

if not sf_ret.empty:
    for _, r in sf_ret.iterrows():
        ret_rows.append({
            "Dataset": "SciFact",
            "Split": str(r["split"]),
            "R@1": fmt(r.get("evidence_doc_recall@1")),
            "R@3": fmt(r.get("evidence_doc_recall@3")),
            "R@5": fmt(r.get("evidence_doc_recall@5")),
            "R@10": fmt(r.get("evidence_doc_recall@10")),
        })

if not fv_ret.empty:
    for _, r in fv_ret.iterrows():
        ret_rows.append({
            "Dataset": "FEVER-pilot",
            "Split": str(r["split"]),
            "R@1": fmt(r.get("evidence_sentence_recall@1")),
            "R@3": fmt(r.get("evidence_sentence_recall@3")),
            "R@5": fmt(r.get("evidence_sentence_recall@5")),
            "R@10": fmt(r.get("evidence_sentence_recall@10")),
        })

ret_df = pd.DataFrame(ret_rows)
ret_df.to_csv(OUT / "clean_table_retrieval.csv", index=False)
df_to_latex_booktabs(
    ret_df,
    "Evidence retrieval performance on SciFact and FEVER-pilot.",
    "tab:retrieval_results",
    OUT / "table_retrieval.tex"
)

# ---------------------------
# Best result table
# ---------------------------
best = read_csv("outputs/final_report/combined_best_results.csv")
best_rows = []

if not best.empty:
    for _, r in best.iterrows():
        method = str(r.get("method", ""))
        if "|" in method:
            model, rule = [s.strip() for s in method.split("|", 1)]
        else:
            model, rule = method, ""
        best_rows.append({
            "Dataset": r.get("dataset", ""),
            "Selection": str(r.get("selection", "")).replace("_", " "),
            "Model": short_model(model),
            "Rule": clean_rule(rule),
            "Alpha": "" if pd.isna(r.get("alpha", np.nan)) else fmt(r.get("alpha"), 2),
            "Macro-F1": fmt(r.get("macro_f1")),
            "FVR": fmt(r.get("false_verification_rate")),
            "Coverage": fmt(r.get("coverage")),
            "Accepted Acc.": fmt(r.get("accepted_accuracy")),
        })

best_df = pd.DataFrame(best_rows)
best_df.to_csv(OUT / "clean_table_best_results.csv", index=False)
df_to_latex_booktabs(
    best_df,
    "Best performance and low-risk operating points.",
    "tab:best_results",
    OUT / "table_best_results.tex"
)

# ---------------------------
# SciFact main verification table
# ---------------------------
sf_main = read_csv("outputs/tables/scifact/table_scifact_main_verification.csv")
sf_rows = []

if not sf_main.empty:
    for _, r in sf_main.head(8).iterrows():
        sf_rows.append({
            "Model": short_model(r.get("model")),
            "Rule": clean_rule(r.get("rule")),
            "Acc.": fmt(r.get("accuracy_all")),
            "Macro-F1": fmt(r.get("macro_f1_all")),
            "Ver. F1": fmt(r.get("verified_f1")),
            "Ref. F1": fmt(r.get("refuted_f1")),
            "Unsup. F1": fmt(r.get("unsupported_f1")),
            "FVR": fmt(r.get("false_verification_rate")),
        })

sf_main_df = pd.DataFrame(sf_rows)
sf_main_df.to_csv(OUT / "clean_table_scifact_main.csv", index=False)
df_to_latex_booktabs(
    sf_main_df,
    "SciFact verification results on the development split.",
    "tab:scifact_verification",
    OUT / "table_scifact_main.tex"
)

# ---------------------------
# Selected risk calibration rows
# ---------------------------
sf_risk = read_csv("outputs/tables/scifact/table_scifact_risk_calibration.csv")
fv_risk = read_csv("outputs/tables/fever/table_fever_pilot_risk_calibration.csv")

risk_rows = []

def add_risk_rows(df, dataset):
    if df.empty:
        return
    for _, r in df.iterrows():
        model = str(r.get("model", ""))
        rule = str(r.get("rule", ""))
        alpha = float(r.get("alpha"))
        keep = False

        # Keep representative rows only.
        if "facebook/bart-large-mnli" in model and rule == "margin_support_refute" and alpha in [0.05, 0.10, 0.30]:
            keep = True
        if "ynie/roberta-large" in model and rule == "margin_support_refute" and alpha in [0.05, 0.10, 0.30]:
            keep = True
        if dataset == "FEVER-pilot" and "ynie/roberta-large" in model and rule == "margin_support_refute" and alpha in [0.05, 0.10, 0.30]:
            keep = True

        if keep:
            risk_rows.append({
                "Dataset": dataset,
                "Model": short_model(model),
                "Rule": clean_rule(rule),
                "Alpha": fmt(alpha, 2),
                "Macro-F1": fmt(r.get("macro_f1_all")),
                "FVR": fmt(r.get("false_verification_rate")),
                "Verified": str(int(r.get("num_predicted_verified", 0))),
                "Abstain": str(int(r.get("num_abstained", 0))),
                "Coverage": fmt(r.get("coverage")),
                "Accepted Acc.": fmt(r.get("accepted_accuracy")),
            })

add_risk_rows(sf_risk, "SciFact")
add_risk_rows(fv_risk, "FEVER-pilot")

risk_df = pd.DataFrame(risk_rows)
risk_df.to_csv(OUT / "clean_table_risk_calibration_selected.csv", index=False)
df_to_latex_booktabs(
    risk_df,
    "Risk-calibrated operating points on SciFact and FEVER-pilot.",
    "tab:risk_calibration",
    OUT / "table_risk_calibration_selected.tex"
)

# ---------------------------
# Clean Markdown report
# ---------------------------
md_path = OUT / "clean_combined_result_report.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# Clean Combined Experimental Result Report\n\n")

    f.write("## Status\n\n")
    f.write("- SciFact is completed with official data, BM25 retrieval, NLI scoring, decision-rule calibration, and risk-calibrated abstention.\n")
    f.write("- FEVER is completed as a pilot experiment using FEVER v1.0 claims, FEVER wiki pages, sampled tune/cal/dev claims, and a sampled sentence corpus.\n")
    f.write("- FEVER results should be reported as pilot results unless we later scale to the full corpus and full split.\n\n")

    f.write("## Retrieval Results\n\n")
    f.write(ret_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Best Operating Points\n\n")
    f.write(best_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## SciFact Main Verification\n\n")
    f.write(sf_main_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Selected Risk-Calibration Results\n\n")
    f.write(risk_df.to_markdown(index=False))
    f.write("\n\n")

    f.write("## Interpretation\n\n")
    f.write("The results show a consistent performance-risk tradeoff. The highest macro-F1 setting usually accepts more verified claims and therefore has a higher false-verification rate. Risk-calibrated gating reduces accepted verified claims and can lower false-verification risk, but it may reduce coverage and macro-F1. This supports the methodological argument that verification should be treated as a calibrated decision problem rather than as a single uncalibrated label prediction task.\n")

status = {
    "scifact": {
        "status": "completed",
        "official_data": True,
        "retrieval": "BM25 top-10",
        "main_best_macro_f1": float(best_df[best_df["Dataset"] == "SciFact"]["Macro-F1"].astype(float).max()) if not best_df.empty else None,
        "note": "Full SciFact official train/dev used for the completed experiments.",
    },
    "fever": {
        "status": "pilot_completed",
        "official_data": True,
        "wiki_pages_exported": True,
        "pilot": True,
        "note": "FEVER experiment used sampled claims and a sampled sentence corpus, so it must be described as FEVER-pilot unless scaled.",
    },
    "files": {
        "clean_report": str(md_path),
        "retrieval_tex": str(OUT / "table_retrieval.tex"),
        "best_results_tex": str(OUT / "table_best_results.tex"),
        "scifact_main_tex": str(OUT / "table_scifact_main.tex"),
        "risk_calibration_tex": str(OUT / "table_risk_calibration_selected.tex"),
    }
}
(OUT / "experiment_status_summary.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

print("Saved clean publication outputs:")
for p in sorted(OUT.glob("clean_*")):
    print(f"  {p} ({p.stat().st_size} bytes)")
for p in sorted(OUT.glob("table_*.tex")):
    print(f"  {p} ({p.stat().st_size} bytes)")
print(f"  {OUT / 'experiment_status_summary.json'} ({(OUT / 'experiment_status_summary.json').stat().st_size} bytes)")

print("\nClean best results:")
print(best_df.to_string(index=False))

print("\nSelected risk-calibration rows:")
print(risk_df.to_string(index=False))
