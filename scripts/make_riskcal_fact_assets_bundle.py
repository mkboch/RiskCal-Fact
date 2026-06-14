import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(".")
ASSETS = ROOT / "paper_assets"
FIG = ASSETS / "figures"
TAB = ASSETS / "tables"
CAP = ASSETS / "captions"
CODES = ROOT / "codes"
CODE_SCRIPTS = CODES / "scripts"
CODE_REPRO = CODES / "reproducibility"

for d in [FIG, TAB, CAP, CODE_SCRIPTS, CODE_REPRO]:
    d.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Missing file: {p}")
        return pd.DataFrame()
    return pd.read_csv(p)

def fmt_num(x):
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)

def fmt_df(df, cols):
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = out[c].map(lambda x: "" if pd.isna(x) else fmt_num(x))
    return out

def save_latex_table(df, path, caption, label):
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += f"\\caption{{{caption}}}\n"
    tex += f"\\label{{{label}}}\n"
    tex += df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    Path(path).write_text(tex, encoding="utf-8")

def save_caption(name, text):
    (CAP / f"{name}.txt").write_text(text.strip() + "\n", encoding="utf-8")

def copy_if_exists(src, dst_dir):
    p = Path(src)
    if p.exists():
        shutil.copy2(p, dst_dir / p.name)
        print(f"Copied code: {p}")
    else:
        print(f"[WARN] Code missing, skipped: {p}")

print("==== Creating paper_assets ====")

# ----------------------------
# Table 1: Benchmark summary
# ----------------------------
table1 = pd.DataFrame([
    ["SciFact", "Retrieval-grounded", "Scientific abstracts", "300 dev", "Scientific claim verification with retrieved evidence"],
    ["FEVER-full-dev", "Retrieval-grounded", "Wikipedia sentences", "9,999 dev", "Large retrieval-grounded verification stress test"],
    ["FEVER oracle", "Oracle diagnostic", "Gold FEVER evidence identifiers", "900 dev", "Isolates verifier behavior from retrieval error"],
    ["VitaminC", "Evidence-given", "Contrastive evidence sentence", "55,197 dev/test", "Evidence-given factual verification"],
    ["PubHealth", "Evidence-given", "Public-health explanations/main text", "1,233 dev/test", "Domain-sensitive health claim verification"],
    ["Climate-FEVER", "Evidence-given", "Climate evidence sentences", "556 dev split", "Climate claim verification and risk-control behavior"],
    ["FEVEROUS", "Structured provenance", "Sentence/table identifiers", "78,991 total", "Structured-provenance characterization"],
], columns=["Benchmark", "Setting", "Evidence type", "Examples", "Role"])
table1.to_csv(TAB / "table1_benchmark_summary.csv", index=False)
save_latex_table(
    table1,
    TAB / "table1_benchmark_summary.tex",
    "Benchmarks and evaluation settings used in RiskCal-Fact.",
    "tab:benchmark_summary"
)

# ----------------------------
# Table 2: Unified ablation
# ----------------------------
ablation = read_csv("outputs/tables/review_hardening/table_unified_ablation_final.csv")
if not ablation.empty:
    if "Alpha" in ablation.columns:
        ablation["Alpha"] = ablation["Alpha"].map(
            lambda x: "--" if pd.isna(x) or str(x).lower() == "nan" or str(x).strip() == "" else f"{float(x):.2f}"
        )
    ablation_show = fmt_df(ablation, ["Macro-F1", "FVR", "Coverage", "Accepted Acc."])
    ablation_show.to_csv(TAB / "table2_unified_ablation.csv", index=False)
    save_latex_table(
        ablation_show,
        TAB / "table2_unified_ablation.tex",
        "Unified ablation of support, contradiction, calibrated risk gating, and modular provenance gating. S and S+K rows are ungated full-coverage ablations; risk-gated rows are selected at alpha=0.10.",
        "tab:unified_ablation"
    )

# ----------------------------
# Table 3: FEVER retrieval vs oracle
# ----------------------------
oracle = read_csv("outputs/tables/review_hardening/table_fever_oracle_flat_diagnostic.csv")
table3_rows = [
    {
        "Setting": "FEVER-full-dev retrieved",
        "Rule": "S+K+P+rho",
        "Macro-F1": 0.582,
        "Accuracy": "",
        "FVR": 0.113,
        "Coverage": 0.999,
        "Predicted verified": "",
        "Note": "Retrieved evidence",
    }
]
if not oracle.empty:
    for rule in ["S", "S+K", "S+K+P", "S+K+P+rho"]:
        sub = oracle[oracle["Rule"].astype(str) == rule].copy()
        if sub.empty:
            continue
        if rule == "S+K+P+rho" and "Alpha" in sub.columns:
            tmp = sub[sub["Alpha"].astype(float).round(2) == 0.05]
            sub = tmp if not tmp.empty else sub.head(1)
        else:
            sub = sub.head(1)
        r = sub.iloc[0]
        table3_rows.append({
            "Setting": "FEVER oracle",
            "Rule": rule,
            "Macro-F1": r.get("macro_f1", ""),
            "Accuracy": r.get("accuracy", ""),
            "FVR": r.get("false_verification_rate", ""),
            "Coverage": r.get("coverage", ""),
            "Predicted verified": r.get("num_predicted_verified", ""),
            "Note": "Gold evidence diagnostic",
        })
table3 = pd.DataFrame(table3_rows)
table3 = fmt_df(table3, ["Macro-F1", "Accuracy", "FVR", "Coverage"])
table3.to_csv(TAB / "table3_fever_retrieval_oracle.csv", index=False)
save_latex_table(
    table3,
    TAB / "table3_fever_retrieval_oracle.tex",
    "FEVER retrieval-grounded result and oracle diagnostic. The oracle uses gold evidence identifiers and is an upper-bound diagnostic, not a deployable setting.",
    "tab:fever_oracle"
)

# ----------------------------
# Table 4: Selected CI rows
# ----------------------------
ci = read_csv("outputs/tables/review_hardening/table_bootstrap_ci_selected.csv")
if not ci.empty:
    selected = []
    def add_ci(dataset, model_contains, alpha):
        sub = ci[
            (ci["Dataset"].astype(str).str.lower() == dataset.lower()) &
            (ci["Model"].astype(str).str.contains(model_contains, case=False, regex=False)) &
            (ci["Alpha"].astype(float).round(2) == alpha)
        ].copy()
        if not sub.empty:
            selected.append(sub.iloc[0].to_dict())

    add_ci("fever_full_dev", "RoBERTa", 0.30)
    add_ci("vitaminc", "RoBERTa", 0.10)
    add_ci("pubhealth", "BART", 0.30)
    add_ci("climate_fever", "RoBERTa", 0.10)

    table4 = pd.DataFrame(selected) if selected else ci.head(8).copy()
    keep_cols = [
        "Dataset", "Model", "Rule", "Alpha",
        "Macro-F1", "Macro-F1 95% CI",
        "FVR", "FVR 95% CI",
        "Coverage", "Coverage 95% CI"
    ]
    table4 = table4[[c for c in keep_cols if c in table4.columns]].copy()
    table4 = fmt_df(table4, ["Alpha", "Macro-F1", "FVR", "Coverage"])
    table4.to_csv(TAB / "table4_selected_ci.csv", index=False)
    save_latex_table(
        table4,
        TAB / "table4_selected_ci.tex",
        "Selected risk-calibrated operating points with bootstrap 95\\% confidence intervals.",
        "tab:selected_ci"
    )

# ----------------------------
# Table 5: FEVEROUS provenance
# ----------------------------
feverous = read_csv("outputs/tables/feverous/table_feverous_by_modality.csv")
if not feverous.empty:
    colmap = {
        "modality": "Evidence modality",
        "n": "Claims",
        "verified": "Verified",
        "refuted": "Refuted",
        "unsupported": "Unsupported",
        "mean_graph_nodes": "Mean graph nodes",
        "mean_graph_edges": "Mean graph edges",
        "mean_unique_pages": "Mean unique pages",
    }
    feverous_show = feverous.rename(columns={k: v for k, v in colmap.items() if k in feverous.columns})
    keep = [v for v in colmap.values() if v in feverous_show.columns]
    feverous_show = feverous_show[keep].copy()
    feverous_show = fmt_df(feverous_show, ["Mean graph nodes", "Mean graph edges", "Mean unique pages"])
    feverous_show.to_csv(TAB / "table5_feverous_provenance.csv", index=False)
    save_latex_table(
        feverous_show,
        TAB / "table5_feverous_provenance.tex",
        "FEVEROUS structured-provenance characterization by evidence modality.",
        "tab:feverous_provenance"
    )

# ----------------------------
# Figure 1: framework
# ----------------------------
fig, ax = plt.subplots(figsize=(12, 6.8))
ax.axis("off")
boxes = {
    "Claim": (0.05, 0.55, 0.16, 0.18),
    "Evidence retrieval\nor evidence input": (0.29, 0.55, 0.20, 0.18),
    "Support\nS(c)": (0.60, 0.76, 0.14, 0.13),
    "Contradiction\nK(c)": (0.60, 0.58, 0.14, 0.13),
    "Risk\nrho(c)": (0.60, 0.40, 0.14, 0.13),
    "Optional gates\nP(c), R(c)": (0.60, 0.22, 0.14, 0.13),
    "Selective decision\nverified / refuted /\nunsupported / abstain": (0.83, 0.50, 0.15, 0.24),
}
for label, (x, y, w, h) in boxes.items():
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.8))
    ax.text(x + w/2, y + h/2, label, ha="center", va="center", fontsize=11)
def arrow(x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", linewidth=1.6))
arrow(0.21, 0.64, 0.29, 0.64)
arrow(0.49, 0.64, 0.60, 0.825)
arrow(0.49, 0.64, 0.60, 0.645)
arrow(0.49, 0.64, 0.60, 0.465)
arrow(0.49, 0.64, 0.60, 0.285)
arrow(0.74, 0.825, 0.83, 0.62)
arrow(0.74, 0.645, 0.83, 0.62)
arrow(0.74, 0.465, 0.83, 0.62)
arrow(0.74, 0.285, 0.83, 0.62)
ax.text(0.60, 0.94, "Empirically active core", ha="left", va="center", fontsize=12, fontweight="bold")
ax.text(0.60, 0.17, "Modular deployment extensions", ha="left", va="center", fontsize=10)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
fig.tight_layout()
fig.savefig(FIG / "fig1_riskcal_fact_framework.pdf", bbox_inches="tight")
fig.savefig(FIG / "fig1_riskcal_fact_framework.png", dpi=300, bbox_inches="tight")
plt.close(fig)

# ----------------------------
# Figure 2: ablation macro-F1
# ----------------------------
if not ablation.empty:
    df = ablation.copy()
    datasets = list(dict.fromkeys(df["Dataset"]))
    rules = ["S", "S+K", "S+K+rho", "S+K+P+rho"]
    x = np.arange(len(datasets))
    width = 0.18
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, rule in enumerate(rules):
        vals = []
        for ds in datasets:
            sub = df[(df["Dataset"] == ds) & (df["Rule"].astype(str) == rule)]
            vals.append(float(sub["Macro-F1"].iloc[0]) if not sub.empty else np.nan)
        ax.bar(x + (i - 1.5) * width, vals, width, label=rule)
    ax.set_ylabel("Macro-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=20, ha="right")
    ax.set_ylim(0, 0.75)
    ax.legend(frameon=False, ncols=2)
    ax.set_title("Unified ablation across verification settings")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_unified_ablation_macro_f1.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig2_unified_ablation_macro_f1.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Figure 3: FEVER retrieved vs oracle
# ----------------------------
if len(table3) > 0:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = [f"{r['Setting']}\n{r['Rule']}" for _, r in table3.iterrows()]
    macro = pd.to_numeric(table3["Macro-F1"], errors="coerce")
    fvr = pd.to_numeric(table3["FVR"], errors="coerce")
    x = np.arange(len(labels))
    width = 0.35
    ax.bar(x - width/2, macro, width, label="Macro-F1")
    ax.bar(x + width/2, fvr, width, label="FVR")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False)
    ax.set_title("FEVER retrieved evidence versus oracle diagnostic")
    fig.tight_layout()
    fig.savefig(FIG / "fig3_fever_retrieval_vs_oracle.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig3_fever_retrieval_vs_oracle.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Figure 4: risk-coverage operating points
# ----------------------------
risk_long = read_csv("outputs/metrics/review_hardening/fast_ablation_all.csv")
if not risk_long.empty:
    risk = risk_long[risk_long["rule"].astype(str) == "S+K+P+rho"].copy()
    if "model" in risk.columns:
        risk = risk[risk["model"].astype(str).str.contains("roberta|RoBERTa|ynie", case=False, regex=True)]
    fig, ax = plt.subplots(figsize=(9, 6))
    for ds, sub in risk.groupby("dataset"):
        sub = sub.sort_values("alpha")
        ax.plot(sub["coverage"], sub["false_verification_rate"], marker="o", label=ds)
        for _, r in sub.iterrows():
            ax.text(r["coverage"], r["false_verification_rate"], f"{float(r['alpha']):.2f}", fontsize=7)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("False-verification rate")
    ax.set_title("Risk-coverage operating points across evaluated alpha grid")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_risk_coverage_operating_points.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig4_risk_coverage_operating_points.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# Figure 5: FEVEROUS complexity
# ----------------------------
if not feverous.empty:
    fdf = feverous.copy()

    def pick_col(df, candidates, contains_any=None):
        for c in candidates:
            if c in df.columns:
                return c
        if contains_any:
            for c in df.columns:
                cl = c.lower()
                if all(x in cl for x in contains_any):
                    return c
        return None

    modality_col = pick_col(
        fdf,
        ["modality", "Evidence modality", "evidence_modality", "evidence_type", "modality_group", "type"],
        None
    )
    edge_col = pick_col(
        fdf,
        ["mean_graph_edges", "Mean graph edges", "graph_edges_mean", "avg_graph_edges"],
        ["edge"]
    )

    if modality_col is None:
        # Fall back to the first object-like column.
        obj_cols = [c for c in fdf.columns if fdf[c].dtype == "object"]
        modality_col = obj_cols[0] if obj_cols else fdf.columns[0]

    if edge_col is None:
        numeric_cols = [c for c in fdf.columns if str(fdf[c].dtype).startswith(("int", "float"))]
        edge_like = [c for c in numeric_cols if "edge" in c.lower()]
        edge_col = edge_like[0] if edge_like else numeric_cols[-1]

    print("FEVEROUS columns:", list(fdf.columns))
    print("Using modality_col:", modality_col)
    print("Using edge_col:", edge_col)

    fdf[edge_col] = pd.to_numeric(fdf[edge_col], errors="coerce")
    fdf = fdf.dropna(subset=[edge_col])
    fdf = fdf.sort_values(edge_col)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(fdf[modality_col].astype(str), fdf[edge_col].astype(float))
    ax.set_xlabel("Mean graph edges")
    ax.set_ylabel("Evidence modality")
    ax.set_title("FEVEROUS structured-provenance complexity")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_feverous_graph_complexity.pdf", bbox_inches="tight")
    fig.savefig(FIG / "fig5_feverous_graph_complexity.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

# Captions as txt only, no md.
captions = {
    "fig1_riskcal_fact_framework": "Overview of RiskCal-Fact. The empirically active core consists of support, contradiction, and calibrated false-verification risk; provenance and rule-consistency gates are modular extensions for structured or domain-specific deployments.",
    "fig2_unified_ablation_macro_f1": "Unified ablation across verification settings. Support plus contradiction improves over support-only verification, while risk gating can reduce macro-F1 by abstaining from uncertain verified decisions.",
    "fig3_fever_retrieval_vs_oracle": "FEVER retrieval-grounded result and oracle diagnostic. Gold evidence sharply improves verification, showing that retrieval is a major bottleneck. The oracle is an upper-bound diagnostic, not a deployable setting.",
    "fig4_risk_coverage_operating_points": "Risk-coverage operating points across the evaluated alpha grid. The curves expose operating-point frontiers available to a practitioner and should not be interpreted as monotonic guarantees under distribution shift.",
    "fig5_feverous_graph_complexity": "FEVEROUS structured-provenance complexity. Table-only and mixed sentence-table evidence induce evidence graphs roughly 27--30 times larger than sentence-only evidence, motivating provenance-aware extensions."
}
for k, v in captions.items():
    save_caption(k, v)

manifest = {
    "title": "RiskCal-Fact: Risk-Calibrated Evidence-Constrained Verification of Factual Claims",
    "main_figures": sorted([str(p.relative_to(ASSETS)) for p in FIG.glob("*.pdf")]),
    "main_tables": sorted([str(p.relative_to(ASSETS)) for p in TAB.glob("*.tex")]),
}
(ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("==== Creating codes folder ====")

# Copy reproducibility scripts only if they exist.
code_files = [
    "scripts/make_riskcal_fact_assets_bundle.py",
    "scripts/make_clean_publication_tables.py",
    "scripts/make_final_all_benchmark_artifacts.py",
    "scripts/run_final_review_fix_artifacts.py",
    "scripts/make_final_prewrite_review_tables.py",
    "scripts/final_adversarial_review_fixes.py",
    "scripts/run_scifact_bm25_retrieval.py",
    "scripts/run_scifact_proper_split_risk_calibration.py",
    "scripts/run_evidence_given_datasets_all.sh",
    "scripts/run_fever_oracle_flat_columns_fixed.sh",
]
for f in code_files:
    copy_if_exists(f, CODE_SCRIPTS)

(CODES / "requirements_minimal.txt").write_text(
    "\n".join([
        "python>=3.10",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "torch",
        "transformers",
        "datasets",
        "sentence-transformers",
        "rank-bm25",
        "faiss-cpu",
        "tabulate",
        "jinja2",
        "tqdm",
        ""
    ]),
    encoding="utf-8"
)

# Plain txt only.
env_lines = []
env_lines.append("RiskCal-Fact environment summary")
env_lines.append("Project directory: " + str(ROOT.resolve()))
try:
    env_lines.append(subprocess.check_output(["python", "--version"], text=True).strip())
except Exception:
    pass
for pkg in ["pandas", "numpy", "matplotlib", "sklearn", "torch", "transformers", "datasets", "tqdm"]:
    try:
        mod = __import__(pkg)
        env_lines.append(f"{pkg}: {getattr(mod, '__version__', 'unknown')}")
    except Exception as e:
        env_lines.append(f"{pkg}: NOT FOUND")
(CODES / "environment_summary.txt").write_text("\n".join(env_lines) + "\n", encoding="utf-8")

(CODE_REPRO / "key_result_files.txt").write_text(
    "\n".join([
        "outputs/tables/review_hardening/table_unified_ablation_final.csv",
        "outputs/tables/review_hardening/table_fever_oracle_flat_diagnostic.csv",
        "outputs/tables/review_hardening/table_bootstrap_ci_selected.csv",
        "outputs/tables/feverous/table_feverous_by_modality.csv",
        "outputs/metrics/review_hardening/fast_ablation_all.csv",
        "outputs/final_report/final_adversarial_review_fixes.md",
        "outputs/final_report/final_prewrite_review_resolution.md",
        ""
    ]),
    encoding="utf-8"
)

(CODE_REPRO / "reporting_conventions.txt").write_text(
    "\n".join([
        "RiskCal-Fact reporting conventions",
        "",
        "1. S and S+K rows are ungated full-coverage ablations.",
        "2. S+K+rho and S+K+P+rho rows are risk-gated operating points.",
        "3. Macro-F1 is computed over original task labels; abstentions are treated as non-matching predictions, not a fourth class.",
        "4. Coverage is the fraction of examples receiving a non-abstaining decision.",
        "5. Accepted accuracy is computed only over the covered subset.",
        "6. FEVER oracle results are upper-bound diagnostics using gold evidence identifiers, not deployable results.",
        "7. P(c) and R(c) are modular domain-specific gates; current experiments empirically emphasize S(c), K(c), and rho(c).",
        ""
    ]),
    encoding="utf-8"
)

print("==== Asset creation done ====")
print("paper_assets files:")
for p in sorted(ASSETS.rglob("*")):
    if p.is_file():
        print(p)
print("codes files:")
for p in sorted(CODES.rglob("*")):
    if p.is_file():
        print(p)
