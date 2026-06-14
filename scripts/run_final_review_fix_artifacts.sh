#!/usr/bin/env bash

PROJECT_DIR="$HOME/risk_calibrated_verification"
cd "$PROJECT_DIR" || { echo "ERROR: cannot cd to $PROJECT_DIR"; exit 0; }

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/final_review_fix_artifacts_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

echo "==== FINAL REVIEW FIX ARTIFACTS START ===="
date

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate rcv_py310
echo "python=$(command -v python)"
python --version

mkdir -p outputs/final_report outputs/metrics/review_hardening outputs/tables/review_hardening outputs/latex_tables scripts

cat > scripts/run_final_review_fix_artifacts.py <<'PY'
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(".")
FINAL = Path("outputs/final_report")
METRIC = Path("outputs/metrics/review_hardening")
TABLE = Path("outputs/tables/review_hardening")
TEX = Path("outputs/latex_tables")
for d in [FINAL, METRIC, TABLE, TEX]:
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]

def read_jsonl(path):
    p = Path(path)
    rows = []
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def safe_model_name(model):
    return model.replace("/", "__")

def short_model(model):
    model = str(model)
    if "ynie/roberta" in model:
        return "RoBERTa-large NLI"
    if "facebook/bart" in model:
        return "BART-large MNLI"
    return model

def eval_preds(rows, preds):
    y = [r["label"] for r in rows]
    macro = f1_score(y, preds, labels=LABELS, average="macro", zero_division=0)
    acc = accuracy_score(y, preds)
    pred_verified = [i for i, p in enumerate(preds) if p == "verified"]
    fvr = sum(1 for i in pred_verified if y[i] != "verified") / len(pred_verified) if pred_verified else 0.0
    accepted = [i for i, p in enumerate(preds) if p != "abstain"]
    cov = len(accepted) / len(rows) if rows else 0.0
    acc_accepted = accuracy_score([y[i] for i in accepted], [preds[i] for i in accepted]) if accepted else 0.0
    return {
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "false_verification_rate": float(fvr),
        "coverage": float(cov),
        "accepted_accuracy": float(acc_accepted),
        "num_predicted_verified": len(pred_verified),
        "num_abstained": len(rows) - len(accepted),
    }

def predict_skrho(r, tau_s, tau_k, margin, risk_thr=None):
    S = float(r.get("S", 0.0))
    K = float(r.get("K", 0.0))
    conf = S - K

    if K >= tau_k and (K - S) >= margin:
        pred = "refuted"
    elif S >= tau_s and (S - K) >= margin:
        pred = "verified"
    else:
        pred = "unsupported"

    if pred == "verified" and risk_thr is not None and conf < risk_thr:
        pred = "abstain"

    return pred, conf

def tune_sk(rows):
    vals = [0.30, 0.40, 0.50, 0.60, 0.70]
    margins = [0.00, 0.10, 0.20, 0.30]
    best = None

    for ts in vals:
        for tk in vals:
            for mg in margins:
                preds = [predict_skrho(r, ts, tk, mg)[0] for r in rows]
                m = eval_preds(rows, preds)
                obj = m["macro_f1"] - 0.25 * m["false_verification_rate"]
                if best is None or obj > best["obj"]:
                    best = {"obj": obj, "tau_s": ts, "tau_k": tk, "margin": mg, "metrics": m}
    return best

def risk_threshold(cal_rows, tau_s, tau_k, margin, alpha):
    pc = [predict_skrho(r, tau_s, tau_k, margin) for r in cal_rows]
    idx = [i for i, (p, c) in enumerate(pc) if p == "verified"]
    if not idx:
        return 999.0
    thresholds = sorted(set(pc[i][1] for i in idx))
    for thr in thresholds:
        kept = [i for i in idx if pc[i][1] >= thr]
        fvr = sum(1 for i in kept if cal_rows[i]["label"] != "verified") / len(kept) if kept else 0.0
        if fvr <= alpha:
            return float(thr)
    return float(max(thresholds) + 1e-6)

def collect_score_sets():
    models = [
        "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "facebook/bart-large-mnli",
    ]
    sets = {}

    eg = Path("outputs/predictions/evidence_given")
    for dataset in ["vitaminc", "pubhealth", "climate_fever"]:
        for model in models:
            safe = safe_model_name(model)
            tune = read_jsonl(eg / f"{dataset}_tune_{safe}_scores.jsonl")
            cal = read_jsonl(eg / f"{dataset}_cal_{safe}_scores.jsonl")
            dev = read_jsonl(eg / f"{dataset}_dev_{safe}_scores.jsonl")
            if tune and cal and dev:
                sets[(dataset, model)] = (tune, cal, dev)

    fv = Path("outputs/predictions/fever")
    for model in models:
        safe = safe_model_name(model)
        tune = read_jsonl(fv / f"fast_full_dev_nli_{safe}_tune_scores.jsonl")
        cal = read_jsonl(fv / f"fast_full_dev_nli_{safe}_cal_scores.jsonl")
        dev = read_jsonl(fv / f"fast_full_dev_nli_{safe}_paper_dev_full_scores.jsonl")
        if tune and cal and dev:
            sets[("fever_full_dev", model)] = (tune, cal, dev)

    return sets

def make_skrho_table():
    sets = collect_score_sets()
    rows = []
    alphas = [0.05, 0.10, 0.20, 0.30]

    for (dataset, model), (tune, cal, dev) in sets.items():
        print("Processing S+K+rho:", dataset, short_model(model))
        best = tune_sk(tune)
        for alpha in alphas:
            thr = risk_threshold(cal, best["tau_s"], best["tau_k"], best["margin"], alpha)
            preds = [
                predict_skrho(r, best["tau_s"], best["tau_k"], best["margin"], risk_thr=thr)[0]
                for r in dev
            ]
            m = eval_preds(dev, preds)
            rows.append({
                "dataset": dataset,
                "model": model,
                "rule": "S+K+rho",
                "alpha": alpha,
                "risk_threshold": thr,
                "tau_s": best["tau_s"],
                "tau_k": best["tau_k"],
                "margin": best["margin"],
                **m,
            })

    df = pd.DataFrame(rows)
    df.to_csv(METRIC / "explicit_s_k_rho_all.csv", index=False)

    compact = []
    for dataset in ["fever_full_dev", "vitaminc", "pubhealth", "climate_fever"]:
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue
        if any(sub["model"].astype(str).str.contains("ynie/roberta")):
            sub = sub[sub["model"].astype(str).str.contains("ynie/roberta")]
        selected = sub[sub["alpha"].astype(str).isin(["0.1", "0.10"])]
        if selected.empty:
            selected = sub.sort_values(["false_verification_rate", "macro_f1"], ascending=[True, False]).head(1)
        else:
            selected = selected.sort_values("macro_f1", ascending=False).head(1)

        r = selected.iloc[0]
        compact.append({
            "Dataset": dataset,
            "Model": short_model(r["model"]),
            "Rule": "S+K+rho",
            "Alpha": r["alpha"],
            "Macro-F1": r["macro_f1"],
            "FVR": r["false_verification_rate"],
            "Coverage": r["coverage"],
            "Accepted Acc.": r["accepted_accuracy"],
        })

    cdf = pd.DataFrame(compact)
    cdf.to_csv(TABLE / "table_explicit_s_k_rho_ablation.csv", index=False)

    tex_df = cdf.copy()
    for c in ["Macro-F1", "FVR", "Coverage", "Accepted Acc."]:
        tex_df[c] = tex_df[c].map(lambda x: f"{float(x):.3f}")
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Explicit S+K+rho ablation rows showing risk gating without the provenance gate.}\n"
    tex += "\\label{tab:explicit_s_k_rho_ablation}\n"
    tex += tex_df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_explicit_s_k_rho_ablation.tex").write_text(tex, encoding="utf-8")

    print("\nExplicit S+K+rho compact table:")
    print(cdf.to_string(index=False))
    return df, cdf

def make_ci_table():
    ci_path = METRIC / "fast_bootstrap_ci_selected.csv"
    if not ci_path.exists():
        print("No CI file found:", ci_path)
        return pd.DataFrame()

    ci = pd.read_csv(ci_path)
    if ci.empty:
        return ci

    rows = []
    for _, r in ci.iterrows():
        rows.append({
            "Dataset": r.get("dataset", ""),
            "Model": short_model(r.get("model", "")),
            "Rule": r.get("rule", r.get("ablation", "")),
            "Alpha": r.get("alpha", ""),
            "Macro-F1": r.get("macro_f1", np.nan),
            "Macro-F1 95% CI": f"[{r.get('macro_f1_ci_lo', np.nan):.3f}, {r.get('macro_f1_ci_hi', np.nan):.3f}]",
            "FVR": r.get("false_verification_rate", r.get("fvr", np.nan)),
            "FVR 95% CI": f"[{r.get('fvr_ci_lo', np.nan):.3f}, {r.get('fvr_ci_hi', np.nan):.3f}]",
            "Coverage": r.get("coverage", np.nan),
            "Coverage 95% CI": f"[{r.get('coverage_ci_lo', np.nan):.3f}, {r.get('coverage_ci_hi', np.nan):.3f}]",
        })

    out = pd.DataFrame(rows)
    out.to_csv(TABLE / "table_bootstrap_ci_selected.csv", index=False)

    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Bootstrap 95\\% confidence intervals for selected risk-calibrated operating points.}\n"
    tex += "\\label{tab:bootstrap_ci_selected}\n"
    tex += out.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_bootstrap_ci_selected.tex").write_text(tex, encoding="utf-8")

    print("\nBootstrap CI selected table:")
    print(out.to_string(index=False))
    return out

def evidence_stats_from_scores():
    # Main purpose: provide a mechanistic explanation for PubHealth high FVR.
    datasets = ["vitaminc", "pubhealth", "climate_fever"]
    model = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    safe = safe_model_name(model)
    rows = []

    for dataset in datasets:
        for split in ["tune", "cal", "dev"]:
            data = read_jsonl(Path("outputs/predictions/evidence_given") / f"{dataset}_{split}_{safe}_scores.jsonl")
            if not data:
                continue

            num_units = []
            max_units = []
            labels = []
            S_vals = []
            K_vals = []

            for r in data:
                labels.append(r.get("label"))
                S_vals.append(float(r.get("S", 0.0)))
                K_vals.append(float(r.get("K", 0.0)))
                # Try common fields.
                n = r.get("num_evidence_units", None)
                if n is None:
                    n = r.get("num_evidence", None)
                if n is None:
                    units = r.get("evidence_units", None)
                    if isinstance(units, list):
                        n = len(units)
                if n is None:
                    scores = r.get("unit_scores", None)
                    if isinstance(scores, list):
                        n = len(scores)
                if n is None:
                    n = np.nan
                num_units.append(n)

            num_units_arr = pd.to_numeric(pd.Series(num_units), errors="coerce")
            rows.append({
                "dataset": dataset,
                "split": split,
                "n": len(data),
                "mean_evidence_units": float(num_units_arr.mean()) if not num_units_arr.isna().all() else np.nan,
                "median_evidence_units": float(num_units_arr.median()) if not num_units_arr.isna().all() else np.nan,
                "p90_evidence_units": float(num_units_arr.quantile(0.90)) if not num_units_arr.isna().all() else np.nan,
                "mean_S": float(np.mean(S_vals)),
                "mean_K": float(np.mean(K_vals)),
                "verified_rate": float(np.mean([x == "verified" for x in labels])),
                "refuted_rate": float(np.mean([x == "refuted" for x in labels])),
                "unsupported_rate": float(np.mean([x == "unsupported" for x in labels])),
            })

    df = pd.DataFrame(rows)
    df.to_csv(TABLE / "table_evidence_given_complexity_stats.csv", index=False)

    tex_df = df.copy()
    for c in ["mean_evidence_units", "median_evidence_units", "p90_evidence_units", "mean_S", "mean_K", "verified_rate", "refuted_rate", "unsupported_rate"]:
        if c in tex_df:
            tex_df[c] = tex_df[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.3f}")
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Evidence-given dataset statistics used to interpret domain-specific risk behavior.}\n"
    tex += "\\label{tab:evidence_given_complexity_stats}\n"
    tex += tex_df.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_evidence_given_complexity_stats.tex").write_text(tex, encoding="utf-8")

    print("\nEvidence-given complexity stats:")
    print(df.to_string(index=False))
    return df

def write_review_fix_report(skrho_compact, ci_table, complexity):
    report = FINAL / "final_review_fix_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Final Review-Fix Artifacts\n\n")
        f.write("## Experimental additions\n\n")
        f.write("1. Explicit S+K+rho ablation rows were generated for FEVER-full-dev, VitaminC, PubHealth, and Climate-FEVER.\n")
        f.write("2. Bootstrap confidence intervals were converted into a paper-ready table.\n")
        f.write("3. Evidence-given complexity statistics were generated to support interpretation of PubHealth's higher false-verification risk.\n\n")

        f.write("## Explicit S+K+rho table\n\n")
        f.write(skrho_compact.to_markdown(index=False))
        f.write("\n\n")

        if not ci_table.empty:
            f.write("## Bootstrap CI table\n\n")
            f.write(ci_table.to_markdown(index=False))
            f.write("\n\n")

        f.write("## Evidence-given complexity statistics\n\n")
        f.write(complexity.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Writing decisions\n\n")
        f.write("- The empirically active core rule should be presented as S+K+rho.\n")
        f.write("- P(c) and R(c) should be presented as modular domain-specific gates, not as empirically necessary in every benchmark.\n")
        f.write("- FEVEROUS motivates P(c), but it is a structured-provenance characterization, not a verification-accuracy benchmark.\n")
        f.write("- The FEVER oracle diagnostic is an upper-bound diagnostic, not a deployable setting.\n")
        f.write("- Coverage should be defined as the fraction of examples not abstained from; accepted accuracy is computed only on covered examples.\n")
        f.write("- Macro-F1 should be explicitly described according to the implementation convention used in each table.\n")

    status = {
        "status": "completed",
        "artifacts": [
            "outputs/tables/review_hardening/table_explicit_s_k_rho_ablation.csv",
            "outputs/latex_tables/table_explicit_s_k_rho_ablation.tex",
            "outputs/tables/review_hardening/table_bootstrap_ci_selected.csv",
            "outputs/latex_tables/table_bootstrap_ci_selected.tex",
            "outputs/tables/review_hardening/table_evidence_given_complexity_stats.csv",
            "outputs/latex_tables/table_evidence_given_complexity_stats.tex",
            "outputs/final_report/final_review_fix_report.md",
        ],
    }
    (FINAL / "final_review_fix_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

def main():
    _, skrho_compact = make_skrho_table()
    ci_table = make_ci_table()
    complexity = evidence_stats_from_scores()
    write_review_fix_report(skrho_compact, ci_table, complexity)

    print("\n==== FINAL REVIEW FIX ARTIFACTS COMPLETE ====")
    print((FINAL / "final_review_fix_status.json").read_text())

if __name__ == "__main__":
    main()
PY

python scripts/run_final_review_fix_artifacts.py

echo ""
echo "==== Final review-fix files ===="
find outputs/final_report outputs/tables/review_hardening outputs/metrics/review_hardening outputs/latex_tables \
  -maxdepth 1 -type f | grep -E "final_review_fix|s_k_rho|bootstrap_ci|evidence_given_complexity" | sort || true

echo ""
echo "==== FINAL REVIEW FIX ARTIFACTS END ===="
date
echo "Log saved to: $LOG"
