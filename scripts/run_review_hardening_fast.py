import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

FINAL = Path("outputs/final_report")
METRIC = Path("outputs/metrics/review_hardening")
TABLE = Path("outputs/tables/review_hardening")
TEX = Path("outputs/latex_tables")
FIG = Path("outputs/figures")
for d in [FINAL, METRIC, TABLE, TEX, FIG]:
    d.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]
RNG = np.random.default_rng(42)

def read_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    rows = []
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
    accepted = [i for i, p in enumerate(preds) if p != "abstain"]
    coverage = len(accepted) / len(rows) if rows else 0.0
    accepted_acc = accuracy_score([y[i] for i in accepted], [preds[i] for i in accepted]) if accepted else 0.0
    pred_verified = [i for i, p in enumerate(preds) if p == "verified"]
    fvr = sum(1 for i in pred_verified if y[i] != "verified") / len(pred_verified) if pred_verified else 0.0
    return {
        "macro_f1": float(macro),
        "accuracy": float(acc),
        "false_verification_rate": float(fvr),
        "coverage": float(coverage),
        "accepted_accuracy": float(accepted_acc),
        "num_predicted_verified": len(pred_verified),
        "num_abstained": len(rows) - len(accepted),
    }

def pred_rule(r, rule, tau_s, tau_k=0.5, margin=0.0, risk_thr=None):
    S = float(r.get("S", 0.0))
    K = float(r.get("K", 0.0))
    P = int(r.get("P", 1))
    conf = S - K

    if rule == "S":
        pred = "verified" if S >= tau_s else "unsupported"
    elif rule == "S+K":
        if K >= tau_k and (K - S) >= margin:
            pred = "refuted"
        elif S >= tau_s and (S - K) >= margin:
            pred = "verified"
        else:
            pred = "unsupported"
    elif rule == "S+K+P":
        if K >= tau_k and (K - S) >= margin:
            pred = "refuted"
        elif S >= tau_s and (S - K) >= margin and P == 1:
            pred = "verified"
        else:
            pred = "unsupported"
    elif rule == "S+K+P+rho":
        if K >= tau_k and (K - S) >= margin:
            pred = "refuted"
        elif S >= tau_s and (S - K) >= margin and P == 1:
            pred = "verified"
            if risk_thr is not None and conf < risk_thr:
                pred = "abstain"
        else:
            pred = "unsupported"
    else:
        raise ValueError(rule)

    return pred, conf

def tune_fast(rows, rule):
    # Small grid to avoid hanging.
    tau_s_vals = [0.30, 0.40, 0.50, 0.60, 0.70]
    tau_k_vals = [0.30, 0.40, 0.50, 0.60, 0.70]
    margin_vals = [0.00, 0.10, 0.20, 0.30]

    best = None
    if rule == "S":
        grid = [(ts, 0.5, 0.0) for ts in tau_s_vals]
    else:
        grid = [(ts, tk, mg) for ts in tau_s_vals for tk in tau_k_vals for mg in margin_vals]

    for tau_s, tau_k, margin in grid:
        preds = [pred_rule(r, rule, tau_s, tau_k, margin)[0] for r in rows]
        m = eval_preds(rows, preds)
        obj = m["macro_f1"] - 0.25 * m["false_verification_rate"]
        if best is None or obj > best["obj"]:
            best = {"obj": obj, "tau_s": tau_s, "tau_k": tau_k, "margin": margin, "metrics": m}
    return best

def choose_risk_thr(cal_rows, tau_s, tau_k, margin, alpha):
    preds_conf = [pred_rule(r, "S+K+P", tau_s, tau_k, margin) for r in cal_rows]
    idx = [i for i, (p, c) in enumerate(preds_conf) if p == "verified"]
    if not idx:
        return 999.0

    confs = sorted(set(preds_conf[i][1] for i in idx))
    for thr in confs:
        kept = [i for i in idx if preds_conf[i][1] >= thr]
        if not kept:
            continue
        fvr = sum(1 for i in kept if cal_rows[i]["label"] != "verified") / len(kept)
        if fvr <= alpha:
            return float(thr)
    return float(max(confs) + 1e-6)

def bootstrap_ci(rows, preds, n_boot=500):
    y = np.array([r["label"] for r in rows])
    p = np.array(preds)
    n = len(rows)
    vals = {"macro_f1": [], "fvr": [], "coverage": []}

    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yy, pp = y[idx], p[idx]
        vals["macro_f1"].append(f1_score(yy, pp, labels=LABELS, average="macro", zero_division=0))
        vals["coverage"].append(float(np.mean(pp != "abstain")))
        pv = pp == "verified"
        vals["fvr"].append(float(np.mean(yy[pv] != "verified")) if np.sum(pv) else 0.0)

    out = {}
    for k, v in vals.items():
        out[f"{k}_ci_lo"] = float(np.quantile(v, 0.025))
        out[f"{k}_ci_hi"] = float(np.quantile(v, 0.975))
    return out

def collect_sets():
    models = [
        "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
        "facebook/bart-large-mnli",
    ]
    sets = {}

    # Evidence-given score files.
    eg = Path("outputs/predictions/evidence_given")
    for dataset in ["vitaminc", "pubhealth", "climate_fever"]:
        for model in models:
            safe = safe_model_name(model)
            tune = read_jsonl(eg / f"{dataset}_tune_{safe}_scores.jsonl")
            cal = read_jsonl(eg / f"{dataset}_cal_{safe}_scores.jsonl")
            dev = read_jsonl(eg / f"{dataset}_dev_{safe}_scores.jsonl")
            if tune and cal and dev:
                sets[(dataset, model)] = (tune, cal, dev)

    # FEVER full-dev score files.
    fv = Path("outputs/predictions/fever")
    for model in models:
        safe = safe_model_name(model)
        tune = read_jsonl(fv / f"fast_full_dev_nli_{safe}_tune_scores.jsonl")
        cal = read_jsonl(fv / f"fast_full_dev_nli_{safe}_cal_scores.jsonl")
        dev = read_jsonl(fv / f"fast_full_dev_nli_{safe}_paper_dev_full_scores.jsonl")
        if tune and cal and dev:
            sets[("fever_full_dev", model)] = (tune, cal, dev)

    return sets

def run_ablation():
    sets = collect_sets()
    print("Score sets found:", len(sets), flush=True)
    for k in sets:
        print("  ", k, flush=True)

    rows = []
    ci_rows = []
    alphas = [0.05, 0.10, 0.20, 0.30]

    for (dataset, model), (tune, cal, dev) in sets.items():
        print(f"\nProcessing ablation: {dataset} / {short_model(model)}", flush=True)

        for rule in ["S", "S+K", "S+K+P"]:
            best = tune_fast(tune, rule)
            preds = [pred_rule(r, rule, best["tau_s"], best["tau_k"], best["margin"])[0] for r in dev]
            m = eval_preds(dev, preds)
            rows.append({
                "dataset": dataset,
                "model": model,
                "rule": rule,
                "alpha": "",
                "tau_s": best["tau_s"],
                "tau_k": best["tau_k"],
                "margin": best["margin"],
                **m,
            })

        best = tune_fast(tune, "S+K+P")
        for alpha in alphas:
            thr = choose_risk_thr(cal, best["tau_s"], best["tau_k"], best["margin"], alpha)
            preds = [
                pred_rule(r, "S+K+P+rho", best["tau_s"], best["tau_k"], best["margin"], risk_thr=thr)[0]
                for r in dev
            ]
            m = eval_preds(dev, preds)
            row = {
                "dataset": dataset,
                "model": model,
                "rule": "S+K+P+rho",
                "alpha": alpha,
                "risk_threshold": thr,
                "tau_s": best["tau_s"],
                "tau_k": best["tau_k"],
                "margin": best["margin"],
                **m,
            }
            rows.append(row)
            if alpha in [0.05, 0.10, 0.30]:
                ci_rows.append({**row, **bootstrap_ci(dev, preds, n_boot=500)})

    df = pd.DataFrame(rows)
    ci = pd.DataFrame(ci_rows)
    df.to_csv(METRIC / "fast_ablation_all.csv", index=False)
    ci.to_csv(METRIC / "fast_bootstrap_ci_selected.csv", index=False)

    compact = []
    for dataset in ["fever_full_dev", "vitaminc", "pubhealth", "climate_fever"]:
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue
        if any(sub["model"].astype(str).str.contains("ynie/roberta")):
            sub = sub[sub["model"].astype(str).str.contains("ynie/roberta")]
        for rule in ["S", "S+K", "S+K+P"]:
            s = sub[sub["rule"] == rule].sort_values("macro_f1", ascending=False).head(1)
            if len(s):
                r = s.iloc[0]
                compact.append({
                    "Dataset": dataset,
                    "Model": short_model(r["model"]),
                    "Rule": rule,
                    "Alpha": "",
                    "Macro-F1": r["macro_f1"],
                    "FVR": r["false_verification_rate"],
                    "Coverage": r["coverage"],
                    "Accepted Acc.": r["accepted_accuracy"],
                })

        s = sub[(sub["rule"] == "S+K+P+rho") & (sub["alpha"].astype(str).isin(["0.1", "0.10"]))]
        if s.empty:
            s = sub[sub["rule"] == "S+K+P+rho"].sort_values(["false_verification_rate", "macro_f1"], ascending=[True, False]).head(1)
        else:
            s = s.sort_values("macro_f1", ascending=False).head(1)
        if len(s):
            r = s.iloc[0]
            compact.append({
                "Dataset": dataset,
                "Model": short_model(r["model"]),
                "Rule": "S+K+P+rho",
                "Alpha": r["alpha"],
                "Macro-F1": r["macro_f1"],
                "FVR": r["false_verification_rate"],
                "Coverage": r["coverage"],
                "Accepted Acc.": r["accepted_accuracy"],
            })

    cdf = pd.DataFrame(compact)
    cdf.to_csv(TABLE / "table_fast_ablation.csv", index=False)

    print("\nCompact ablation table:", flush=True)
    print(cdf.to_string(index=False), flush=True)

    return df, cdf, ci

def make_figures(df):
    risk = df[df["rule"] == "S+K+P+rho"].copy()
    for ds in ["fever_full_dev", "vitaminc", "climate_fever"]:
        sub = risk[risk["dataset"] == ds].copy()
        if sub.empty:
            continue
        if any(sub["model"].astype(str).str.contains("ynie/roberta")):
            sub = sub[sub["model"].astype(str).str.contains("ynie/roberta")]
        sub = sub.sort_values("alpha")

        plt.figure(figsize=(6, 4))
        plt.plot(sub["coverage"], sub["false_verification_rate"], marker="o")
        for _, r in sub.iterrows():
            plt.annotate(str(r["alpha"]), (r["coverage"], r["false_verification_rate"]), fontsize=7)
        plt.xlabel("Coverage")
        plt.ylabel("False-verification rate")
        plt.title(f"Risk-coverage curve: {ds}")
        plt.tight_layout()
        plt.savefig(FIG / f"fig_fast_risk_coverage_{ds}.png", dpi=300)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.plot(sub["false_verification_rate"], sub["macro_f1"], marker="o")
        for _, r in sub.iterrows():
            plt.annotate(str(r["alpha"]), (r["false_verification_rate"], r["macro_f1"]), fontsize=7)
        plt.xlabel("False-verification rate")
        plt.ylabel("Macro-F1")
        plt.title(f"Utility-risk curve: {ds}")
        plt.tight_layout()
        plt.savefig(FIG / f"fig_fast_utility_risk_{ds}.png", dpi=300)
        plt.close()

def write_latex(cdf):
    out = cdf.copy()
    for c in ["Macro-F1", "FVR", "Coverage", "Accepted Acc."]:
        out[c] = out[c].map(lambda x: f"{float(x):.3f}")
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Fast ablation of support, contradiction, provenance, and risk-calibrated gating.}\n"
    tex += "\\label{tab:ablation_rule_components}\n"
    tex += out.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_fast_ablation_rule_components.tex").write_text(tex, encoding="utf-8")

def write_oracle_limitation_note():
    # We avoid rerunning heavy NLI here. We record the experimental need and the reason if not available.
    note = {
        "status": "not_run_in_fast_hardening",
        "reason": "Oracle FEVER NLI requires scoring gold evidence with large NLI models. This should be run only if time permits; current paper can include it as a limitation or future diagnostic if not completed.",
        "paper_language": "Because FEVER-full-dev retrieval uses a sampled evidence corpus and TF-IDF retrieval, verifier and retriever errors cannot be fully separated in that setting. We therefore report retrieval recall alongside verification metrics and treat oracle retrieval as a future diagnostic unless gold-evidence NLI scores are added."
    }
    (METRIC / "fever_oracle_status_note.json").write_text(json.dumps(note, indent=2), encoding="utf-8")

def write_report(cdf, ci):
    report = FINAL / "review_hardening_fast_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Review-Hardening Fast Report\n\n")
        f.write("## Added Artifacts\n\n")
        f.write("- Fast component ablation: S, S+K, S+K+P, and S+K+P+rho.\n")
        f.write("- Bootstrap confidence intervals for selected risk-calibrated rows.\n")
        f.write("- Focused risk-coverage and utility-risk curves for FEVER-full-dev, VitaminC, and Climate-FEVER when score files are available.\n")
        f.write("- Oracle FEVER is recorded as a required limitation/future diagnostic unless separately run.\n\n")

        f.write("## Ablation Table\n\n")
        f.write(cdf.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Bootstrap CI Rows\n\n")
        f.write(ci.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Required Paper Fixes\n\n")
        f.write("1. Use factual-claim framing instead of AI-generated-claim framing unless an LLM-generated claim experiment is added.\n")
        f.write("2. Explain that S and K may be correlated because they can come from the same NLI model, but they are separated operationally for asymmetric thresholding.\n")
        f.write("3. Treat R(c) as an optional domain-specific rule-consistency gate in the current experiments.\n")
        f.write("4. Describe rho(c) as empirical split calibration under exchangeability assumptions, not an unconditional conformal guarantee.\n")
        f.write("5. Reframe FEVEROUS as a structured-provenance characterization study.\n")
        f.write("6. State that FEVER-full-dev uses all paper-dev claims with a large sampled evidence corpus, not exhaustive Wikipedia retrieval.\n")

def main():
    df, cdf, ci = run_ablation()
    make_figures(df)
    write_latex(cdf)
    write_oracle_limitation_note()
    write_report(cdf, ci)

    status = {
        "status": "completed",
        "files": [
            "outputs/tables/review_hardening/table_fast_ablation.csv",
            "outputs/metrics/review_hardening/fast_ablation_all.csv",
            "outputs/metrics/review_hardening/fast_bootstrap_ci_selected.csv",
            "outputs/latex_tables/table_fast_ablation_rule_components.tex",
            "outputs/final_report/review_hardening_fast_report.md",
            "outputs/metrics/review_hardening/fever_oracle_status_note.json"
        ]
    }
    (FINAL / "review_hardening_fast_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    print("\n==== FAST REVIEW HARDENING COMPLETE ====", flush=True)
    print(json.dumps(status, indent=2), flush=True)

if __name__ == "__main__":
    main()
