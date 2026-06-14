import os
import re
import gc
import json
import random
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

ROOT = Path(".")
FINAL = Path("outputs/final_report")
METRIC = Path("outputs/metrics/review_hardening")
TABLE = Path("outputs/tables/review_hardening")
PRED = Path("outputs/predictions/review_hardening")
FIG = Path("outputs/figures")
TEX = Path("outputs/latex_tables")

for d in [FINAL, METRIC, TABLE, PRED, FIG, TEX]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)

LABELS = ["verified", "refuted", "unsupported"]
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BOOT_N = 1000

MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "facebook/bart-large-mnli",
]

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def safe_model_name(model):
    return model.replace("/", "__")

def short_model(x):
    x = str(x)
    if "ynie/roberta" in x:
        return "RoBERTa-large NLI"
    if "facebook/bart" in x:
        return "BART-large MNLI"
    return x

def clean_rule(x):
    return {
        "support_only": "S only",
        "support_contradiction": "S+K",
        "support_contradiction_provenance": "S+K+P",
        "full_risk_calibrated": "S+K+P+rho",
        "margin_support_refute": "Support-refute margin",
    }.get(str(x), str(x))

def eval_preds(rows, preds):
    y_true = [r["label"] for r in rows]
    y_pred = list(preds)

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    pred_verified = [i for i, p in enumerate(y_pred) if p == "verified"]
    fvr = sum(1 for i in pred_verified if y_true[i] != "verified") / len(pred_verified) if pred_verified else 0.0

    pred_refuted = [i for i, p in enumerate(y_pred) if p == "refuted"]
    frr = sum(1 for i in pred_refuted if y_true[i] != "refuted") / len(pred_refuted) if pred_refuted else 0.0

    abstained = [i for i, p in enumerate(y_pred) if p == "abstain"]
    accepted = [i for i, p in enumerate(y_pred) if p != "abstain"]
    accepted_acc = accuracy_score([y_true[i] for i in accepted], [y_pred[i] for i in accepted]) if accepted else 0.0

    return {
        "accuracy": float(acc),
        "macro_f1": float(macro),
        "verified_f1": float(per[0]),
        "refuted_f1": float(per[1]),
        "unsupported_f1": float(per[2]),
        "num_predicted_verified": len(pred_verified),
        "false_verification_rate": float(fvr),
        "num_predicted_refuted": len(pred_refuted),
        "false_refuted_rate": float(frr),
        "num_abstained": len(abstained),
        "coverage": float(len(accepted) / len(rows)) if rows else 0.0,
        "accepted_accuracy": float(accepted_acc),
    }

def pred_rule(r, rule, params):
    S = float(r.get("S", 0.0))
    K = float(r.get("K", 0.0))
    Pval = int(r.get("P", 1))
    tau_s = float(params.get("tau_s", 0.5))
    tau_k = float(params.get("tau_k", 0.5))
    margin = float(params.get("margin", 0.0))

    if rule == "support_only":
        return "verified" if S >= tau_s else "unsupported"

    if rule == "support_contradiction":
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        if S >= tau_s and (S - K) >= margin:
            return "verified"
        return "unsupported"

    if rule == "support_contradiction_provenance":
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        if S >= tau_s and (S - K) >= margin and Pval == 1:
            return "verified"
        return "unsupported"

    raise ValueError(rule)

def verified_conf(r, rule):
    if rule == "support_only":
        return float(r.get("S", 0.0))
    return float(r.get("S", 0.0)) - float(r.get("K", 0.0))

def tune_params(rows, rule):
    vals = np.round(np.arange(0.05, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.0, 0.81, 0.05), 2)

    if rule == "support_only":
        grid = [{"tau_s": ts} for ts in vals]
    else:
        grid = [{"tau_s": ts, "tau_k": tk, "margin": mg} for ts in vals for tk in vals for mg in margins]

    best = None
    for params in grid:
        preds = [pred_rule(r, rule, params) for r in rows]
        m = eval_preds(rows, preds)
        obj = m["macro_f1"] + 0.10 * m["accepted_accuracy"] - 0.50 * m["false_verification_rate"] - 0.20 * m["false_refuted_rate"]
        if best is None or obj > best["objective"]:
            best = {"params": params, "objective": obj, "metrics": m}

    return best["params"], best["metrics"]

def choose_threshold(cal_rows, cal_preds, cal_confs, alpha):
    idxs = [i for i, p in enumerate(cal_preds) if p == "verified"]
    if not idxs:
        return float("inf"), {"cal_retained_fvr": 0.0, "cal_verified_retention": 0.0, "cal_base_verified": 0, "cal_retained_verified": 0}

    thresholds = sorted(set(cal_confs[i] for i in idxs))
    for t in thresholds:
        retained = [i for i in idxs if cal_confs[i] >= t]
        if not retained:
            continue
        fvr = sum(1 for i in retained if cal_rows[i]["label"] != "verified") / len(retained)
        if fvr <= alpha:
            return float(t), {
                "cal_retained_fvr": float(fvr),
                "cal_verified_retention": float(len(retained) / len(idxs)),
                "cal_base_verified": len(idxs),
                "cal_retained_verified": len(retained),
            }

    return float(max(thresholds) + 1e-6), {
        "cal_retained_fvr": 0.0,
        "cal_verified_retention": 0.0,
        "cal_base_verified": len(idxs),
        "cal_retained_verified": 0,
    }

def apply_gate(preds, confs, threshold):
    return ["abstain" if p == "verified" and c < threshold else p for p, c in zip(preds, confs)]

def bootstrap_ci(rows, preds, n_boot=BOOT_N):
    if not rows:
        return {}
    y_true = np.array([r["label"] for r in rows])
    y_pred = np.array(preds)
    n = len(rows)

    macro_vals, fvr_vals, cov_vals, acc_vals = [], [], [], []

    for _ in range(n_boot):
        idx = RNG.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred[idx]
        macro_vals.append(f1_score(yt, yp, labels=LABELS, average="macro", zero_division=0))
        acc_vals.append(accuracy_score(yt, yp))
        accepted = yp != "abstain"
        cov_vals.append(float(np.mean(accepted)))
        pv = yp == "verified"
        if np.sum(pv) > 0:
            fvr_vals.append(float(np.mean(yt[pv] != "verified")))
        else:
            fvr_vals.append(0.0)

    def ci(vals):
        vals = np.asarray(vals, dtype=float)
        return {
            "mean": float(np.mean(vals)),
            "lo": float(np.quantile(vals, 0.025)),
            "hi": float(np.quantile(vals, 0.975)),
        }

    return {
        "macro_f1": ci(macro_vals),
        "accuracy": ci(acc_vals),
        "fvr": ci(fvr_vals),
        "coverage": ci(cov_vals),
    }

def collect_score_sets():
    sets = {}

    # Evidence-given datasets.
    eg_dir = Path("outputs/predictions/evidence_given")
    for dataset in ["vitaminc", "pubhealth", "climate_fever"]:
        for model in MODELS:
            safe = safe_model_name(model)
            tune = read_jsonl(eg_dir / f"{dataset}_tune_{safe}_scores.jsonl")
            cal = read_jsonl(eg_dir / f"{dataset}_cal_{safe}_scores.jsonl")
            dev = read_jsonl(eg_dir / f"{dataset}_dev_{safe}_scores.jsonl")
            if tune and cal and dev:
                sets[(dataset, model)] = {"tune": tune, "cal": cal, "dev": dev}

    # FEVER full-dev fast scores.
    fever_dir = Path("outputs/predictions/fever")
    for model in MODELS:
        safe = safe_model_name(model)
        tune = read_jsonl(fever_dir / f"fast_full_dev_nli_{safe}_tune_scores.jsonl")
        cal = read_jsonl(fever_dir / f"fast_full_dev_nli_{safe}_cal_scores.jsonl")
        dev = read_jsonl(fever_dir / f"fast_full_dev_nli_{safe}_paper_dev_full_scores.jsonl")
        if tune and cal and dev:
            sets[("fever_full_dev", model)] = {"tune": tune, "cal": cal, "dev": dev}

    return sets

def run_ablation():
    print("\n==== Ablation: S-only vs S+K vs S+K+P vs S+K+P+rho ====")
    sets = collect_score_sets()
    print("Found score sets:", list(sets.keys()))

    rows_out = []
    selected_pred_records = []

    for (dataset, model), splits in sets.items():
        tune, cal, dev = splits["tune"], splits["cal"], splits["dev"]

        for rule in ["support_only", "support_contradiction", "support_contradiction_provenance"]:
            params, tune_metrics = tune_params(tune, rule)
            base_dev_preds = [pred_rule(r, rule, params) for r in dev]
            base_metrics = eval_preds(dev, base_dev_preds)

            row = {
                "dataset": dataset,
                "model": model,
                "ablation": rule,
                "alpha": "",
                **{f"param_{k}": v for k, v in params.items()},
                **base_metrics,
            }
            rows_out.append(row)

            # Full risk gate is built on S+K+P rule.
            if rule == "support_contradiction_provenance":
                cal_preds = [pred_rule(r, rule, params) for r in cal]
                cal_confs = [verified_conf(r, rule) for r in cal]
                dev_confs = [verified_conf(r, rule) for r in dev]

                for alpha in ALPHAS:
                    thr, cal_info = choose_threshold(cal, cal_preds, cal_confs, alpha)
                    gated = apply_gate(base_dev_preds, dev_confs, thr)
                    m = eval_preds(dev, gated)
                    risk_row = {
                        "dataset": dataset,
                        "model": model,
                        "ablation": "full_risk_calibrated",
                        "alpha": alpha,
                        "threshold": thr,
                        **{f"param_{k}": v for k, v in params.items()},
                        **cal_info,
                        **m,
                    }
                    rows_out.append(risk_row)

                    if alpha in [0.05, 0.10, 0.30]:
                        ci = bootstrap_ci(dev, gated)
                        selected_pred_records.append({
                            "dataset": dataset,
                            "model": model,
                            "ablation": "full_risk_calibrated",
                            "alpha": alpha,
                            "macro_f1": m["macro_f1"],
                            "macro_f1_ci_lo": ci["macro_f1"]["lo"],
                            "macro_f1_ci_hi": ci["macro_f1"]["hi"],
                            "fvr": m["false_verification_rate"],
                            "fvr_ci_lo": ci["fvr"]["lo"],
                            "fvr_ci_hi": ci["fvr"]["hi"],
                            "coverage": m["coverage"],
                            "coverage_ci_lo": ci["coverage"]["lo"],
                            "coverage_ci_hi": ci["coverage"]["hi"],
                            "accepted_accuracy": m["accepted_accuracy"],
                        })

    df = pd.DataFrame(rows_out)
    ci_df = pd.DataFrame(selected_pred_records)

    df.to_csv(METRIC / "ablation_all_score_sets.csv", index=False)
    ci_df.to_csv(METRIC / "bootstrap_ci_selected_ablation_rows.csv", index=False)

    # Paper compact ablation table: choose one retrieval-grounded and one evidence-given.
    compact_rows = []
    for dataset in ["fever_full_dev", "vitaminc", "pubhealth", "climate_fever"]:
        sub = df[df["dataset"] == dataset].copy()
        if sub.empty:
            continue

        # Prefer RoBERTa if present.
        if any(sub["model"].astype(str).str.contains("ynie/roberta")):
            sub = sub[sub["model"].astype(str).str.contains("ynie/roberta")]

        # Add S-only, S+K, S+K+P, and best alpha=0.10 full if available.
        for ab in ["support_only", "support_contradiction", "support_contradiction_provenance"]:
            s = sub[sub["ablation"] == ab].sort_values("macro_f1", ascending=False).head(1)
            if len(s):
                r = s.iloc[0]
                compact_rows.append({
                    "Dataset": dataset,
                    "Model": short_model(r["model"]),
                    "Rule": clean_rule(ab),
                    "Alpha": "",
                    "Macro-F1": r["macro_f1"],
                    "FVR": r["false_verification_rate"],
                    "Coverage": r["coverage"],
                    "Accepted Acc.": r["accepted_accuracy"],
                })

        full = sub[(sub["ablation"] == "full_risk_calibrated") & (sub["alpha"].astype(str).isin(["0.1", "0.10"]))]
        if full.empty:
            full = sub[sub["ablation"] == "full_risk_calibrated"].sort_values(["false_verification_rate", "macro_f1"], ascending=[True, False]).head(1)
        else:
            full = full.sort_values("macro_f1", ascending=False).head(1)

        if len(full):
            r = full.iloc[0]
            compact_rows.append({
                "Dataset": dataset,
                "Model": short_model(r["model"]),
                "Rule": "S+K+P+rho",
                "Alpha": r["alpha"],
                "Macro-F1": r["macro_f1"],
                "FVR": r["false_verification_rate"],
                "Coverage": r["coverage"],
                "Accepted Acc.": r["accepted_accuracy"],
            })

    compact = pd.DataFrame(compact_rows)
    compact.to_csv(TABLE / "table_ablation_support_contradiction_provenance_risk.csv", index=False)

    print("Saved ablation table:", TABLE / "table_ablation_support_contradiction_provenance_risk.csv")
    print(compact.to_string(index=False))

    return df, compact

def sent_id_from_ev(ev):
    wiki = str(ev.get("wiki_url", "")).replace(" ", "_")
    try:
        sid = int(ev.get("sentence_id", -1))
    except Exception:
        sid = -1
    if wiki and sid >= 0:
        return f"{wiki}::{sid}"
    return None

def load_fever_corpus_map():
    corpus_path = Path("data/processed/fever/full_dev_sentence_corpus.jsonl")
    rows = read_jsonl(corpus_path)
    return {r["sent_id"]: r for r in rows}

def make_oracle_rows(split_name, claims, corpus_map):
    out = []
    found, total = 0, 0

    for r in claims:
        units = []
        ids = []
        for ev in r.get("gold_evidence", []) or []:
            sid = sent_id_from_ev(ev)
            if not sid:
                continue
            total += 1
            item = corpus_map.get(sid)
            if item:
                found += 1
                ids.append(sid)
                units.append((item.get("wiki_url", "") + ". " + item.get("text", "")).strip())

        # Deduplicate and limit.
        seen = set()
        units2 = []
        ids2 = []
        for sid, txt in zip(ids, units):
            if sid not in seen and txt:
                seen.add(sid)
                ids2.append(sid)
                units2.append(txt)
        units2 = units2[:10]
        ids2 = ids2[:10]

        out.append({
            "id": r.get("id"),
            "dataset": "fever_oracle_full_dev",
            "split": split_name,
            "claim": r.get("claim", ""),
            "label": r.get("label", ""),
            "evidence_units": units2,
            "gold_sentence_ids": ids2,
            "P": 1 if units2 else 0,
        })

    stats = {
        "split": split_name,
        "claims": len(claims),
        "gold_evidence_ids_total": total,
        "gold_evidence_ids_found_in_corpus": found,
        "gold_evidence_id_found_rate": found / total if total else None,
        "claims_with_oracle_evidence": sum(1 for r in out if r["P"] == 1),
    }

    return out, stats

def infer_mapping(model):
    id2label = model.config.id2label
    mapping = {}
    for idx, label in id2label.items():
        l = str(label).lower()
        if "entail" in l:
            mapping["entailment"] = int(idx)
        elif "contrad" in l:
            mapping["contradiction"] = int(idx)
        elif "neutral" in l:
            mapping["neutral"] = int(idx)
    if len(id2label) == 3 and ("entailment" not in mapping or "contradiction" not in mapping):
        mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}
    return mapping

def score_evidence_units(model_name, dataset, split, rows, device):
    safe = safe_model_name(model_name)
    out_path = PRED / f"{dataset}_{split}_{safe}_oracle_scores.jsonl"
    if out_path.exists():
        print("Loading cached oracle scores:", out_path)
        return read_jsonl(out_path)

    print("Loading oracle NLI model:", model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    mapping = infer_mapping(model)
    print("Mapping:", mapping)

    claims, premises, refs = [], [], []
    for i, r in enumerate(rows):
        for j, ev in enumerate(r.get("evidence_units", [])[:10]):
            claims.append(r["claim"])
            premises.append(ev)
            refs.append((i, j))

    scores_by_i = defaultdict(list)
    with torch.no_grad():
        for start in tqdm(range(0, len(claims), 32), desc=f"Oracle NLI {split} {model_name}"):
            bc = claims[start:start+32]
            bp = premises[start:start+32]
            enc = tok(bp, bc, padding=True, truncation=True, max_length=256, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).detach().cpu().numpy()
            for row, ref in zip(probs, refs[start:start+32]):
                scores_by_i[ref[0]].append({
                    "unit_index": ref[1],
                    "entailment": float(row[mapping["entailment"]]),
                    "contradiction": float(row[mapping["contradiction"]]),
                    "neutral": float(row[mapping["neutral"]]),
                })

    out = []
    for i, r in enumerate(rows):
        sc = scores_by_i.get(i, [])
        out.append({
            "id": r["id"],
            "dataset": dataset,
            "split": split,
            "claim": r["claim"],
            "label": r["label"],
            "model": model_name,
            "S": max([x["entailment"] for x in sc], default=0.0),
            "K": max([x["contradiction"] for x in sc], default=0.0),
            "N": max([x["neutral"] for x in sc], default=1.0),
            "P": int(r.get("P", 0)),
            "num_evidence_units": len(r.get("evidence_units", [])),
            "unit_scores": sc,
        })

    write_jsonl(out_path, out)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return out

def run_oracle_fever():
    print("\n==== FEVER oracle-evidence diagnostic ====")

    tune_claims = read_jsonl("data/processed/fever/full_dev_tune_claims.jsonl")
    cal_claims = read_jsonl("data/processed/fever/full_dev_cal_claims.jsonl")
    dev_claims = read_jsonl("data/processed/fever/full_dev_paper_dev_claims.jsonl")

    if not (tune_claims and cal_claims and dev_claims):
        print("WARNING: missing FEVER full-dev claims; skipping oracle experiment.")
        return pd.DataFrame()

    corpus_map = load_fever_corpus_map()
    print("Loaded corpus map:", len(corpus_map))

    tune_oracle, s1 = make_oracle_rows("tune", tune_claims, corpus_map)
    cal_oracle, s2 = make_oracle_rows("cal", cal_claims, corpus_map)
    dev_oracle, s3 = make_oracle_rows("dev", dev_claims, corpus_map)

    write_jsonl(PRED / "fever_oracle_full_dev_tune_inputs.jsonl", tune_oracle)
    write_jsonl(PRED / "fever_oracle_full_dev_cal_inputs.jsonl", cal_oracle)
    write_jsonl(PRED / "fever_oracle_full_dev_dev_inputs.jsonl", dev_oracle)

    oracle_stats = [s1, s2, s3]
    pd.DataFrame(oracle_stats).to_csv(METRIC / "fever_oracle_evidence_recovery_stats.csv", index=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    rows_out = []
    for model_name in MODELS:
        tune_scores = score_evidence_units(model_name, "fever_oracle_full_dev", "tune", tune_oracle, device)
        cal_scores = score_evidence_units(model_name, "fever_oracle_full_dev", "cal", cal_oracle, device)
        dev_scores = score_evidence_units(model_name, "fever_oracle_full_dev", "dev", dev_oracle, device)

        for base_rule in ["support_only", "support_contradiction_provenance"]:
            params, _ = tune_params(tune_scores, base_rule)
            dev_preds = [pred_rule(r, base_rule, params) for r in dev_scores]
            base_m = eval_preds(dev_scores, dev_preds)

            rows_out.append({
                "dataset": "FEVER-oracle-full-dev",
                "model": model_name,
                "rule": base_rule,
                "alpha": "",
                **{f"param_{k}": v for k, v in params.items()},
                **base_m,
            })

            if base_rule == "support_contradiction_provenance":
                cal_preds = [pred_rule(r, base_rule, params) for r in cal_scores]
                cal_confs = [verified_conf(r, base_rule) for r in cal_scores]
                dev_confs = [verified_conf(r, base_rule) for r in dev_scores]

                for alpha in ALPHAS:
                    thr, cal_info = choose_threshold(cal_scores, cal_preds, cal_confs, alpha)
                    gated = apply_gate(dev_preds, dev_confs, thr)
                    m = eval_preds(dev_scores, gated)
                    rows_out.append({
                        "dataset": "FEVER-oracle-full-dev",
                        "model": model_name,
                        "rule": "full_risk_calibrated",
                        "alpha": alpha,
                        "threshold": thr,
                        **{f"param_{k}": v for k, v in params.items()},
                        **cal_info,
                        **m,
                    })

    df = pd.DataFrame(rows_out)
    df.to_csv(METRIC / "fever_oracle_full_dev_risk_calibration_summary.csv", index=False)

    compact = []
    for model_name in MODELS:
        sub = df[df["model"] == model_name]
        best = sub.sort_values("macro_f1", ascending=False).head(1)
        low = sub[(sub["coverage"] >= 0.75) & (sub["rule"] == "full_risk_calibrated")].sort_values(["false_verification_rate", "macro_f1"], ascending=[True, False]).head(1)
        for sel, s in [("oracle best macro-F1", best), ("oracle lowest FVR coverage>=0.75", low)]:
            if len(s):
                r = s.iloc[0]
                compact.append({
                    "Selection": sel,
                    "Model": short_model(model_name),
                    "Rule": clean_rule(r["rule"]),
                    "Alpha": r["alpha"],
                    "Macro-F1": r["macro_f1"],
                    "FVR": r["false_verification_rate"],
                    "Coverage": r["coverage"],
                    "Accepted Acc.": r["accepted_accuracy"],
                })

    cdf = pd.DataFrame(compact)
    cdf.to_csv(TABLE / "table_fever_oracle_full_dev.csv", index=False)

    print("Oracle FEVER evidence recovery:")
    print(pd.DataFrame(oracle_stats).to_string(index=False))
    print("Oracle FEVER compact:")
    print(cdf.to_string(index=False))

    return df

def make_curves_and_tables(ablation_df, oracle_df):
    print("\n==== Make figures, LaTeX, and final review-hardening report ====")

    # Focused curves from ablation risk rows.
    risk = ablation_df[ablation_df["ablation"] == "full_risk_calibrated"].copy()
    if not risk.empty:
        for ds in ["fever_full_dev", "vitaminc", "climate_fever"]:
            sub = risk[risk["dataset"] == ds].copy()
            if sub.empty:
                continue
            # Prefer RoBERTa.
            if any(sub["model"].astype(str).str.contains("ynie/roberta")):
                sub = sub[sub["model"].astype(str).str.contains("ynie/roberta")]
            sub = sub.sort_values("alpha")

            import matplotlib.pyplot as plt

            plt.figure(figsize=(6, 4))
            plt.plot(sub["coverage"], sub["false_verification_rate"], marker="o")
            for _, r in sub.iterrows():
                plt.annotate(str(r["alpha"]), (r["coverage"], r["false_verification_rate"]), fontsize=7)
            plt.xlabel("Coverage")
            plt.ylabel("False-verification rate")
            plt.title(f"Risk-coverage curve: {ds}")
            plt.tight_layout()
            plt.savefig(FIG / f"fig_risk_coverage_{ds}.png", dpi=300)
            plt.close()

            plt.figure(figsize=(6, 4))
            plt.plot(sub["false_verification_rate"], sub["macro_f1"], marker="o")
            for _, r in sub.iterrows():
                plt.annotate(str(r["alpha"]), (r["false_verification_rate"], r["macro_f1"]), fontsize=7)
            plt.xlabel("False-verification rate")
            plt.ylabel("Macro-F1")
            plt.title(f"Utility-risk curve: {ds}")
            plt.tight_layout()
            plt.savefig(FIG / f"fig_utility_risk_{ds}.png", dpi=300)
            plt.close()

    # LaTeX ablation table.
    abl = pd.read_csv(TABLE / "table_ablation_support_contradiction_provenance_risk.csv")
    tex_abl = abl.copy()
    for c in ["Macro-F1", "FVR", "Coverage", "Accepted Acc."]:
        tex_abl[c] = tex_abl[c].map(lambda x: f"{float(x):.3f}" if str(x) != "nan" else "")
    tex = "\\begin{table}[t]\n\\centering\n\\small\n"
    tex += "\\caption{Ablation of support, contradiction, provenance, and risk-calibrated gating.}\n"
    tex += "\\label{tab:ablation_rule_components}\n"
    tex += tex_abl.to_latex(index=False, escape=True)
    tex += "\\end{table}\n"
    (TEX / "table_ablation_rule_components.tex").write_text(tex, encoding="utf-8")

    # LaTeX oracle table.
    oracle_table = TABLE / "table_fever_oracle_full_dev.csv"
    if oracle_table.exists():
        odf = pd.read_csv(oracle_table)
        tex_o = odf.copy()
        for c in ["Macro-F1", "FVR", "Coverage", "Accepted Acc."]:
            tex_o[c] = tex_o[c].map(lambda x: f"{float(x):.3f}" if str(x) != "nan" else "")
        tex = "\\begin{table}[t]\n\\centering\n\\small\n"
        tex += "\\caption{FEVER-full-dev oracle-evidence diagnostic, isolating verifier behavior from retrieval errors.}\n"
        tex += "\\label{tab:fever_oracle_diagnostic}\n"
        tex += tex_o.to_latex(index=False, escape=True)
        tex += "\\end{table}\n"
        (TEX / "table_fever_oracle_diagnostic.tex").write_text(tex, encoding="utf-8")

    # Final report.
    report = FINAL / "review_hardening_report.md"
    with report.open("w", encoding="utf-8") as f:
        f.write("# Review-Hardening Experimental Report\n\n")

        f.write("## Added Experiments\n\n")
        f.write("1. Component ablation: S only, S+K, S+K+P, and S+K+P+rho.\n")
        f.write("2. FEVER oracle-evidence diagnostic to isolate verifier behavior from retrieval errors.\n")
        f.write("3. Bootstrap confidence intervals for selected risk-calibrated rows.\n")
        f.write("4. Focused risk-coverage and utility-risk curves for representative datasets.\n\n")

        f.write("## Component Ablation\n\n")
        f.write(abl.to_markdown(index=False))
        f.write("\n\n")

        if oracle_table.exists():
            f.write("## FEVER Oracle-Evidence Diagnostic\n\n")
            f.write(pd.read_csv(oracle_table).to_markdown(index=False))
            f.write("\n\n")

        ci_path = METRIC / "bootstrap_ci_selected_ablation_rows.csv"
        if ci_path.exists():
            f.write("## Bootstrap Confidence Intervals\n\n")
            f.write(pd.read_csv(ci_path).to_markdown(index=False))
            f.write("\n\n")

        f.write("## Paper Fixes Required by the Review\n\n")
        f.write("- Use title/framing around factual claims, not only AI-generated claims.\n")
        f.write("- Present R(c) as optional domain-specific rule consistency unless operationalized.\n")
        f.write("- Describe rho(c) as empirical split calibration under exchangeability assumptions, not an unconditional conformal guarantee.\n")
        f.write("- Reframe FEVEROUS as a structured-provenance characterization study.\n")
        f.write("- State that FEVER-full-dev uses all paper-dev claims but a large sampled evidence corpus, not exhaustive all-Wikipedia retrieval.\n")

    print("Saved report:", report)

def main():
    ablation_df, compact = run_ablation()
    oracle_df = run_oracle_fever()
    make_curves_and_tables(ablation_df, oracle_df)

    status = {
        "review_hardening_status": "completed",
        "artifacts": {
            "ablation_csv": str(TABLE / "table_ablation_support_contradiction_provenance_risk.csv"),
            "ablation_latex": str(TEX / "table_ablation_rule_components.tex"),
            "oracle_csv": str(TABLE / "table_fever_oracle_full_dev.csv"),
            "oracle_latex": str(TEX / "table_fever_oracle_diagnostic.tex"),
            "bootstrap_ci": str(METRIC / "bootstrap_ci_selected_ablation_rows.csv"),
            "report": str(FINAL / "review_hardening_report.md"),
        },
        "paper_framing_fixes": [
            "Change title/framing to factual claims or risk-calibrated factual claim verification.",
            "R(c) should be optional/domain-specific if not empirically operationalized.",
            "rho(c) is empirical split calibration, not a distribution-free conformal guarantee.",
            "FEVEROUS is a characterization study.",
        ],
    }
    (FINAL / "review_hardening_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")

    print("\n==== REVIEW HARDENING COMPLETE ====")
    print(json.dumps(status, indent=2))

if __name__ == "__main__":
    main()
