import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

PRED_DIR = Path("outputs/predictions/scifact")
METRIC_DIR = Path("outputs/metrics/scifact")
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]

MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "facebook/bart-large-mnli",
]

RULES = [
    "support_only",
    "margin_support_refute",
]

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
SEED = 42
TUNE_FRAC = 0.70

def safe_name(model_name):
    return model_name.replace("/", "__")

def read_jsonl(path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def stratified_split(rows, tune_frac=0.7, seed=42):
    rng = random.Random(seed)
    by_label = {}
    for r in rows:
        by_label.setdefault(r["label"], []).append(r)

    tune, cal = [], []
    for label, items in by_label.items():
        items = list(items)
        rng.shuffle(items)
        n_tune = int(round(len(items) * tune_frac))
        tune.extend(items[:n_tune])
        cal.extend(items[n_tune:])

    rng.shuffle(tune)
    rng.shuffle(cal)
    return tune, cal

def pred(rule, r, params):
    S = float(r["S"])
    K = float(r["K"])
    P = int(r["P"])

    tau_s = float(params.get("tau_s", 0.5))
    tau_k = float(params.get("tau_k", 0.5))
    margin = float(params.get("margin", 0.0))

    if rule == "support_only":
        return "verified" if S >= tau_s and P == 1 else "unsupported"

    if rule == "margin_support_refute":
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        if S >= tau_s and (S - K) >= margin and P == 1:
            return "verified"
        return "unsupported"

    raise ValueError(rule)

def verified_conf(rule, r):
    S = float(r["S"])
    K = float(r["K"])

    if rule == "support_only":
        return S

    if rule == "margin_support_refute":
        return S - K

    raise ValueError(rule)

def eval_preds(rows, preds):
    y_true = [r["label"] for r in rows]
    y_pred = preds

    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    pred_verified = [i for i, p in enumerate(y_pred) if p == "verified"]
    fvr = (
        sum(1 for i in pred_verified if y_true[i] != "verified") / len(pred_verified)
        if pred_verified else 0.0
    )

    pred_refuted = [i for i, p in enumerate(y_pred) if p == "refuted"]
    frr = (
        sum(1 for i in pred_refuted if y_true[i] != "refuted") / len(pred_refuted)
        if pred_refuted else 0.0
    )

    abstained = [i for i, p in enumerate(y_pred) if p == "abstain"]
    accepted = [i for i, p in enumerate(y_pred) if p != "abstain"]

    accepted_acc = (
        accuracy_score([y_true[i] for i in accepted], [y_pred[i] for i in accepted])
        if accepted else 0.0
    )

    return {
        "accuracy_all": float(acc),
        "macro_f1_all": float(macro),
        "verified_f1": float(per[0]),
        "refuted_f1": float(per[1]),
        "unsupported_f1": float(per[2]),
        "num_predicted_verified": int(len(pred_verified)),
        "false_verification_rate": float(fvr),
        "num_predicted_refuted": int(len(pred_refuted)),
        "false_refuted_rate": float(frr),
        "num_abstained": int(len(abstained)),
        "coverage": float(len(accepted) / len(rows)),
        "accepted_accuracy": float(accepted_acc),
        "confusion_matrix_labels": LABELS + ["abstain"],
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS + ["abstain"]).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS + ["abstain"], output_dict=True, zero_division=0
        ),
    }

def tune_grid(rule):
    vals = np.round(np.arange(0.05, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.00, 0.81, 0.05), 2)

    if rule == "support_only":
        return [{"tau_s": ts} for ts in vals]

    if rule == "margin_support_refute":
        return [
            {"tau_s": ts, "tau_k": tk, "margin": mg}
            for ts in vals for tk in vals for mg in margins
        ]

    raise ValueError(rule)

def objective(m):
    return (
        m["macro_f1_all"]
        + 0.10 * m["accepted_accuracy"]
        - 0.50 * m["false_verification_rate"]
        - 0.20 * m["false_refuted_rate"]
    )

def tune_base_params(tune_rows, rule):
    best = None
    records = []

    for params in tune_grid(rule):
        preds = [pred(rule, r, params) for r in tune_rows]
        m = eval_preds(tune_rows, preds)
        obj = objective(m)
        m["objective"] = float(obj)
        m.update({f"param_{k}": float(v) for k, v in params.items()})
        records.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})

        if best is None or obj > best["objective"]:
            best = m

    params = {}
    for k, v in best.items():
        if k.startswith("param_"):
            params[k.replace("param_", "")] = float(v)

    return params, best, records

def base_predictions(rows, rule, params):
    preds = [pred(rule, r, params) for r in rows]
    confs = [verified_conf(rule, r) for r in rows]
    return preds, confs

def choose_threshold_empirical(cal_rows, cal_preds, cal_confs, alpha):
    candidates = sorted(set(
        cal_confs[i] for i, p in enumerate(cal_preds) if p == "verified"
    ))

    n_base = sum(1 for p in cal_preds if p == "verified")

    if n_base == 0:
        return float("inf"), {
            "cal_num_base_verified": 0,
            "cal_num_retained_verified": 0,
            "cal_retained_fvr": 0.0,
            "cal_verified_retention": 0.0,
        }

    best = None
    for t in candidates:
        retained = [
            i for i, p in enumerate(cal_preds)
            if p == "verified" and cal_confs[i] >= t
        ]
        if not retained:
            continue

        false = sum(1 for i in retained if cal_rows[i]["label"] != "verified")
        fvr = false / len(retained)

        if fvr <= alpha:
            best = {
                "threshold": float(t),
                "cal_num_base_verified": int(n_base),
                "cal_num_retained_verified": int(len(retained)),
                "cal_retained_fvr": float(fvr),
                "cal_verified_retention": float(len(retained) / n_base),
            }
            break

    if best is None:
        return float(max(candidates) + 1e-6), {
            "cal_num_base_verified": int(n_base),
            "cal_num_retained_verified": 0,
            "cal_retained_fvr": 0.0,
            "cal_verified_retention": 0.0,
        }

    return best["threshold"], best

def apply_gate(base_preds, confs, threshold):
    out = []
    for p, c in zip(base_preds, confs):
        if p == "verified" and c < threshold:
            out.append("abstain")
        else:
            out.append(p)
    return out

def run_one(model_name, rule, alpha):
    train_path = PRED_DIR / f"selected_nli_{safe_name(model_name)}_train_scores.jsonl"
    dev_path = PRED_DIR / f"selected_nli_{safe_name(model_name)}_dev_scores.jsonl"

    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)

    tune_rows, cal_rows = stratified_split(train_rows, tune_frac=TUNE_FRAC, seed=SEED)

    params, tune_best, tune_records = tune_base_params(tune_rows, rule)

    tune_base_preds, tune_confs = base_predictions(tune_rows, rule, params)
    cal_base_preds, cal_confs = base_predictions(cal_rows, rule, params)
    dev_base_preds, dev_confs = base_predictions(dev_rows, rule, params)

    base_tune_metrics = eval_preds(tune_rows, tune_base_preds)
    base_cal_metrics = eval_preds(cal_rows, cal_base_preds)
    base_dev_metrics = eval_preds(dev_rows, dev_base_preds)

    threshold, cal_info = choose_threshold_empirical(
        cal_rows, cal_base_preds, cal_confs, alpha
    )

    dev_risk_preds = apply_gate(dev_base_preds, dev_confs, threshold)
    dev_metrics = eval_preds(dev_rows, dev_risk_preds)

    out_rows = []
    for r, base_p, risk_p, conf in zip(dev_rows, dev_base_preds, dev_risk_preds, dev_confs):
        rr = dict(r)
        rr["base_prediction"] = base_p
        rr["risk_calibrated_prediction"] = risk_p
        rr["verified_confidence"] = float(conf)
        rr["verified_conf_threshold"] = float(threshold)
        rr["alpha"] = float(alpha)
        rr["rule"] = rule
        rr["params"] = params
        out_rows.append(rr)

    metrics = {
        "model": model_name,
        "rule": rule,
        "alpha": float(alpha),
        "seed": SEED,
        "tune_frac": TUNE_FRAC,
        "num_tune": len(tune_rows),
        "num_cal": len(cal_rows),
        "verified_conf_threshold": float(threshold),
        **{f"param_{k}": float(v) for k, v in params.items()},
        **cal_info,

        "tune_base_macro_f1": base_tune_metrics["macro_f1_all"],
        "tune_base_false_verification_rate": base_tune_metrics["false_verification_rate"],
        "cal_base_macro_f1": base_cal_metrics["macro_f1_all"],
        "cal_base_false_verification_rate": base_cal_metrics["false_verification_rate"],
        "dev_base_macro_f1": base_dev_metrics["macro_f1_all"],
        "dev_base_false_verification_rate": base_dev_metrics["false_verification_rate"],

        **dev_metrics,
    }

    return metrics, out_rows, tune_records

def compact(m):
    keys = [
        "model", "rule", "alpha", "verified_conf_threshold",
        "param_tau_s", "param_tau_k", "param_margin",
        "macro_f1_all", "false_verification_rate",
        "num_predicted_verified", "num_abstained",
        "coverage", "accepted_accuracy",
        "cal_num_base_verified", "cal_num_retained_verified",
        "cal_retained_fvr", "cal_verified_retention",
        "dev_base_macro_f1", "dev_base_false_verification_rate",
    ]
    return {k: m.get(k) for k in keys if k in m}

def main():
    all_summary = []
    all_tune = []

    for model_name in MODELS:
        for rule in RULES:
            for alpha in ALPHAS:
                print("\n" + "=" * 100)
                print(f"model={model_name}")
                print(f"rule={rule}")
                print(f"alpha={alpha}")

                metrics, out_rows, tune_records = run_one(model_name, rule, alpha)

                print(json.dumps(compact(metrics), indent=2))

                out_path = PRED_DIR / f"proper_split_risk_{safe_name(model_name)}_{rule}_alpha{alpha:.2f}_dev.jsonl"
                metric_path = METRIC_DIR / f"proper_split_risk_{safe_name(model_name)}_{rule}_alpha{alpha:.2f}_dev_metrics.json"

                write_jsonl(out_path, out_rows)
                with metric_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                all_summary.append({k: v for k, v in metrics.items() if not isinstance(v, (dict, list))})

                for tr in tune_records:
                    tr["model"] = model_name
                    tr["rule"] = rule
                    all_tune.append(tr)

    summary_df = pd.DataFrame(all_summary)
    summary_path = METRIC_DIR / "proper_split_risk_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    tune_path = METRIC_DIR / "proper_split_risk_tune_grid.csv"
    pd.DataFrame(all_tune).to_csv(tune_path, index=False)

    print("\n" + "=" * 100)
    print("PROPER SPLIT RISK CALIBRATION SUMMARY")
    display_cols = [
        "model", "rule", "alpha",
        "macro_f1_all", "false_verification_rate",
        "num_predicted_verified", "num_abstained",
        "coverage", "accepted_accuracy",
        "cal_retained_fvr", "cal_verified_retention",
        "dev_base_macro_f1", "dev_base_false_verification_rate",
    ]
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved tune grid: {tune_path}")

if __name__ == "__main__":
    main()
