import json
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
    "verified_safe",
]

ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

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

def pred(rule, r, params):
    S = float(r["S"])
    K = float(r["K"])
    N = float(r["N"])
    P = int(r["P"])

    tau_s = float(params.get("tau_s", 0.5))
    tau_k = float(params.get("tau_k", 0.5))
    margin = float(params.get("margin", 0.0))

    if rule == "support_only":
        return "verified" if S >= tau_s and P == 1 else "unsupported"

    if rule in ["margin_support_refute", "verified_safe"]:
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        if S >= tau_s and (S - K) >= margin and P == 1:
            return "verified"
        return "unsupported"

    raise ValueError(rule)

def confidence_score(rule, r):
    S = float(r["S"])
    K = float(r["K"])
    N = float(r["N"])

    if rule == "support_only":
        # Confidence in verified claim is support strength.
        return S

    if rule in ["margin_support_refute", "verified_safe"]:
        # Confidence in verified claim should be high support and clear separation from contradiction.
        return S - K

    raise ValueError(rule)

def evaluate(rows, predictions):
    y_true = [r["label"] for r in rows]
    y_pred = predictions

    acc_all = accuracy_score(y_true, y_pred)
    macro_all = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
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

    metrics = {
        "accuracy_all": float(acc_all),
        "macro_f1_all": float(macro_all),
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
    return metrics

def get_best_params_from_grid(model_name, rule):
    grid_path = METRIC_DIR / "proper_eval_selected_nli_train_grid.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Missing grid file: {grid_path}")

    df = pd.read_csv(grid_path)
    sub = df[(df["model"] == model_name) & (df["rule"] == rule)].copy()
    if len(sub) == 0:
        raise RuntimeError(f"No grid rows for model={model_name}, rule={rule}")

    # Reuse the objective used in previous proper evaluation.
    best = sub.sort_values("objective", ascending=False).iloc[0].to_dict()

    params = {}
    for col in ["param_tau_s", "param_tau_k", "param_margin"]:
        if col in best and pd.notna(best[col]):
            params[col.replace("param_", "")] = float(best[col])

    return params, best

def base_predictions(rows, rule, params):
    preds = []
    confs = []
    for r in rows:
        p = pred(rule, r, params)
        c = confidence_score(rule, r)
        preds.append(p)
        confs.append(float(c))
    return preds, confs

def choose_verified_conf_threshold(train_rows, train_preds, train_confs, alpha):
    # Among train examples predicted verified, choose the lowest confidence threshold
    # that gives false-verification rate <= alpha, maximizing retained verified coverage.
    candidates = []
    for i, p in enumerate(train_preds):
        if p == "verified":
            candidates.append(train_confs[i])

    if not candidates:
        return float("inf"), {
            "train_num_base_verified": 0,
            "train_num_retained_verified": 0,
            "train_retained_fvr": 0.0,
        }

    thresholds = sorted(set(candidates))

    best = None
    for t in thresholds:
        retained = [
            i for i, p in enumerate(train_preds)
            if p == "verified" and train_confs[i] >= t
        ]

        if not retained:
            continue

        false = sum(1 for i in retained if train_rows[i]["label"] != "verified")
        fvr = false / len(retained)

        if fvr <= alpha:
            record = {
                "threshold": float(t),
                "train_num_base_verified": int(len(candidates)),
                "train_num_retained_verified": int(len(retained)),
                "train_retained_fvr": float(fvr),
                "train_verified_retention": float(len(retained) / len(candidates)),
            }

            # Since thresholds are ascending, first valid threshold retains the most verified claims.
            best = record
            break

    if best is None:
        # No threshold can satisfy alpha except retaining zero. Use threshold above max.
        return float(max(candidates) + 1e-6), {
            "train_num_base_verified": int(len(candidates)),
            "train_num_retained_verified": 0,
            "train_retained_fvr": 0.0,
            "train_verified_retention": 0.0,
        }

    return best["threshold"], best

def apply_verified_risk_gate(rows, base_preds, confs, threshold):
    out_preds = []
    for p, c in zip(base_preds, confs):
        if p == "verified" and c < threshold:
            out_preds.append("abstain")
        else:
            out_preds.append(p)
    return out_preds

def run_one(model_name, rule, alpha):
    train_path = PRED_DIR / f"selected_nli_{safe_name(model_name)}_train_scores.jsonl"
    dev_path = PRED_DIR / f"selected_nli_{safe_name(model_name)}_dev_scores.jsonl"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(f"Missing cached score files for {model_name}")

    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)

    params, train_best = get_best_params_from_grid(model_name, rule)

    train_base_preds, train_confs = base_predictions(train_rows, rule, params)
    dev_base_preds, dev_confs = base_predictions(dev_rows, rule, params)

    base_dev_metrics = evaluate(dev_rows, dev_base_preds)

    threshold, calib_info = choose_verified_conf_threshold(
        train_rows, train_base_preds, train_confs, alpha
    )

    dev_risk_preds = apply_verified_risk_gate(dev_rows, dev_base_preds, dev_confs, threshold)
    dev_metrics = evaluate(dev_rows, dev_risk_preds)

    out_rows = []
    for r, bp, rp, conf in zip(dev_rows, dev_base_preds, dev_risk_preds, dev_confs):
        rr = dict(r)
        rr["base_prediction"] = bp
        rr["risk_calibrated_prediction"] = rp
        rr["verified_confidence"] = float(conf)
        rr["verified_conf_threshold"] = float(threshold)
        rr["alpha"] = float(alpha)
        rr["rule"] = rule
        rr["params"] = params
        out_rows.append(rr)

    dev_metrics.update({
        "model": model_name,
        "rule": rule,
        "alpha": float(alpha),
        "verified_conf_threshold": float(threshold),
        **{f"param_{k}": float(v) for k, v in params.items()},
        **calib_info,
        "base_dev_accuracy_all": base_dev_metrics["accuracy_all"],
        "base_dev_macro_f1_all": base_dev_metrics["macro_f1_all"],
        "base_dev_num_predicted_verified": base_dev_metrics["num_predicted_verified"],
        "base_dev_false_verification_rate": base_dev_metrics["false_verification_rate"],
        "base_dev_coverage": base_dev_metrics["coverage"],
        "selected_train_objective": float(train_best["objective"]),
    })

    return dev_metrics, out_rows

def compact(m):
    keys = [
        "model", "rule", "alpha", "verified_conf_threshold",
        "accuracy_all", "macro_f1_all", "verified_f1", "refuted_f1", "unsupported_f1",
        "num_predicted_verified", "false_verification_rate",
        "num_predicted_refuted", "false_refuted_rate",
        "num_abstained", "coverage", "accepted_accuracy",
        "train_num_base_verified", "train_num_retained_verified",
        "train_retained_fvr", "train_verified_retention",
        "base_dev_macro_f1_all", "base_dev_false_verification_rate",
    ]
    return {k: m.get(k) for k in keys if k in m}

def main():
    all_summary = []

    for model_name in MODELS:
        for rule in RULES:
            for alpha in ALPHAS:
                print("\n" + "=" * 100)
                print(f"model={model_name}")
                print(f"rule={rule}")
                print(f"alpha={alpha}")

                metrics, out_rows = run_one(model_name, rule, alpha)

                print(json.dumps(compact(metrics), indent=2))

                out_path = PRED_DIR / f"risk_calibrated_{safe_name(model_name)}_{rule}_alpha{alpha:.2f}_dev.jsonl"
                metric_path = METRIC_DIR / f"risk_calibrated_{safe_name(model_name)}_{rule}_alpha{alpha:.2f}_dev_metrics.json"

                write_jsonl(out_path, out_rows)
                with metric_path.open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)

                flat = {k: v for k, v in metrics.items() if not isinstance(v, (dict, list))}
                all_summary.append(flat)

    summary_df = pd.DataFrame(all_summary)
    summary_path = METRIC_DIR / "risk_calibrated_verified_acceptance_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 100)
    print("RISK-CALIBRATED VERIFIED ACCEPTANCE SUMMARY")
    display_cols = [
        "model", "rule", "alpha", "macro_f1_all", "false_verification_rate",
        "num_predicted_verified", "num_abstained", "coverage", "accepted_accuracy",
        "base_dev_macro_f1_all", "base_dev_false_verification_rate"
    ]
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved summary: {summary_path}")

if __name__ == "__main__":
    main()
