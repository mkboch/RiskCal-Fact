import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

PRED_DIR = Path("outputs/predictions/scifact")
METRIC_DIR = Path("outputs/metrics/scifact")
METRIC_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]

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

def predict(S, K, P, tau_s, tau_k, mode):
    if mode == "support_only":
        if S >= tau_s and P == 1:
            return "verified"
        return "unsupported"

    if mode == "support_contradiction":
        if K >= tau_k:
            return "refuted"
        if S >= tau_s and P == 1:
            return "verified"
        return "unsupported"

    if mode == "ours_no_abstain":
        # Same decision rule but without calibrated abstention yet.
        if K >= tau_k:
            return "refuted"
        if S >= tau_s and K < tau_k and P == 1:
            return "verified"
        return "unsupported"

    raise ValueError(f"Unknown mode: {mode}")

def evaluate_examples(rows, tau_s, tau_k, mode):
    y_true = []
    y_pred = []
    output = []

    for r in rows:
        S = float(r["S_support"])
        K = float(r["K_contradiction"])
        P = int(r["P_provenance"])
        pred = predict(S, K, P, tau_s, tau_k, mode)

        y_true.append(r["label"])
        y_pred.append(pred)

        rr = dict(r)
        rr["prediction_calibrated"] = pred
        rr["calibrated_tau_s"] = tau_s
        rr["calibrated_tau_k"] = tau_k
        rr["calibration_mode"] = mode
        output.append(rr)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per_f1 = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    pred_verified_idx = [i for i, p in enumerate(y_pred) if p == "verified"]
    if pred_verified_idx:
        false_verified = sum(1 for i in pred_verified_idx if y_true[i] != "verified")
        fvr = false_verified / len(pred_verified_idx)
    else:
        fvr = 0.0

    pred_refuted_idx = [i for i, p in enumerate(y_pred) if p == "refuted"]
    false_refuted = sum(1 for i in pred_refuted_idx if y_true[i] != "refuted")
    false_refuted_rate = false_refuted / len(pred_refuted_idx) if pred_refuted_idx else 0.0

    metrics = {
        "mode": mode,
        "tau_s": float(tau_s),
        "tau_k": float(tau_k),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "verified_f1": float(per_f1[0]),
        "refuted_f1": float(per_f1[1]),
        "unsupported_f1": float(per_f1[2]),
        "num_predicted_verified": int(len(pred_verified_idx)),
        "false_verification_rate": float(fvr),
        "num_predicted_refuted": int(len(pred_refuted_idx)),
        "false_refuted_rate": float(false_refuted_rate),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
    }

    return metrics, output

def select_best(train_rows, mode):
    tau_s_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tau_k_grid = np.round(np.arange(0.05, 0.96, 0.05), 2)

    records = []
    best = None

    if mode == "support_only":
        grid = [(ts, 1.0) for ts in tau_s_grid]
    else:
        grid = [(ts, tk) for ts in tau_s_grid for tk in tau_k_grid]

    for tau_s, tau_k in grid:
        m, _ = evaluate_examples(train_rows, tau_s, tau_k, mode)

        # Reliability-oriented objective:
        # prefer macro-F1, but penalize false verification.
        objective = m["macro_f1"] - 0.25 * m["false_verification_rate"]
        m["objective"] = float(objective)
        records.append(m)

        if best is None or m["objective"] > best["objective"]:
            best = m

    return best, records

def compact_metrics(m):
    keep = [
        "mode", "tau_s", "tau_k", "accuracy", "macro_f1",
        "verified_f1", "refuted_f1", "unsupported_f1",
        "num_predicted_verified", "false_verification_rate",
        "num_predicted_refuted", "false_refuted_rate",
        "objective"
    ]
    return {k: m.get(k) for k in keep}

def main():
    train_path = PRED_DIR / "bm25_top10_nli_train.jsonl"
    dev_path = PRED_DIR / "bm25_top10_nli_dev.jsonl"

    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("Missing NLI prediction files. Run BM25+NLI verifier first.")

    train_rows = read_jsonl(train_path)
    dev_rows = read_jsonl(dev_path)

    print(f"Loaded train rows: {len(train_rows)}")
    print(f"Loaded dev rows:   {len(dev_rows)}")

    modes = ["support_only", "support_contradiction", "ours_no_abstain"]

    all_summary = []
    all_grid_records = []

    for mode in modes:
        print("\n" + "=" * 80)
        print(f"Calibrating mode: {mode}")

        best_train, grid_records = select_best(train_rows, mode)
        all_grid_records.extend(grid_records)

        print("Best train:")
        print(json.dumps(compact_metrics(best_train), indent=2))

        dev_metrics, dev_output = evaluate_examples(
            dev_rows,
            best_train["tau_s"],
            best_train["tau_k"],
            mode
        )

        dev_metrics["selected_from_train_objective"] = best_train["objective"]
        dev_metrics["selected_train_macro_f1"] = best_train["macro_f1"]
        dev_metrics["selected_train_false_verification_rate"] = best_train["false_verification_rate"]

        print("Dev result with train-selected thresholds:")
        print(json.dumps(compact_metrics(dev_metrics), indent=2))
        print("Dev confusion matrix labels:", LABELS)
        print(np.array(dev_metrics["confusion_matrix"]))

        pred_path = PRED_DIR / f"bm25_top10_nli_calibrated_{mode}_dev.jsonl"
        write_jsonl(pred_path, dev_output)

        metric_path = METRIC_DIR / f"bm25_top10_nli_calibrated_{mode}_dev_metrics.json"
        with metric_path.open("w", encoding="utf-8") as f:
            json.dump(dev_metrics, f, indent=2, ensure_ascii=False)

        all_summary.append({
            "mode": mode,
            "selected_tau_s": best_train["tau_s"],
            "selected_tau_k": best_train["tau_k"],
            "train_objective": best_train["objective"],
            "train_macro_f1": best_train["macro_f1"],
            "train_false_verification_rate": best_train["false_verification_rate"],
            "dev_accuracy": dev_metrics["accuracy"],
            "dev_macro_f1": dev_metrics["macro_f1"],
            "dev_verified_f1": dev_metrics["verified_f1"],
            "dev_refuted_f1": dev_metrics["refuted_f1"],
            "dev_unsupported_f1": dev_metrics["unsupported_f1"],
            "dev_num_predicted_verified": dev_metrics["num_predicted_verified"],
            "dev_false_verification_rate": dev_metrics["false_verification_rate"],
            "dev_num_predicted_refuted": dev_metrics["num_predicted_refuted"],
            "dev_false_refuted_rate": dev_metrics["false_refuted_rate"],
        })

    summary_df = pd.DataFrame(all_summary)
    summary_path = METRIC_DIR / "bm25_top10_nli_threshold_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_df = pd.DataFrame([
        {k: v for k, v in r.items() if not isinstance(v, (dict, list))}
        for r in all_grid_records
    ])
    grid_path = METRIC_DIR / "bm25_top10_nli_threshold_grid_train.csv"
    grid_df.to_csv(grid_path, index=False)

    print("\n" + "=" * 80)
    print("Final calibration summary:")
    print(summary_df.to_string(index=False))

    print(f"\nSaved summary: {summary_path}")
    print(f"Saved grid:    {grid_path}")

if __name__ == "__main__":
    main()
