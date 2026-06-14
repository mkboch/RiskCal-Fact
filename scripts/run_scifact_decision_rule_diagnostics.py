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

def enrich_scores(rows):
    out = []
    for r in rows:
        cands = r.get("sentence_candidates", [])
        if not cands:
            rr = dict(r)
            rr.update({
                "S": 0.0,
                "K": 0.0,
                "N": 1.0,
                "M_SK": 0.0,
                "M_KS": 0.0,
                "best_label_score": 0.0,
                "best_relation": "none",
                "num_candidates": 0,
            })
            out.append(rr)
            continue

        S = max(float(c.get("nli_entailment", 0.0)) for c in cands)
        K = max(float(c.get("nli_contradiction", 0.0)) for c in cands)
        N = max(float(c.get("nli_neutral", 0.0)) for c in cands)

        # Best support and contradiction candidates.
        best_s = max(cands, key=lambda c: float(c.get("nli_entailment", 0.0)))
        best_k = max(cands, key=lambda c: float(c.get("nli_contradiction", 0.0)))

        # Margins.
        M_SK = S - K
        M_KS = K - S

        # Relation winner from global max of support/contradiction/neutral.
        triples = [("verified", S), ("refuted", K), ("unsupported", N)]
        best_relation, best_label_score = max(triples, key=lambda x: x[1])

        rr = dict(r)
        rr.update({
            "S": float(S),
            "K": float(K),
            "N": float(N),
            "M_SK": float(M_SK),
            "M_KS": float(M_KS),
            "best_label_score": float(best_label_score),
            "best_relation": best_relation,
            "best_support_diag": best_s,
            "best_contradiction_diag": best_k,
            "num_candidates": len(cands),
        })
        out.append(rr)
    return out

def predict_rule(r, rule, params):
    S = float(r["S"])
    K = float(r["K"])
    N = float(r["N"])
    P = int(r.get("P_provenance", 1))
    margin = float(params.get("margin", 0.0))
    tau_s = float(params.get("tau_s", 0.5))
    tau_k = float(params.get("tau_k", 0.5))
    tau_n = float(params.get("tau_n", 0.5))
    tau_accept = float(params.get("tau_accept", 0.5))

    if rule == "support_only":
        if S >= tau_s and P == 1:
            return "verified"
        return "unsupported"

    if rule == "winner_takes_all":
        scores = {"verified": S, "refuted": K, "unsupported": N}
        return max(scores, key=scores.get)

    if rule == "margin_support_refute":
        # Refute only if contradiction is high and beats support by margin.
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        if S >= tau_s and (S - K) >= margin and P == 1:
            return "verified"
        return "unsupported"

    if rule == "support_first_margin":
        # Conservative for contradiction. Prioritize support when support is high.
        if S >= tau_s and (S - K) >= margin and P == 1:
            return "verified"
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        return "unsupported"

    if rule == "risk_abstain_margin":
        # If top relation is not confident or margin is small, abstain.
        scores = {"verified": S, "refuted": K, "unsupported": N}
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = ordered[0]
        second_score = ordered[1][1]

        if top_score < tau_accept:
            return "abstain"
        if (top_score - second_score) < margin:
            return "abstain"
        return top_label

    raise ValueError(f"Unknown rule: {rule}")

def evaluate(rows, rule, params, include_abstain=False):
    y_true_all = []
    y_pred_all = []
    out = []

    for r in rows:
        pred = predict_rule(r, rule, params)
        rr = dict(r)
        rr["prediction"] = pred
        rr["decision_rule"] = rule
        rr["decision_params"] = params
        out.append(rr)
        y_true_all.append(r["label"])
        y_pred_all.append(pred)

    # Standard metrics treating abstain as wrong/not a class.
    eval_labels = LABELS + (["abstain"] if include_abstain else [])
    acc = accuracy_score(y_true_all, y_pred_all)
    macro = f1_score(y_true_all, y_pred_all, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true_all, y_pred_all, labels=LABELS, average=None, zero_division=0)

    pred_verified = [i for i, p in enumerate(y_pred_all) if p == "verified"]
    fvr = (
        sum(1 for i in pred_verified if y_true_all[i] != "verified") / len(pred_verified)
        if pred_verified else 0.0
    )

    pred_refuted = [i for i, p in enumerate(y_pred_all) if p == "refuted"]
    frr = (
        sum(1 for i in pred_refuted if y_true_all[i] != "refuted") / len(pred_refuted)
        if pred_refuted else 0.0
    )

    abstained = [i for i, p in enumerate(y_pred_all) if p == "abstain"]
    accepted = [i for i, p in enumerate(y_pred_all) if p != "abstain"]

    if accepted:
        accepted_acc = accuracy_score(
            [y_true_all[i] for i in accepted],
            [y_pred_all[i] for i in accepted],
        )
    else:
        accepted_acc = 0.0

    metrics = {
        "rule": rule,
        **{f"param_{k}": float(v) for k, v in params.items()},
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
        "confusion_matrix": confusion_matrix(y_true_all, y_pred_all, labels=LABELS + ["abstain"]).tolist(),
        "classification_report": classification_report(
            y_true_all,
            y_pred_all,
            labels=LABELS + ["abstain"],
            output_dict=True,
            zero_division=0,
        ),
    }
    return metrics, out

def objective(m):
    # Conservative objective aligned with our paper:
    # reward macro-F1 and accepted accuracy, penalize false verification and false refutation.
    return (
        m["macro_f1_all"]
        + 0.10 * m["accepted_accuracy"]
        - 0.50 * m["false_verification_rate"]
        - 0.20 * m["false_refuted_rate"]
        - 0.05 * (1.0 - m["coverage"])
    )

def grid_for_rule(rule):
    vals = np.round(np.arange(0.05, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.00, 0.81, 0.05), 2)

    if rule == "support_only":
        return [{"tau_s": ts} for ts in vals]

    if rule == "winner_takes_all":
        return [{}]

    if rule in ["margin_support_refute", "support_first_margin"]:
        return [
            {"tau_s": ts, "tau_k": tk, "margin": mg}
            for ts in vals
            for tk in vals
            for mg in margins
        ]

    if rule == "risk_abstain_margin":
        return [
            {"tau_accept": ta, "margin": mg}
            for ta in vals
            for mg in margins
        ]

    raise ValueError(rule)

def calibrate(train_rows, rule):
    best = None
    records = []

    for params in grid_for_rule(rule):
        m, _ = evaluate(train_rows, rule, params)
        obj = objective(m)
        m["objective"] = float(obj)
        records.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})

        if best is None or obj > best["objective"]:
            best = m

    return best, records

def compact(m):
    keys = [
        "rule", "objective", "accuracy_all", "macro_f1_all",
        "verified_f1", "refuted_f1", "unsupported_f1",
        "num_predicted_verified", "false_verification_rate",
        "num_predicted_refuted", "false_refuted_rate",
        "num_abstained", "coverage", "accepted_accuracy",
        "param_tau_s", "param_tau_k", "param_margin", "param_tau_accept",
    ]
    return {k: m.get(k) for k in keys if k in m}

def score_distribution(rows, split):
    recs = []
    for label in LABELS:
        subset = [r for r in rows if r["label"] == label]
        if not subset:
            continue
        for score in ["S", "K", "N", "M_SK", "M_KS"]:
            vals = np.array([float(r[score]) for r in subset])
            recs.append({
                "split": split,
                "label": label,
                "score": score,
                "n": len(vals),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "p10": float(np.quantile(vals, 0.10)),
                "p25": float(np.quantile(vals, 0.25)),
                "p75": float(np.quantile(vals, 0.75)),
                "p90": float(np.quantile(vals, 0.90)),
            })
    return recs

def main():
    train_path = PRED_DIR / "bm25_top10_sentence_nli_scores_train.jsonl"
    dev_path = PRED_DIR / "bm25_top10_sentence_nli_scores_dev.jsonl"

    train = enrich_scores(read_jsonl(train_path))
    dev = enrich_scores(read_jsonl(dev_path))

    print(f"Loaded train: {len(train)}")
    print(f"Loaded dev:   {len(dev)}")

    dist = score_distribution(train, "train") + score_distribution(dev, "dev")
    dist_path = METRIC_DIR / "sentence_nli_score_distributions.csv"
    pd.DataFrame(dist).to_csv(dist_path, index=False)
    print(f"Saved score distributions: {dist_path}")

    rules = [
        "support_only",
        "winner_takes_all",
        "margin_support_refute",
        "support_first_margin",
        "risk_abstain_margin",
    ]

    summary = []
    all_grid = []

    for rule in rules:
        print("\n" + "=" * 80)
        print("Calibrating rule:", rule)
        best, grid = calibrate(train, rule)
        all_grid.extend(grid)

        print("Best train:")
        print(json.dumps(compact(best), indent=2))

        params = {}
        for k, v in best.items():
            if k.startswith("param_"):
                params[k.replace("param_", "")] = v

        dev_metrics, dev_out = evaluate(dev, rule, params)
        dev_metrics["selected_train_objective"] = best["objective"]
        dev_metrics["selected_train_macro_f1_all"] = best["macro_f1_all"]
        dev_metrics["selected_train_false_verification_rate"] = best["false_verification_rate"]
        dev_metrics["selected_train_false_refuted_rate"] = best["false_refuted_rate"]

        print("Dev:")
        print(json.dumps(compact(dev_metrics), indent=2))
        print("Dev confusion matrix labels:", dev_metrics["confusion_matrix_labels"])
        print(np.array(dev_metrics["confusion_matrix"]))

        pred_path = PRED_DIR / f"sentence_nli_decision_{rule}_dev.jsonl"
        metric_path = METRIC_DIR / f"sentence_nli_decision_{rule}_dev_metrics.json"
        write_jsonl(pred_path, dev_out)
        with metric_path.open("w", encoding="utf-8") as f:
            json.dump(dev_metrics, f, indent=2, ensure_ascii=False)

        row = compact(dev_metrics)
        row.update({
            "selected_train_objective": best["objective"],
            "selected_train_macro_f1_all": best["macro_f1_all"],
            "selected_train_false_verification_rate": best["false_verification_rate"],
            "selected_train_false_refuted_rate": best["false_refuted_rate"],
        })
        summary.append(row)

    summary_df = pd.DataFrame(summary)
    summary_path = METRIC_DIR / "sentence_nli_decision_rule_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_path = METRIC_DIR / "sentence_nli_decision_rule_train_grid.csv"
    pd.DataFrame(all_grid).to_csv(grid_path, index=False)

    print("\n" + "=" * 80)
    print("Final decision-rule summary:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved grid:    {grid_path}")

if __name__ == "__main__":
    main()
