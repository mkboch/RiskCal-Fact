import os
import json
import gc
import random
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = Path("data/processed/evidence_given")
PRED_DIR = Path("outputs/predictions/evidence_given")
METRIC_DIR = Path("outputs/metrics/evidence_given")
TABLE_DIR = Path("outputs/tables/evidence_given")
FINAL_DIR = Path("outputs/final_report")
for d in [PRED_DIR, METRIC_DIR, TABLE_DIR, FINAL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

LABELS = ["verified", "refuted", "unsupported"]
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
MAX_EVIDENCE_UNITS = 5
BATCH_SIZE = 32
MAX_LENGTH = 256

MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "facebook/bart-large-mnli",
]

DATASETS = {
    "vitaminc": {
        "tune": "vitaminc_train.jsonl",
        "cal": "vitaminc_validation.jsonl",
        "dev": "vitaminc_test.jsonl",
        "tune_per_label": 5000,
        "cal_per_label": 2500,
        "dev_per_label": None,
    },
    "pubhealth": {
        "tune": "pubhealth_train.jsonl",
        "cal": "pubhealth_validation.jsonl",
        "dev": "pubhealth_test.jsonl",
        "tune_per_label": None,
        "cal_per_label": None,
        "dev_per_label": None,
    },
    "climate_fever": {
        "tune": "climate_fever_tune.jsonl",
        "cal": "climate_fever_cal.jsonl",
        "dev": "climate_fever_dev.jsonl",
        "tune_per_label": None,
        "cal_per_label": None,
        "dev_per_label": None,
    },
}

def read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stratified_cap(rows, per_label):
    if per_label is None:
        return rows
    rng = random.Random(SEED)
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)
    out = []
    for lab in LABELS:
        items = by.get(lab, [])
        rng.shuffle(items)
        out.extend(items[:min(per_label, len(items))])
    rng.shuffle(out)
    return out

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

def score_dataset(model_name, dataset_name, split_name, rows, device):
    safe_model = model_name.replace("/", "__")
    out_path = PRED_DIR / f"{dataset_name}_{split_name}_{safe_model}_scores.jsonl"
    if out_path.exists():
        print("Loading cached scores:", out_path)
        return read_jsonl(out_path)

    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    mapping = infer_mapping(model)
    print("Mapping:", mapping)

    claims, premises, refs = [], [], []
    for i, r in enumerate(rows):
        units = r.get("evidence_units", [])[:MAX_EVIDENCE_UNITS]
        for j, ev in enumerate(units):
            claims.append(r["claim"])
            premises.append(str(ev))
            refs.append((i, j))

    ent, con, neu = [], [], []

    with torch.no_grad():
        for start in tqdm(range(0, len(claims), BATCH_SIZE), desc=f"NLI {dataset_name}/{split_name}/{model_name}"):
            bc = claims[start:start+BATCH_SIZE]
            bp = premises[start:start+BATCH_SIZE]
            enc = tok(bp, bc, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
            enc = {k: v.to(device) for k, v in enc.items()}
            probs = torch.softmax(model(**enc).logits, dim=-1).detach().cpu().numpy()
            for row in probs:
                ent.append(float(row[mapping["entailment"]]))
                con.append(float(row[mapping["contradiction"]]))
                neu.append(float(row[mapping["neutral"]]))

    full = []
    score_by_i = defaultdict(list)
    for (i, j), e, c, n in zip(refs, ent, con, neu):
        score_by_i[i].append({"unit_index": j, "entailment": e, "contradiction": c, "neutral": n})

    for i, r in enumerate(rows):
        scores = score_by_i.get(i, [])
        S = max([x["entailment"] for x in scores], default=0.0)
        K = max([x["contradiction"] for x in scores], default=0.0)
        N = max([x["neutral"] for x in scores], default=1.0)
        rr = {
            "id": r["id"],
            "dataset": dataset_name,
            "split": split_name,
            "claim": r["claim"],
            "label": r["label"],
            "model": model_name,
            "S": float(S),
            "K": float(K),
            "N": float(N),
            "P": 1 if len(scores) > 0 else 0,
            "num_evidence_units": len(r.get("evidence_units", [])[:MAX_EVIDENCE_UNITS]),
            "unit_scores": scores,
        }
        full.append(rr)

    write_jsonl(out_path, full)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return full

def predict(rule, r, params):
    S, K, P = float(r["S"]), float(r["K"]), int(r["P"])
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
    return float(r["S"]) if rule == "support_only" else float(r["S"]) - float(r["K"])

def eval_preds(rows, preds):
    y_true = [r["label"] for r in rows]
    y_pred = preds

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
        "accuracy_all": float(acc),
        "macro_f1_all": float(macro),
        "verified_f1": float(per[0]),
        "refuted_f1": float(per[1]),
        "unsupported_f1": float(per[2]),
        "num_predicted_verified": len(pred_verified),
        "false_verification_rate": float(fvr),
        "num_predicted_refuted": len(pred_refuted),
        "false_refuted_rate": float(frr),
        "num_abstained": len(abstained),
        "coverage": float(len(accepted) / len(rows)),
        "accepted_accuracy": float(accepted_acc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS + ["abstain"]).tolist(),
        "confusion_matrix_labels": LABELS + ["abstain"],
    }

def tune_params(rows, rule):
    vals = np.round(np.arange(0.05, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.0, 0.81, 0.05), 2)

    if rule == "support_only":
        grid = [{"tau_s": ts} for ts in vals]
    else:
        grid = [{"tau_s": ts, "tau_k": tk, "margin": mg} for ts in vals for tk in vals for mg in margins]

    best = None
    records = []

    for params in grid:
        preds = [predict(rule, r, params) for r in rows]
        m = eval_preds(rows, preds)
        obj = m["macro_f1_all"] + 0.1 * m["accepted_accuracy"] - 0.5 * m["false_verification_rate"] - 0.2 * m["false_refuted_rate"]
        m["objective"] = float(obj)
        m.update({f"param_{k}": float(v) for k, v in params.items()})
        records.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})

        if best is None or obj > best["objective"]:
            best = m

    params = {k.replace("param_", ""): float(v) for k, v in best.items() if k.startswith("param_")}
    return params, best, records

def choose_threshold(cal_rows, cal_preds, cal_confs, alpha):
    idxs = [i for i, p in enumerate(cal_preds) if p == "verified"]

    if not idxs:
        return float("inf"), {
            "cal_base_verified": 0,
            "cal_retained_verified": 0,
            "cal_retained_fvr": 0.0,
            "cal_verified_retention": 0.0,
        }

    thresholds = sorted(set(cal_confs[i] for i in idxs))
    for t in thresholds:
        retained = [i for i in idxs if cal_confs[i] >= t]
        if not retained:
            continue
        fvr = sum(1 for i in retained if cal_rows[i]["label"] != "verified") / len(retained)
        if fvr <= alpha:
            return float(t), {
                "cal_base_verified": len(idxs),
                "cal_retained_verified": len(retained),
                "cal_retained_fvr": float(fvr),
                "cal_verified_retention": float(len(retained) / len(idxs)),
            }

    return float(max(thresholds) + 1e-6), {
        "cal_base_verified": len(idxs),
        "cal_retained_verified": 0,
        "cal_retained_fvr": 0.0,
        "cal_verified_retention": 0.0,
    }

def apply_gate(preds, confs, threshold):
    return ["abstain" if p == "verified" and c < threshold else p for p, c in zip(preds, confs)]

def run_calibration(dataset_name, model_name, tune_rows, cal_rows, dev_rows):
    summary = []
    grid_records = []

    for rule in ["support_only", "margin_support_refute"]:
        print("Calibrating:", dataset_name, model_name, rule)

        params, best, grid = tune_params(tune_rows, rule)
        for g in grid:
            g["dataset"] = dataset_name
            g["model"] = model_name
            g["rule"] = rule
        grid_records.extend(grid)

        cal_preds = [predict(rule, r, params) for r in cal_rows]
        dev_preds = [predict(rule, r, params) for r in dev_rows]
        cal_confs = [verified_conf(rule, r) for r in cal_rows]
        dev_confs = [verified_conf(rule, r) for r in dev_rows]

        base_dev = eval_preds(dev_rows, dev_preds)

        for alpha in ALPHAS:
            thr, cal_info = choose_threshold(cal_rows, cal_preds, cal_confs, alpha)
            risk_preds = apply_gate(dev_preds, dev_confs, thr)
            m = eval_preds(dev_rows, risk_preds)

            row = {
                "dataset": dataset_name,
                "model": model_name,
                "rule": rule,
                "alpha": float(alpha),
                "threshold": float(thr),
                **{f"param_{k}": float(v) for k, v in params.items()},
                **cal_info,
                "base_dev_macro_f1": base_dev["macro_f1_all"],
                "base_dev_false_verification_rate": base_dev["false_verification_rate"],
                **m,
            }
            summary.append(row)

    return summary, grid_records

def main():
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    all_summary = []
    all_grids = []
    dataset_status = {}

    for dataset_name, spec in DATASETS.items():
        print("\n" + "=" * 100)
        print("DATASET:", dataset_name)

        tune = read_jsonl(DATA_DIR / spec["tune"])
        cal = read_jsonl(DATA_DIR / spec["cal"])
        dev = read_jsonl(DATA_DIR / spec["dev"])

        tune = stratified_cap(tune, spec["tune_per_label"])
        cal = stratified_cap(cal, spec["cal_per_label"])
        dev = stratified_cap(dev, spec["dev_per_label"])

        dataset_status[dataset_name] = {
            "tune_rows": len(tune),
            "cal_rows": len(cal),
            "dev_rows": len(dev),
            "tune_labels": dict(Counter(r["label"] for r in tune)),
            "cal_labels": dict(Counter(r["label"] for r in cal)),
            "dev_labels": dict(Counter(r["label"] for r in dev)),
        }
        print(json.dumps(dataset_status[dataset_name], indent=2))

        for model_name in MODELS:
            tune_scores = score_dataset(model_name, dataset_name, "tune", tune, device)
            cal_scores = score_dataset(model_name, dataset_name, "cal", cal, device)
            dev_scores = score_dataset(model_name, dataset_name, "dev", dev, device)

            summary, grids = run_calibration(dataset_name, model_name, tune_scores, cal_scores, dev_scores)
            all_summary.extend(summary)
            all_grids.extend(grids)

    summary_df = pd.DataFrame([{k: v for k, v in r.items() if not isinstance(v, (dict, list))} for r in all_summary])
    summary_path = METRIC_DIR / "evidence_given_risk_calibration_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_path = METRIC_DIR / "evidence_given_tune_grid.csv"
    pd.DataFrame(all_grids).to_csv(grid_path, index=False)

    keep = [
        "dataset", "model", "rule", "alpha", "macro_f1_all", "accuracy_all",
        "false_verification_rate", "false_refuted_rate",
        "num_predicted_verified", "num_predicted_refuted",
        "num_abstained", "coverage", "accepted_accuracy",
        "cal_retained_fvr", "cal_verified_retention",
        "base_dev_macro_f1", "base_dev_false_verification_rate",
    ]
    table = summary_df[[c for c in keep if c in summary_df.columns]].copy()
    table_path = TABLE_DIR / "table_evidence_given_risk_calibration.csv"
    table.to_csv(table_path, index=False)

    status_path = FINAL_DIR / "evidence_given_status.json"
    status_path.write_text(json.dumps(dataset_status, indent=2), encoding="utf-8")

    # Best table.
    best_rows = []
    for dataset_name in sorted(summary_df["dataset"].unique()):
        sub = summary_df[summary_df["dataset"] == dataset_name].copy()

        best_macro = sub.sort_values("macro_f1_all", ascending=False).head(1)
        if len(best_macro):
            r = best_macro.iloc[0]
            best_rows.append({
                "dataset": dataset_name,
                "selection": "best_macro_f1",
                "model": r["model"],
                "rule": r["rule"],
                "alpha": r["alpha"],
                "macro_f1": r["macro_f1_all"],
                "fvr": r["false_verification_rate"],
                "coverage": r["coverage"],
                "accepted_accuracy": r["accepted_accuracy"],
            })

        low = sub[sub["coverage"] >= 0.75].sort_values(["false_verification_rate", "macro_f1_all"], ascending=[True, False]).head(1)
        if len(low):
            r = low.iloc[0]
            best_rows.append({
                "dataset": dataset_name,
                "selection": "lowest_fvr_coverage_ge_0.75",
                "model": r["model"],
                "rule": r["rule"],
                "alpha": r["alpha"],
                "macro_f1": r["macro_f1_all"],
                "fvr": r["false_verification_rate"],
                "coverage": r["coverage"],
                "accepted_accuracy": r["accepted_accuracy"],
            })

    best_df = pd.DataFrame(best_rows)
    best_path = TABLE_DIR / "table_evidence_given_best_results.csv"
    best_df.to_csv(best_path, index=False)

    report_path = FINAL_DIR / "evidence_given_experiment_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Evidence-Given Verification Experiments\n\n")
        f.write("Datasets: VitaminC, PubHealth, and Climate-FEVER.\n\n")
        f.write("## Dataset Status\n\n")
        f.write(pd.DataFrame(dataset_status).T.to_markdown())
        f.write("\n\n")
        f.write("## Best Results\n\n")
        f.write(best_df.to_markdown(index=False))
        f.write("\n\n")
        f.write("## Full Risk Calibration Table\n\n")
        f.write(table.to_markdown(index=False))
        f.write("\n")

    print("\n==== EVIDENCE-GIVEN BEST RESULTS ====")
    print(best_df.to_string(index=False))
    print("\nSaved:")
    for p in [summary_path, grid_path, table_path, best_path, report_path, status_path]:
        print(f"  {p} ({p.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
