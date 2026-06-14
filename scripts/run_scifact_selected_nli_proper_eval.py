import json
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

PROC_DIR = Path("data/processed/scifact")
PRED_DIR = Path("outputs/predictions/scifact")
METRIC_DIR = Path("outputs/metrics/scifact")
PRED_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]

SELECTED_MODELS = [
    "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli",
    "facebook/bart-large-mnli",
]

TOP_DOCS = 10
MAX_SENTENCES_PER_DOC = 20
MAX_LENGTH = 256
BATCH_SIZE = 32

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

def load_corpus_map():
    corpus = read_jsonl(PROC_DIR / "corpus.jsonl")
    return {str(x["doc_id"]): x for x in corpus}

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
        # Common MNLI fallback.
        mapping = {"contradiction": 0, "neutral": 1, "entailment": 2}

    if "entailment" not in mapping or "contradiction" not in mapping:
        raise RuntimeError(f"Cannot infer mapping from id2label={id2label}")

    return mapping

def build_pairs(split, corpus_map):
    retrieval_path = PRED_DIR / f"bm25_top10_{split}.jsonl"
    examples = read_jsonl(retrieval_path)

    pair_claims = []
    pair_premises = []
    pair_refs = []

    for ex_idx, ex in enumerate(examples):
        claim = ex["claim"]
        candidates = []

        for doc_rank, doc in enumerate(ex["retrieved"][:TOP_DOCS], start=1):
            doc_id = str(doc["doc_id"])
            cdoc = corpus_map.get(doc_id)
            if not cdoc:
                continue

            title = cdoc.get("title", "")
            sentences = cdoc.get("sentences", [])
            if not isinstance(sentences, list):
                sentences = [str(sentences)]

            for sent_idx, sent in enumerate(sentences[:MAX_SENTENCES_PER_DOC]):
                sent = str(sent).strip()
                if not sent:
                    continue

                premise = (title + ". " + sent).strip()
                cand = {
                    "doc_rank": doc_rank,
                    "doc_id": doc_id,
                    "sentence_index": sent_idx,
                    "sentence": sent,
                    "bm25_doc_score": float(doc.get("score", 0.0)),
                }

                cand_idx = len(candidates)
                candidates.append(cand)
                pair_claims.append(claim)
                pair_premises.append(premise)
                pair_refs.append((ex_idx, cand_idx))

        ex["sentence_candidates"] = candidates

    return examples, pair_claims, pair_premises, pair_refs

def score_split(model_name, split, tokenizer, model, device, corpus_map):
    out_path = PRED_DIR / f"selected_nli_{safe_name(model_name)}_{split}_scores.jsonl"
    if out_path.exists():
        print(f"Loading cached scores: {out_path}")
        return read_jsonl(out_path)

    examples, pair_claims, pair_premises, pair_refs = build_pairs(split, corpus_map)
    print(f"{split}: examples={len(examples)}, sentence pairs={len(pair_claims)}")

    mapping = infer_mapping(model)
    print("id2label:", model.config.id2label)
    print("mapping:", mapping)

    ent, con, neu = [], [], []

    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, len(pair_claims), BATCH_SIZE), desc=f"{split} NLI {model_name}"):
            batch_claims = pair_claims[start:start+BATCH_SIZE]
            batch_premises = pair_premises[start:start+BATCH_SIZE]

            enc = tokenizer(
                batch_premises,
                batch_claims,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}

            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

            for row in probs:
                ent.append(float(row[mapping["entailment"]]))
                con.append(float(row[mapping["contradiction"]]))
                if "neutral" in mapping:
                    neu.append(float(row[mapping["neutral"]]))
                else:
                    neu.append(float(max(0.0, 1.0 - row[mapping["entailment"]] - row[mapping["contradiction"]])))

    for (ex_idx, cand_idx), e, c, n in zip(pair_refs, ent, con, neu):
        examples[ex_idx]["sentence_candidates"][cand_idx]["entailment"] = e
        examples[ex_idx]["sentence_candidates"][cand_idx]["contradiction"] = c
        examples[ex_idx]["sentence_candidates"][cand_idx]["neutral"] = n

    rows = []
    for ex in examples:
        cands = ex["sentence_candidates"]

        if cands:
            S = max(float(c["entailment"]) for c in cands)
            K = max(float(c["contradiction"]) for c in cands)
            N = max(float(c["neutral"]) for c in cands)
            best_support = max(cands, key=lambda c: float(c["entailment"]))
            best_contradiction = max(cands, key=lambda c: float(c["contradiction"]))
            best_neutral = max(cands, key=lambda c: float(c["neutral"]))
        else:
            S, K, N = 0.0, 0.0, 1.0
            best_support = None
            best_contradiction = None
            best_neutral = None

        rows.append({
            "id": ex["id"],
            "dataset": "scifact",
            "split": split,
            "claim": ex["claim"],
            "label": ex["label"],
            "model": model_name,
            "S": float(S),
            "K": float(K),
            "N": float(N),
            "M_SK": float(S - K),
            "M_KS": float(K - S),
            "P": 1 if cands else 0,
            "best_support": best_support,
            "best_contradiction": best_contradiction,
            "best_neutral": best_neutral,
        })

    write_jsonl(out_path, rows)
    print(f"Saved scores: {out_path}")
    return rows

def pred(rule, r, params):
    S = float(r["S"])
    K = float(r["K"])
    N = float(r["N"])
    P = int(r["P"])

    tau_s = float(params.get("tau_s", 0.5))
    tau_k = float(params.get("tau_k", 0.5))
    margin = float(params.get("margin", 0.0))
    tau_accept = float(params.get("tau_accept", 0.0))

    if rule == "support_only":
        return "verified" if S >= tau_s and P == 1 else "unsupported"

    if rule == "winner_takes_all":
        return max({"verified": S, "refuted": K, "unsupported": N}, key={"verified": S, "refuted": K, "unsupported": N}.get)

    if rule == "margin_support_refute":
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        if S >= tau_s and (S - K) >= margin and P == 1:
            return "verified"
        return "unsupported"

    if rule == "verified_safe":
        # High-precision verified rule. It verifies only when support is high and contradiction does not dominate.
        if S >= tau_s and (S - K) >= margin and P == 1:
            return "verified"
        if K >= tau_k and (K - S) >= margin:
            return "refuted"
        return "unsupported"

    if rule == "risk_abstain":
        scores = {"verified": S, "refuted": K, "unsupported": N}
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_label, top_score = ordered[0]
        second_score = ordered[1][1]

        if top_score < tau_accept:
            return "abstain"
        if (top_score - second_score) < margin:
            return "abstain"
        return top_label

    raise ValueError(rule)

def eval_rows(rows, rule, params):
    y_true = []
    y_pred = []
    out = []

    for r in rows:
        p = pred(rule, r, params)
        rr = dict(r)
        rr["prediction"] = p
        rr["rule"] = rule
        rr["params"] = params
        out.append(rr)
        y_true.append(r["label"])
        y_pred.append(p)

    acc_all = accuracy_score(y_true, y_pred)
    macro_all = f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)
    per = f1_score(y_true, y_pred, labels=LABELS, average=None, zero_division=0)

    pred_verified = [i for i, p in enumerate(y_pred) if p == "verified"]
    fvr = sum(1 for i in pred_verified if y_true[i] != "verified") / len(pred_verified) if pred_verified else 0.0

    pred_refuted = [i for i, p in enumerate(y_pred) if p == "refuted"]
    frr = sum(1 for i in pred_refuted if y_true[i] != "refuted") / len(pred_refuted) if pred_refuted else 0.0

    abstained = [i for i, p in enumerate(y_pred) if p == "abstain"]
    accepted = [i for i, p in enumerate(y_pred) if p != "abstain"]

    accepted_acc = (
        accuracy_score([y_true[i] for i in accepted], [y_pred[i] for i in accepted])
        if accepted else 0.0
    )

    metrics = {
        "rule": rule,
        **{f"param_{k}": float(v) for k, v in params.items()},
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
    return metrics, out

def objective(m):
    return (
        m["macro_f1_all"]
        + 0.10 * m["accepted_accuracy"]
        - 0.50 * m["false_verification_rate"]
        - 0.20 * m["false_refuted_rate"]
        - 0.05 * (1.0 - m["coverage"])
    )

def grid(rule):
    vals = np.round(np.arange(0.05, 0.96, 0.05), 2)
    margins = np.round(np.arange(0.0, 0.81, 0.05), 2)

    if rule == "support_only":
        return [{"tau_s": ts} for ts in vals]

    if rule == "winner_takes_all":
        return [{}]

    if rule in ["margin_support_refute", "verified_safe"]:
        return [
            {"tau_s": ts, "tau_k": tk, "margin": mg}
            for ts in vals for tk in vals for mg in margins
        ]

    if rule == "risk_abstain":
        return [
            {"tau_accept": ta, "margin": mg}
            for ta in vals for mg in margins
        ]

    raise ValueError(rule)

def calibrate(train_rows, rule):
    best = None
    records = []

    for params in grid(rule):
        m, _ = eval_rows(train_rows, rule, params)
        obj = objective(m)
        m["objective"] = float(obj)
        records.append({k: v for k, v in m.items() if not isinstance(v, (dict, list))})

        if best is None or obj > best["objective"]:
            best = m

    best_params = {}
    for k, v in best.items():
        if k.startswith("param_"):
            best_params[k.replace("param_", "")] = v

    return best, best_params, records

def compact(m):
    keys = [
        "model", "rule", "objective", "accuracy_all", "macro_f1_all",
        "verified_f1", "refuted_f1", "unsupported_f1",
        "num_predicted_verified", "false_verification_rate",
        "num_predicted_refuted", "false_refuted_rate",
        "num_abstained", "coverage", "accepted_accuracy",
        "param_tau_s", "param_tau_k", "param_margin", "param_tau_accept",
    ]
    return {k: m.get(k) for k in keys if k in m}

def main():
    print("CUDA_VISIBLE_DEVICES:", __import__("os").environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("cuda devices:", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    corpus_map = load_corpus_map()

    rules = [
        "support_only",
        "winner_takes_all",
        "margin_support_refute",
        "verified_safe",
        "risk_abstain",
    ]

    all_summary = []
    all_grid = []

    for model_name in SELECTED_MODELS:
        print("\n" + "#" * 100)
        print("MODEL:", model_name)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)

        train_rows = score_split(model_name, "train", tokenizer, model, device, corpus_map)
        dev_rows = score_split(model_name, "dev", tokenizer, model, device, corpus_map)

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        for rule in rules:
            print("\n" + "=" * 80)
            print(f"Calibrating rule={rule} on train for model={model_name}")

            best_train, params, grid_records = calibrate(train_rows, rule)
            for gr in grid_records:
                gr["model"] = model_name
                all_grid.append(gr)

            dev_metrics, dev_out = eval_rows(dev_rows, rule, params)
            dev_metrics["model"] = model_name
            dev_metrics["selected_train_objective"] = best_train["objective"]
            dev_metrics["selected_train_macro_f1_all"] = best_train["macro_f1_all"]
            dev_metrics["selected_train_false_verification_rate"] = best_train["false_verification_rate"]
            dev_metrics["selected_train_false_refuted_rate"] = best_train["false_refuted_rate"]

            print("Best train:")
            print(json.dumps(compact({**best_train, "model": model_name}), indent=2))
            print("Dev result:")
            print(json.dumps(compact(dev_metrics), indent=2))
            print("Dev confusion matrix labels:", dev_metrics["confusion_matrix_labels"])
            print(np.array(dev_metrics["confusion_matrix"]))

            pred_path = PRED_DIR / f"proper_eval_{safe_name(model_name)}_{rule}_dev.jsonl"
            metric_path = METRIC_DIR / f"proper_eval_{safe_name(model_name)}_{rule}_dev_metrics.json"
            write_jsonl(pred_path, dev_out)
            with metric_path.open("w", encoding="utf-8") as f:
                json.dump(dev_metrics, f, indent=2, ensure_ascii=False)

            row = compact(dev_metrics)
            row.update({
                "selected_train_objective": best_train["objective"],
                "selected_train_macro_f1_all": best_train["macro_f1_all"],
                "selected_train_false_verification_rate": best_train["false_verification_rate"],
                "selected_train_false_refuted_rate": best_train["false_refuted_rate"],
            })
            all_summary.append(row)

    summary_df = pd.DataFrame(all_summary)
    summary_path = METRIC_DIR / "proper_eval_selected_nli_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    grid_path = METRIC_DIR / "proper_eval_selected_nli_train_grid.csv"
    pd.DataFrame(all_grid).to_csv(grid_path, index=False)

    print("\n" + "#" * 100)
    print("FINAL PROPER EVAL SUMMARY:")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary: {summary_path}")
    print(f"Saved grid:    {grid_path}")

if __name__ == "__main__":
    main()
