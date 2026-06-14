import json
import math
import re
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from tqdm import tqdm

PROC_DIR = Path("data/processed/scifact")
INDEX_DIR = Path("data/indexes/scifact")
OUT_DIR = Path("outputs/predictions/scifact")
METRIC_DIR = Path("outputs/metrics/scifact")
INDEX_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)

TOPK = 10

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

_token_re = re.compile(r"[A-Za-z0-9]+")

def tokenize(text):
    return _token_re.findall((text or "").lower())

def gold_doc_ids(example):
    ids = set()
    for ev in example.get("gold_evidence", []):
        doc_id = ev.get("doc_id")
        if doc_id is not None:
            ids.add(str(doc_id))
    # Unsupported claims have no gold evidence. We exclude them from evidence-recall denominators.
    return ids

def build_or_load_bm25(corpus):
    index_path = INDEX_DIR / "bm25_doc_index.pkl"
    meta_path = INDEX_DIR / "bm25_doc_meta.jsonl"

    if index_path.exists() and meta_path.exists():
        print(f"Loading existing BM25 index: {index_path}")
        with index_path.open("rb") as f:
            bm25 = pickle.load(f)
        meta = read_jsonl(meta_path)
        return bm25, meta

    print("Building BM25 index...")
    tokenized = []
    meta = []

    for doc in tqdm(corpus, desc="Tokenizing corpus"):
        doc_id = str(doc["doc_id"])
        title = doc.get("title", "")
        text = doc.get("text", "")
        full_text = (title + " " + text).strip()
        tokenized.append(tokenize(full_text))
        meta.append({
            "doc_id": doc_id,
            "title": title,
            "text": text,
        })

    bm25 = BM25Okapi(tokenized)

    with index_path.open("wb") as f:
        pickle.dump(bm25, f)

    write_jsonl(meta_path, meta)
    print(f"Saved BM25 index: {index_path}")
    print(f"Saved BM25 metadata: {meta_path}")

    return bm25, meta

def retrieve_split(split_name, bm25, meta):
    examples = read_jsonl(PROC_DIR / f"{split_name}.jsonl")
    predictions = []

    recall_counts = {1: 0, 3: 0, 5: 0, 10: 0}
    denom = 0

    for ex in tqdm(examples, desc=f"Retrieving {split_name}"):
        claim = ex["claim"]
        query_tokens = tokenize(claim)
        scores = bm25.get_scores(query_tokens)

        top_idx = np.argsort(scores)[::-1][:TOPK]
        retrieved = []

        for rank, idx in enumerate(top_idx, start=1):
            m = meta[int(idx)]
            retrieved.append({
                "rank": rank,
                "doc_id": str(m["doc_id"]),
                "title": m.get("title", ""),
                "score": float(scores[idx]),
                "text": m.get("text", "")[:2000],
            })

        g = gold_doc_ids(ex)
        if len(g) > 0:
            denom += 1
            retrieved_ids = [r["doc_id"] for r in retrieved]
            for k in recall_counts:
                if any(doc_id in g for doc_id in retrieved_ids[:k]):
                    recall_counts[k] += 1

        predictions.append({
            "id": ex["id"],
            "dataset": ex["dataset"],
            "split": split_name,
            "claim": claim,
            "label": ex["label"],
            "gold_doc_ids": sorted(list(g)),
            "retrieved": retrieved,
        })

    out_path = OUT_DIR / f"bm25_top{TOPK}_{split_name}.jsonl"
    write_jsonl(out_path, predictions)

    metrics = {
        "dataset": "scifact",
        "split": split_name,
        "method": "bm25_doc_retrieval",
        "topk": TOPK,
        "num_examples": len(examples),
        "num_examples_with_gold_evidence": denom,
    }

    for k, v in recall_counts.items():
        metrics[f"evidence_doc_recall@{k}"] = v / denom if denom > 0 else None

    metric_path = METRIC_DIR / f"bm25_top{TOPK}_{split_name}_metrics.json"
    with metric_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved predictions: {out_path}")
    print(f"Saved metrics: {metric_path}")
    print(json.dumps(metrics, indent=2))

    return metrics

def main():
    corpus_path = PROC_DIR / "corpus.jsonl"
    if not corpus_path.exists():
        raise FileNotFoundError(f"Missing {corpus_path}. Run SciFact processing first.")

    corpus = read_jsonl(corpus_path)
    print(f"Loaded corpus docs: {len(corpus)}")

    bm25, meta = build_or_load_bm25(corpus)
    print(f"BM25 meta docs: {len(meta)}")

    all_metrics = []
    for split in ["train", "dev"]:
        all_metrics.append(retrieve_split(split, bm25, meta))

    summary_path = METRIC_DIR / "bm25_top10_summary.csv"
    pd.DataFrame(all_metrics).to_csv(summary_path, index=False)
    print(f"\nSaved summary CSV: {summary_path}")

if __name__ == "__main__":
    main()
