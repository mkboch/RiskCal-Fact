import os
import json
import tarfile
import urllib.request
from pathlib import Path
from collections import Counter, defaultdict

RAW_DIR = Path("data/raw/scifact_official")
PROC_DIR = Path("data/processed/scifact")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

URLS = [
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz",
    "https://scifact.s3-us-west-2.amazonaws.com/release/latest/scifact.tar.gz",
]

tar_path = RAW_DIR / "scifact_data.tar.gz"

def download():
    if tar_path.exists() and tar_path.stat().st_size > 1000:
        print(f"Tarball already exists: {tar_path} ({tar_path.stat().st_size} bytes)")
        return

    last_err = None
    for url in URLS:
        print(f"Trying download: {url}")
        try:
            urllib.request.urlretrieve(url, tar_path)
            print(f"Downloaded to {tar_path} ({tar_path.stat().st_size} bytes)")
            return
        except Exception as e:
            print(f"Failed URL: {url}")
            print(repr(e))
            last_err = e

    raise RuntimeError(f"All download URLs failed. Last error: {last_err!r}")

def extract():
    extract_dir = RAW_DIR / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)

    marker = extract_dir / ".extracted_ok"
    if marker.exists():
        print(f"Already extracted: {extract_dir}")
        return extract_dir

    print(f"Extracting {tar_path} -> {extract_dir}")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)
    marker.write_text("ok\n")
    return extract_dir

def find_file(root: Path, filename: str):
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find {filename} under {root}")
    if len(matches) > 1:
        print(f"WARNING: multiple matches for {filename}: {matches}")
    return matches[0]

def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

def normalize_claim_label(evidence_dict):
    labels = []
    if not evidence_dict:
        return "unsupported"
    for doc_id, entries in evidence_dict.items():
        for ev in entries:
            label = ev.get("label")
            if label:
                labels.append(str(label).upper())
    if any(x == "SUPPORT" or x == "SUPPORTED" for x in labels):
        return "verified"
    if any(x == "CONTRADICT" or x == "REFUTE" or x == "REFUTED" for x in labels):
        return "refuted"
    return "unsupported"

def flatten_gold_evidence(evidence_dict, corpus_map):
    out = []
    if not evidence_dict:
        return out

    for doc_id, entries in evidence_dict.items():
        doc_id_str = str(doc_id)
        doc = corpus_map.get(doc_id_str, {})
        title = doc.get("title", "")
        abstract = doc.get("abstract", doc.get("text", ""))
        abstract_sentences = doc.get("abstract", [])
        if isinstance(abstract_sentences, str):
            abstract_sentences = [abstract_sentences]

        for ev in entries:
            label = ev.get("label", "")
            sent_ids = ev.get("sentences", [])
            sent_texts = []
            for sid in sent_ids:
                try:
                    sid_int = int(sid)
                    if 0 <= sid_int < len(abstract_sentences):
                        sent_texts.append(abstract_sentences[sid_int])
                except Exception:
                    pass

            out.append({
                "doc_id": doc_id_str,
                "title": title,
                "label": label,
                "sentences": sent_ids,
                "sentence_texts": sent_texts,
                "doc_text": " ".join(abstract_sentences) if isinstance(abstract_sentences, list) else str(abstract),
            })
    return out

def process():
    extract_dir = RAW_DIR / "extracted"

    corpus_path = find_file(extract_dir, "corpus.jsonl")
    train_path = find_file(extract_dir, "claims_train.jsonl")
    dev_path = find_file(extract_dir, "claims_dev.jsonl")
    test_path = find_file(extract_dir, "claims_test.jsonl")

    print("Found files:")
    print("corpus:", corpus_path)
    print("train :", train_path)
    print("dev   :", dev_path)
    print("test  :", test_path)

    corpus = read_jsonl(corpus_path)
    train = read_jsonl(train_path)
    dev = read_jsonl(dev_path)
    test = read_jsonl(test_path)

    print("\nRaw counts:")
    print("corpus:", len(corpus))
    print("train :", len(train))
    print("dev   :", len(dev))
    print("test  :", len(test))

    print("\nExample corpus row:")
    print(json.dumps(corpus[0], indent=2, ensure_ascii=False)[:3000])
    print("\nExample train claim:")
    print(json.dumps(train[0], indent=2, ensure_ascii=False)[:3000])
    print("\nExample dev claim:")
    print(json.dumps(dev[0], indent=2, ensure_ascii=False)[:3000])
    print("\nExample test claim:")
    print(json.dumps(test[0], indent=2, ensure_ascii=False)[:3000])

    corpus_map = {str(x.get("doc_id", x.get("_id", x.get("id")))): x for x in corpus}

    # Save normalized corpus.
    norm_corpus = []
    for row in corpus:
        doc_id = str(row.get("doc_id", row.get("_id", row.get("id"))))
        title = row.get("title", "")
        abstract = row.get("abstract", row.get("text", ""))
        if isinstance(abstract, list):
            text = " ".join(abstract)
            sentences = abstract
        else:
            text = str(abstract)
            sentences = [text]
        norm_corpus.append({
            "doc_id": doc_id,
            "title": title,
            "text": text,
            "sentences": sentences,
            "metadata": row,
        })

    write_jsonl(PROC_DIR / "corpus.jsonl", norm_corpus)

    def normalize_split(rows, split_name, has_labels=True):
        out = []
        label_counts = Counter()
        evidence_counts = []
        for row in rows:
            claim_id = str(row.get("id", row.get("claim_id", row.get("_id"))))
            claim = row.get("claim", row.get("text", ""))
            evidence = row.get("evidence", {})
            label = normalize_claim_label(evidence) if has_labels else "unlabeled"
            gold = flatten_gold_evidence(evidence, corpus_map) if has_labels else []

            label_counts[label] += 1
            evidence_counts.append(len(gold))

            out.append({
                "id": claim_id,
                "dataset": "scifact",
                "split": split_name,
                "claim": claim,
                "label": label,
                "gold_evidence": gold,
                "candidate_corpus": "scifact_corpus",
                "metadata": row,
            })

        write_jsonl(PROC_DIR / f"{split_name}.jsonl", out)
        return {
            "split": split_name,
            "rows": len(out),
            "label_counts": dict(label_counts),
            "gold_evidence_min": min(evidence_counts) if evidence_counts else 0,
            "gold_evidence_max": max(evidence_counts) if evidence_counts else 0,
            "gold_evidence_mean": sum(evidence_counts) / len(evidence_counts) if evidence_counts else 0,
        }

    summaries = []
    summaries.append(normalize_split(train, "train", has_labels=True))
    summaries.append(normalize_split(dev, "dev", has_labels=True))
    summaries.append(normalize_split(test, "test", has_labels=False))

    summary = {
        "raw_files": {
            "corpus": str(corpus_path),
            "train": str(train_path),
            "dev": str(dev_path),
            "test": str(test_path),
        },
        "processed_files": {
            "corpus": str(PROC_DIR / "corpus.jsonl"),
            "train": str(PROC_DIR / "train.jsonl"),
            "dev": str(PROC_DIR / "dev.jsonl"),
            "test": str(PROC_DIR / "test.jsonl"),
        },
        "corpus_rows": len(norm_corpus),
        "splits": summaries,
    }

    with (PROC_DIR / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nProcessed summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\nProcessed examples:")
    for fname in ["corpus.jsonl", "train.jsonl", "dev.jsonl", "test.jsonl"]:
        p = PROC_DIR / fname
        print(f"\n--- {p} ---")
        rows = read_jsonl(p)
        print(json.dumps(rows[0], indent=2, ensure_ascii=False)[:3000])

def main():
    download()
    extract()
    process()

if __name__ == "__main__":
    main()
