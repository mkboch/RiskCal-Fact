import json
import random
from pathlib import Path
from collections import defaultdict
from datasets import load_dataset

SEED = 42
random.seed(SEED)

OUT = Path("outputs/predictions/review_hardening")
MET = Path("outputs/metrics/review_hardening")
OUT.mkdir(parents=True, exist_ok=True)
MET.mkdir(parents=True, exist_ok=True)

LABELS = ["verified", "refuted", "unsupported"]

def read_jsonl(path):
    rows = []
    p = Path(path)
    if not p.exists():
        return rows
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_title(x):
    return str(x).replace(" ", "_")

def label_norm(x):
    s = str(x).strip().lower()
    if s in {"supports", "support", "supported", "verified", "1"}:
        return "verified"
    if s in {"refutes", "refute", "refuted", "0"}:
        return "refuted"
    if s in {"not enough info", "nei", "unsupported", "2"}:
        return "unsupported"
    return s

def first_or_list(x):
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]

def get_flat_evidence_pairs(row):
    pages = first_or_list(row.get("evidence_wiki_url"))
    sids = first_or_list(row.get("evidence_sentence_id"))

    pairs = []
    for page, sid in zip(pages, sids):
        if page is None or sid is None:
            continue
        page = str(page).strip()
        if page == "" or page.lower() in {"none", "null"}:
            continue
        try:
            sid_int = int(sid)
        except Exception:
            continue
        if sid_int < 0:
            continue
        pairs.append((norm_title(page), sid_int))

    seen = set()
    out = []
    for p in pairs:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out

def load_corpus_map():
    candidates = [
        Path("data/processed/fever/full_dev_sentence_corpus.jsonl"),
        Path("data/processed/fever/pilot_sentence_corpus.jsonl"),
    ]

    corpus_file = None
    for c in candidates:
        if c.exists():
            corpus_file = c
            break

    if corpus_file is None:
        raise FileNotFoundError("No FEVER sentence corpus found.")

    rows = read_jsonl(corpus_file)
    m = {}

    for r in rows:
        text = r.get("text") or r.get("sentence") or ""
        page = r.get("wiki_url") or r.get("wikipedia_url") or r.get("page") or r.get("title") or r.get("doc_id") or ""
        sid = r.get("sentence_id") or r.get("line_num") or r.get("line") or None

        if "sent_id" in r:
            sid_str = str(r["sent_id"])
            if "::" in sid_str:
                page2, sid2 = sid_str.rsplit("::", 1)
                try:
                    m[(norm_title(page2), int(sid2))] = text
                except Exception:
                    pass

        if page and sid is not None:
            try:
                m[(norm_title(page), int(sid))] = text
            except Exception:
                pass

    print(f"Using corpus: {corpus_file}")
    print(f"Corpus lookup entries: {len(m)}")
    return m, corpus_file

def row_to_example(row, corpus_map, source_split, idx):
    claim = row.get("claim", "")
    label = label_norm(row.get("label", ""))
    pairs = get_flat_evidence_pairs(row)

    ev_texts = []
    found = 0
    seen_txt = set()

    for p in pairs:
        txt = corpus_map.get(p)
        if txt:
            found += 1
            if txt not in seen_txt:
                seen_txt.add(txt)
                ev_texts.append(txt)

    ev_texts = ev_texts[:5]

    return {
        "id": str(row.get("id", f"{source_split}_{idx}")),
        "source_split": source_split,
        "claim": claim,
        "label": label,
        "evidence_units": ev_texts,
        "P": 1 if ev_texts else 0,
        "num_gold_pairs": len(pairs),
        "num_gold_pairs_found": found,
        "num_oracle_units": len(ev_texts),
        "raw_evidence_wiki_url": row.get("evidence_wiki_url"),
        "raw_evidence_sentence_id": row.get("evidence_sentence_id"),
    }

def make_balanced_splits(examples):
    usable = []
    for ex in examples:
        if ex["label"] == "unsupported":
            usable.append(ex)
        elif ex["label"] in {"verified", "refuted"} and ex["P"] == 1:
            usable.append(ex)

    by = defaultdict(list)
    for ex in usable:
        by[ex["label"]].append(ex)

    for lab in LABELS:
        random.shuffle(by[lab])
        print(f"Usable {lab}: {len(by[lab])}")

    targets = {"tune": 300, "cal": 150, "dev": 300}
    splits = {"tune": [], "cal": [], "dev": []}

    for lab in LABELS:
        start = 0
        for split, n in targets.items():
            chunk = by[lab][start:start+n]
            for ex in chunk:
                ex = dict(ex)
                ex["split"] = split
                splits[split].append(ex)
            start += n

    for split in splits:
        random.shuffle(splits[split])

    return splits, usable

def main():
    corpus_map, corpus_file = load_corpus_map()

    print("Loading raw FEVER...")
    ds = load_dataset("fever/fever", "v1.0", trust_remote_code=True)
    print(ds)

    source_splits = ["paper_dev", "labelled_dev", "train"]
    raw_examples = []
    raw_stats = []

    for split in source_splits:
        print(f"Processing {split}: {len(ds[split])}")
        part = []
        for i, row in enumerate(ds[split]):
            ex = row_to_example(row, corpus_map, split, i)
            if ex["label"] in LABELS and ex["claim"]:
                raw_examples.append(ex)
                part.append(ex)

        raw_stats.append({
            "source_split": split,
            "rows": len(part),
            "verified": sum(1 for e in part if e["label"] == "verified"),
            "refuted": sum(1 for e in part if e["label"] == "refuted"),
            "unsupported": sum(1 for e in part if e["label"] == "unsupported"),
            "verified_or_refuted_with_found_evidence": sum(1 for e in part if e["label"] in {"verified", "refuted"} and e["P"] == 1),
            "mean_gold_pairs": sum(e["num_gold_pairs"] for e in part) / len(part) if part else 0,
            "mean_found_pairs": sum(e["num_gold_pairs_found"] for e in part) / len(part) if part else 0,
        })

    splits, usable = make_balanced_splits(raw_examples)

    for split, rows in splits.items():
        write_jsonl(OUT / f"fever_oracle_flat_{split}_inputs.jsonl", rows)

    stats = {
        "corpus_file": str(corpus_file),
        "raw_stats": raw_stats,
        "usable_total": len(usable),
        "split_counts": {k: len(v) for k, v in splits.items()},
        "label_counts": {
            k: {lab: sum(1 for e in v if e["label"] == lab) for lab in LABELS}
            for k, v in splits.items()
        },
    }

    (MET / "fever_oracle_flat_input_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))

    # Hard fail if oracle evidence still absent.
    if stats["label_counts"]["dev"]["verified"] == 0 or stats["label_counts"]["dev"]["refuted"] == 0:
        raise RuntimeError("Oracle input construction failed: verified/refuted evidence not recovered.")

if __name__ == "__main__":
    main()
