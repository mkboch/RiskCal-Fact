import json
from pathlib import Path
from collections import Counter
from datasets import load_dataset

RAW_OUT = Path("data/raw/fever_hf_export")
PROC_OUT = Path("data/processed/fever")
RAW_OUT.mkdir(parents=True, exist_ok=True)
PROC_OUT.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_label(label):
    s = str(label).strip().upper().replace(" ", "_")
    if s in {"SUPPORTS", "SUPPORTED", "SUPPORT"}:
        return "verified"
    if s in {"REFUTES", "REFUTED", "REFUTE"}:
        return "refuted"
    if s in {"NOT_ENOUGH_INFO", "NEI"}:
        return "unsupported"
    return s.lower()

def extract_gold_evidence(ex):
    out = []
    ev = ex.get("evidence", None)

    # FEVER evidence in HF usually has nested lists with fields:
    # annotation_id, evidence_id, wikipedia_url, sentence_id.
    if ev is None:
        return out

    try:
        # Some HF versions store evidence as list of evidence groups.
        if isinstance(ev, list):
            for group in ev:
                if isinstance(group, list):
                    for item in group:
                        if isinstance(item, dict):
                            out.append(item)
                        elif isinstance(item, (list, tuple)) and len(item) >= 4:
                            out.append({
                                "annotation_id": item[0],
                                "evidence_id": item[1],
                                "wikipedia_url": item[2],
                                "sentence_id": item[3],
                            })
                elif isinstance(group, dict):
                    out.append(group)

        elif isinstance(ev, dict):
            # Some versions store columnar lists.
            keys = list(ev.keys())
            lengths = [len(ev[k]) for k in keys if hasattr(ev[k], "__len__") and not isinstance(ev[k], str)]
            if lengths:
                n = max(lengths)
                for i in range(n):
                    item = {}
                    for k in keys:
                        v = ev[k]
                        try:
                            item[k] = v[i]
                        except Exception:
                            item[k] = v
                    out.append(item)
            else:
                out.append(ev)

    except Exception as e:
        out.append({"parse_error": repr(e), "raw": ev})

    return out

def normalize_split(ds, split):
    rows = []
    labels = Counter()

    for ex in ds[split]:
        label = norm_label(ex.get("label", ""))
        labels[label] += 1

        rows.append({
            "id": str(ex.get("id", "")),
            "dataset": "fever",
            "split": split,
            "claim": ex.get("claim", ""),
            "label": label,
            "gold_evidence": extract_gold_evidence(ex),
            "metadata": ex,
        })

    return rows, labels

def main():
    print("Loading fever/fever with old datasets...")
    ds = load_dataset("fever/fever", "v1.0", trust_remote_code=True)
    print(ds)

    summary = {
        "splits": {},
        "processed_files": {},
    }

    for split in ds.keys():
        print("\n" + "=" * 80)
        print("Split:", split)
        print("Rows:", len(ds[split]))
        print("Columns:", ds[split].column_names)
        print("Example:")
        print(json.dumps(ds[split][0], indent=2, ensure_ascii=False)[:5000])

        # Raw export.
        raw_rows = [dict(x) for x in ds[split]]
        raw_path = RAW_OUT / f"{split}.jsonl"
        write_jsonl(raw_path, raw_rows)

        # Normalized export.
        rows, labels = normalize_split(ds, split)
        proc_path = PROC_OUT / f"{split}.jsonl"
        write_jsonl(proc_path, rows)

        summary["splits"][split] = {
            "rows": len(rows),
            "label_counts": dict(labels),
            "columns": list(ds[split].column_names),
        }
        summary["processed_files"][split] = str(proc_path)

        print("Label counts:", dict(labels))
        print("Saved raw:", raw_path)
        print("Saved processed:", proc_path)

    summary_path = PROC_OUT / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nFinal summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved summary:", summary_path)

if __name__ == "__main__":
    main()
