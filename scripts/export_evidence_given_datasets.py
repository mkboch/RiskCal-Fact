import json
import re
import random
from pathlib import Path
from collections import Counter, defaultdict
from datasets import load_dataset

OUT = Path("data/processed/evidence_given")
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)

def write_jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def norm_label_text(x):
    s = str(x).strip().lower().replace("_", " ").replace("-", " ")
    if s in {"supports", "support", "supported", "true", "mostly true"}:
        return "verified"
    if s in {"refutes", "refute", "refuted", "false", "mostly false", "pants on fire", "pants fire"}:
        return "refuted"
    if s in {"not enough info", "nei", "unknown", "unproven", "mixture", "mixed", "half true", "barely true"}:
        return "unsupported"
    return None

def label_from_feature(ds_split, raw_label):
    try:
        feat = ds_split.features.get("label", None)
        if hasattr(feat, "int2str") and isinstance(raw_label, int):
            return feat.int2str(raw_label)
    except Exception:
        pass
    return raw_label

def split_sentences(text, max_units=5):
    text = str(text or "").replace("\n", " ").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    parts = [p.strip() for p in parts if len(p.strip()) > 20]
    if not parts:
        return [text[:2500]]
    chunks = []
    cur = ""
    for p in parts:
        if len(cur) + len(p) < 900:
            cur = (cur + " " + p).strip()
        else:
            if cur:
                chunks.append(cur)
            cur = p
        if len(chunks) >= max_units:
            break
    if cur and len(chunks) < max_units:
        chunks.append(cur)
    return chunks[:max_units]

def export_vitaminc():
    print("\n==== Export VitaminC ====")
    ds = load_dataset("tals/vitaminc", "default", trust_remote_code=True)
    print(ds)

    summary = {"dataset": "vitaminc", "splits": {}}

    for split in ds.keys():
        rows = []
        labels = Counter()

        for ex in ds[split]:
            label = norm_label_text(ex.get("label", ""))
            if label is None:
                continue
            evidence = str(ex.get("evidence", "") or "").strip()
            if not evidence:
                continue

            row = {
                "id": str(ex.get("unique_id", "")),
                "dataset": "vitaminc",
                "split": split,
                "claim": str(ex.get("claim", "")),
                "label": label,
                "evidence_units": [evidence],
                "metadata": {
                    "page": ex.get("page", ""),
                    "revision_type": ex.get("revision_type", ""),
                    "case_id": ex.get("case_id", ""),
                    "FEVER_id": ex.get("FEVER_id", ""),
                },
            }
            rows.append(row)
            labels[label] += 1

        path = OUT / f"vitaminc_{split}.jsonl"
        write_jsonl(path, rows)
        summary["splits"][split] = {"rows": len(rows), "label_counts": dict(labels), "path": str(path)}
        print(split, summary["splits"][split])

    return summary

def export_pubhealth():
    print("\n==== Export PubHealth ====")
    ds = load_dataset("health_fact", "default", trust_remote_code=True)
    print(ds)

    summary = {"dataset": "pubhealth", "splits": {}}

    for split in ds.keys():
        rows = []
        labels = Counter()
        raw_labels = Counter()

        for ex in ds[split]:
            raw = label_from_feature(ds[split], ex.get("label", ""))
            raw_labels[str(raw)] += 1
            label = norm_label_text(raw)

            if label is None:
                continue

            claim = str(ex.get("claim", "") or "").strip()
            explanation = str(ex.get("explanation", "") or "").strip()
            main_text = str(ex.get("main_text", "") or "").strip()

            evidence_units = []
            if explanation:
                evidence_units.append(explanation[:2500])
            evidence_units.extend(split_sentences(main_text, max_units=4))
            evidence_units = [x for x in evidence_units if x.strip()]

            if not claim or not evidence_units:
                continue

            rows.append({
                "id": str(ex.get("claim_id", "")),
                "dataset": "pubhealth",
                "split": split,
                "claim": claim,
                "label": label,
                "evidence_units": evidence_units[:5],
                "metadata": {
                    "date_published": ex.get("date_published", ""),
                    "subjects": ex.get("subjects", ""),
                    "raw_label": str(raw),
                },
            })
            labels[label] += 1

        path = OUT / f"pubhealth_{split}.jsonl"
        write_jsonl(path, rows)
        summary["splits"][split] = {
            "rows": len(rows),
            "label_counts": dict(labels),
            "raw_label_counts": dict(raw_labels),
            "path": str(path),
        }
        print(split, summary["splits"][split])

    return summary

def climate_label(x):
    # Climate-FEVER uses FEVER-style integer labels in the HF version:
    # 0 SUPPORTS, 1 REFUTES, 2 NOT_ENOUGH_INFO.
    try:
        x = int(x)
    except Exception:
        return norm_label_text(x)
    return {0: "verified", 1: "refuted", 2: "unsupported"}.get(x, None)

def stratified_split(rows):
    by = defaultdict(list)
    for r in rows:
        by[r["label"]].append(r)

    tune, cal, dev = [], [], []
    rng = random.Random(SEED)

    for lab, items in by.items():
        rng.shuffle(items)
        n = len(items)
        n_tune = int(0.4 * n)
        n_cal = int(0.2 * n)
        tune.extend(items[:n_tune])
        cal.extend(items[n_tune:n_tune+n_cal])
        dev.extend(items[n_tune+n_cal:])

    rng.shuffle(tune)
    rng.shuffle(cal)
    rng.shuffle(dev)
    return {"tune": tune, "cal": cal, "dev": dev}

def export_climate_fever():
    print("\n==== Export Climate-FEVER ====")
    ds = load_dataset("climate_fever", "default", trust_remote_code=True)
    print(ds)

    all_rows = []
    labels = Counter()

    for ex in ds["test"]:
        label = climate_label(ex.get("claim_label"))
        if label is None:
            continue

        evidences = ex.get("evidences", []) or []
        evidence_units = []
        for ev in evidences:
            txt = str(ev.get("evidence", "") or "").strip()
            art = str(ev.get("article", "") or "").strip()
            if txt:
                evidence_units.append((art + ". " + txt).strip())

        if not evidence_units:
            continue

        row = {
            "id": str(ex.get("claim_id", "")),
            "dataset": "climate_fever",
            "split": "test",
            "claim": str(ex.get("claim", "")),
            "label": label,
            "evidence_units": evidence_units[:10],
            "metadata": {
                "claim_label": ex.get("claim_label"),
                "num_evidences": len(evidence_units),
            },
        }
        all_rows.append(row)
        labels[label] += 1

    splits = stratified_split(all_rows)

    summary = {
        "dataset": "climate_fever",
        "original_rows": len(all_rows),
        "original_label_counts": dict(labels),
        "splits": {},
    }

    for split, rows in splits.items():
        path = OUT / f"climate_fever_{split}.jsonl"
        write_jsonl(path, rows)
        summary["splits"][split] = {
            "rows": len(rows),
            "label_counts": dict(Counter(r["label"] for r in rows)),
            "path": str(path),
        }
        print(split, summary["splits"][split])

    return summary

def main():
    summaries = []
    summaries.append(export_vitaminc())
    summaries.append(export_pubhealth())
    summaries.append(export_climate_fever())

    summary_path = OUT / "evidence_given_export_summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nSaved summary:", summary_path)
    print(json.dumps(summaries, indent=2, ensure_ascii=False)[:8000])

if __name__ == "__main__":
    main()
