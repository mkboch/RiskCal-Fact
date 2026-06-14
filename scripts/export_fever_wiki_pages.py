import json
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/raw/fever_wiki_pages")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "wiki_pages.jsonl"
SUMMARY_PATH = OUT_DIR / "summary.json"

if OUT_PATH.exists() and OUT_PATH.stat().st_size > 1000000:
    print(f"Wiki pages already exported: {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    if SUMMARY_PATH.exists():
        print(SUMMARY_PATH.read_text()[:3000])
    raise SystemExit(0)

print("Loading fever/fever wiki_pages...")
ds = load_dataset("fever/fever", "wiki_pages", trust_remote_code=True)
print(ds)

summary = {"splits": {}}
total = 0

with OUT_PATH.open("w", encoding="utf-8") as fout:
    for split in ds.keys():
        n = len(ds[split])
        cols = ds[split].column_names
        summary["splits"][split] = {"rows": n, "columns": cols}
        print(f"Split={split}, rows={n}, columns={cols}")
        print("Example:", json.dumps(ds[split][0], ensure_ascii=False)[:2000])

        for row in ds[split]:
            item = dict(row)
            item["_split"] = split
            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += 1

summary["total_rows"] = total
summary["output"] = str(OUT_PATH)

with SUMMARY_PATH.open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("Saved wiki pages:", OUT_PATH)
print("Summary:", json.dumps(summary, indent=2)[:5000])
