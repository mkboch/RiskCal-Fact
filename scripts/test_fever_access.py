import json
import traceback
from pathlib import Path
from datasets import load_dataset

OUT_DIR = Path("data/raw/fever_access_test")
OUT_DIR.mkdir(parents=True, exist_ok=True)

candidates = [
    ("fever/fever", None),
    ("fever", None),
    ("pminervini/fever", None),
    ("BeIR/fever", "corpus"),
    ("BeIR/fever", "queries"),
    ("BeIR/fever-qrels", None),
]

summary = []

for name, config in candidates:
    print("\n" + "=" * 100)
    print(f"Trying dataset: name={name!r}, config={config!r}")
    rec = {"name": name, "config": config, "status": "unknown"}

    try:
        if config is None:
            ds = load_dataset(name)
        else:
            ds = load_dataset(name, config)

        print("SUCCESS")
        print(ds)

        rec["status"] = "success"
        rec["splits"] = list(ds.keys())
        rec["split_info"] = {}

        for split in ds.keys():
            n = len(ds[split])
            cols = ds[split].column_names
            print(f"\nSplit: {split}")
            print(f"Rows: {n}")
            print(f"Columns: {cols}")

            examples = []
            for i in range(min(2, n)):
                ex = ds[split][i]
                examples.append(ex)
                print(f"Example {i}:")
                print(json.dumps(ex, indent=2, ensure_ascii=False)[:4000])

            rec["split_info"][split] = {
                "rows": n,
                "columns": cols,
                "examples": examples,
            }

        safe = name.replace("/", "__") + ("__" + str(config) if config else "")
        with open(OUT_DIR / f"{safe}_summary.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print("FAILED")
        print(repr(e))
        traceback.print_exc(limit=2)
        rec["status"] = "failed"
        rec["error"] = repr(e)

    summary.append(rec)

with open(OUT_DIR / "all_candidates_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 100)
print("Final FEVER candidate summary:")
for r in summary:
    print(f"{r['status'].upper():8s} | {r['name']} | {r['config']} | {r.get('splits', '')} | {r.get('error', '')}")

print("\nSaved summaries to:", OUT_DIR.resolve())
