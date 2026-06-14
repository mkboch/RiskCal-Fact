import json
import traceback
from pathlib import Path

from datasets import load_dataset

out_dir = Path("data/raw/scifact_access_test")
out_dir.mkdir(parents=True, exist_ok=True)

candidates = [
    ("allenai/scifact", None),
    ("scifact", None),
    ("BeIR/scifact", "corpus"),
    ("BeIR/scifact", "queries"),
    ("BeIR/scifact-qrels", None),
]

summary = []

for name, config in candidates:
    print("\n" + "=" * 80)
    print(f"Trying dataset: name={name!r}, config={config!r}")
    record = {"name": name, "config": config, "status": "unknown"}
    try:
        if config is None:
            ds = load_dataset(name)
        else:
            ds = load_dataset(name, config)

        print("SUCCESS")
        print(ds)
        record["status"] = "success"
        record["splits"] = list(ds.keys())

        split_info = {}
        for split in ds.keys():
            n = len(ds[split])
            cols = ds[split].column_names
            print(f"Split: {split}, rows={n}, columns={cols}")
            examples = []
            for idx in range(min(2, n)):
                ex = ds[split][idx]
                examples.append(ex)
                print(f"Example {idx}:")
                print(json.dumps(ex, indent=2, ensure_ascii=False)[:3000])
            split_info[split] = {"rows": n, "columns": cols, "examples": examples}

        record["split_info"] = split_info

        safe_name = name.replace("/", "__") + ("__" + str(config) if config else "")
        with open(out_dir / f"{safe_name}_summary.json", "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)

    except Exception as e:
        print("FAILED")
        print(repr(e))
        traceback.print_exc(limit=2)
        record["status"] = "failed"
        record["error"] = repr(e)

    summary.append(record)

with open(out_dir / "all_candidates_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 80)
print("Final candidate summary:")
for r in summary:
    print(f"{r['status'].upper():8s} | {r['name']} | {r['config']} | {r.get('splits', '')} | {r.get('error', '')}")

print("\nSaved summaries to:", out_dir.resolve())
