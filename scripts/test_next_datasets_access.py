import json
import traceback
from pathlib import Path
from datasets import load_dataset, get_dataset_config_names

OUT = Path("data/raw/next_dataset_access_tests")
OUT.mkdir(parents=True, exist_ok=True)

candidates = [
    # FEVEROUS candidates
    {"group": "FEVEROUS", "name": "fever/feverous", "configs": "auto"},
    {"group": "FEVEROUS", "name": "feverous", "configs": "auto"},

    # VitaminC candidates
    {"group": "VitaminC", "name": "tals/vitaminc", "configs": "auto"},
    {"group": "VitaminC", "name": "vitaminc", "configs": "auto"},

    # PubHealth candidates
    {"group": "PubHealth", "name": "health_fact", "configs": "auto"},
    {"group": "PubHealth", "name": "ImperialCollegeLondon/health_fact", "configs": "auto"},

    # Climate-FEVER candidates
    {"group": "Climate-FEVER", "name": "climate_fever", "configs": "auto"},
    {"group": "Climate-FEVER", "name": "tdiggelm/climate_fever", "configs": "auto"},
]

summary = []

def try_configs(name):
    try:
        cfgs = get_dataset_config_names(name, trust_remote_code=True)
        if cfgs:
            return cfgs
        return [None]
    except Exception:
        return [None]

for cand in candidates:
    group = cand["group"]
    name = cand["name"]

    print("\n" + "=" * 100)
    print(f"GROUP={group} NAME={name}")

    configs = try_configs(name) if cand["configs"] == "auto" else cand["configs"]
    print("Candidate configs:", configs[:20] if isinstance(configs, list) else configs)

    tested_any = False

    for config in configs[:5]:
        tested_any = True
        rec = {
            "group": group,
            "name": name,
            "config": config,
            "status": "unknown",
        }

        print("\n" + "-" * 80)
        print(f"Trying name={name!r}, config={config!r}")

        try:
            if config is None:
                ds = load_dataset(name, trust_remote_code=True)
            else:
                ds = load_dataset(name, config, trust_remote_code=True)

            rec["status"] = "success"
            rec["splits"] = list(ds.keys())
            rec["split_info"] = {}

            print("SUCCESS")
            print(ds)

            for split in ds.keys():
                n = len(ds[split])
                cols = ds[split].column_names
                examples = []

                for i in range(min(2, n)):
                    ex = ds[split][i]
                    examples.append(ex)

                rec["split_info"][split] = {
                    "rows": n,
                    "columns": cols,
                    "examples": examples,
                }

                print(f"Split={split}, rows={n}, columns={cols}")
                print(json.dumps(examples[:1], indent=2, ensure_ascii=False)[:3000])

            safe = f"{group}__{name.replace('/', '__')}__{str(config).replace('/', '__')}"
            path = OUT / f"{safe}_summary.json"
            path.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
            rec["summary_path"] = str(path)

            # If one config works for a dataset group/name, do not test more configs for now.
            summary.append(rec)
            break

        except Exception as e:
            print("FAILED:", repr(e))
            traceback.print_exc(limit=2)
            rec["status"] = "failed"
            rec["error"] = repr(e)
            summary.append(rec)

    if not tested_any:
        summary.append({
            "group": group,
            "name": name,
            "config": None,
            "status": "no_configs_found",
        })

summary_path = OUT / "next_datasets_access_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

print("\n" + "=" * 100)
print("FINAL NEXT DATASET ACCESS SUMMARY")
for r in summary:
    print(f"{r['group']:15s} | {r['status']:10s} | {r['name']} | {r.get('config')} | {r.get('splits', '')} | {r.get('error', '')[:160]}")

print("\nSaved:", summary_path)
