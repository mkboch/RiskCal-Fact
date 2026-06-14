import json
from pathlib import Path

FINAL = Path("outputs/final_report")
FINAL.mkdir(parents=True, exist_ok=True)

status = {
    "paper_experiment_status": "ready_for_manuscript_drafting",
    "completed": [
        "SciFact full labelled evaluation",
        "FEVER pilot evaluation",
        "FEVER full paper_dev evaluation with large sampled evidence corpus",
        "VitaminC evidence-given verification",
        "PubHealth evidence-given verification",
        "Climate-FEVER evidence-given verification",
        "FEVEROUS structured-provenance characterization",
        "Unified risk-coverage figures",
        "Unified LaTeX result tables",
    ],
    "recommended_paper_framing": {
        "main_claim": "Verification should be formulated as risk-calibrated evidence-constrained decision-making rather than a single classifier score.",
        "core_equation": "verified iff S(c) >= tau_s and K(c) < tau_k and R(c)=1 and P(c)=1 and rho(c)<=alpha",
        "main_experiment_groups": [
            "retrieval-grounded claim verification: SciFact and FEVER",
            "evidence-given verification: VitaminC, PubHealth, Climate-FEVER",
            "structured-provenance stress test: FEVEROUS",
        ],
    },
}
path = FINAL / "project_ready_for_paper_status.json"
path.write_text(json.dumps(status, indent=2), encoding="utf-8")
print("Saved:", path)
print(json.dumps(status, indent=2))
