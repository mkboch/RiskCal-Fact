from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path.home() / "risk_calibrated_verification"
INPUT = ROOT / "outputs/metrics/review_hardening/fast_ablation_all.csv"
OUT_DIR = ROOT / "paper_assets/figures"

OUT_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT}")

df = pd.read_csv(INPUT)

print("Input columns:", list(df.columns))
print("Input rows:", len(df))

required = {
    "dataset",
    "model",
    "rule",
    "alpha",
    "coverage",
    "false_verification_rate",
}
missing = required.difference(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Keep the complete risk-gated rule.
risk = df[df["rule"].astype(str).eq("S+K+P+rho")].copy()

# Use the RoBERTa-large verifier consistently across benchmarks.
risk = risk[
    risk["model"]
    .astype(str)
    .str.contains("roberta|ynie", case=False, regex=True, na=False)
].copy()

risk["alpha"] = pd.to_numeric(risk["alpha"], errors="coerce")
risk["coverage"] = pd.to_numeric(risk["coverage"], errors="coerce")
risk["false_verification_rate"] = pd.to_numeric(
    risk["false_verification_rate"], errors="coerce"
)

risk = risk.dropna(
    subset=["alpha", "coverage", "false_verification_rate"]
)

display_names = {
    "fever_full_dev": "FEVER-full-dev",
    "vitaminc": "VitaminC",
    "pubhealth": "PubHealth",
    "climate_fever": "Climate-FEVER",
}

dataset_order = [
    "FEVER-full-dev",
    "VitaminC",
    "PubHealth",
    "Climate-FEVER",
]

risk["Dataset"] = risk["dataset"].map(display_names).fillna(risk["dataset"])

# Keep only the alpha values used in the paper.
risk = risk[
    (risk["alpha"] >= 0.05) &
    (risk["alpha"] <= 0.30)
].copy()

if risk.empty:
    raise RuntimeError("No matching risk-calibration rows were found.")

print("\nRows used for Figure 4:")
print(
    risk[
        [
            "Dataset",
            "alpha",
            "coverage",
            "false_verification_rate",
        ]
    ]
    .sort_values(["Dataset", "alpha"])
    .to_string(index=False)
)

fig, ax = plt.subplots(figsize=(8.8, 5.8))

for dataset in dataset_order:
    sub = risk[risk["Dataset"] == dataset].copy()

    if sub.empty:
        print(f"WARNING: no rows found for {dataset}")
        continue

    # Sort by coverage so connecting lines follow the plotted frontier.
    sub = sub.sort_values(["coverage", "alpha"])

    ax.plot(
        sub["coverage"],
        sub["false_verification_rate"],
        marker="o",
        linewidth=1.8,
        markersize=5.5,
        label=dataset,
    )

ax.set_xlabel("Coverage", fontsize=11)
ax.set_ylabel("False-verification rate", fontsize=11)
ax.set_title(
    r"Risk--coverage operating points across the evaluated $\alpha$ grid",
    fontsize=12,
)

ax.set_xlim(
    max(0.0, risk["coverage"].min() - 0.04),
    min(1.02, risk["coverage"].max() + 0.02),
)
ax.set_ylim(
    0.0,
    min(1.0, risk["false_verification_rate"].max() + 0.05),
)

ax.grid(True, linewidth=0.5, alpha=0.3)
ax.legend(
    frameon=False,
    loc="best",
    fontsize=9,
)

fig.tight_layout()

pdf_path = OUT_DIR / "fig4_risk_coverage_operating_points.pdf"
png_path = OUT_DIR / "fig4_risk_coverage_operating_points.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print("\nSaved:")
print(pdf_path)
print(png_path)
