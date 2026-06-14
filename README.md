# RiskCal-Fact

RiskCal-Fact is the reproducibility repository for the paper **RiskCal-Fact: Risk-Calibrated Evidence-Constrained Verification of Factual Claims**.

## Scope

This repository contains reproducibility code only. It intentionally excludes:

* manuscript source files;
* paper figures and tables;
* generated predictions;
* benchmark data;
* model checkpoints;
* execution logs; and
* temporary caches.

## Repository structure

```text
scripts/                  Experiment, calibration, evaluation, and analysis scripts
configs/                  Configuration files, when available
environment.yml           Conda environment history
requirements.txt          Minimal Python requirements
requirements_frozen.txt   Frozen package snapshot
MANIFEST.txt              Generated file inventory
```

## Experimental groups

The repository includes code for:

1. retrieval-grounded verification on SciFact and FEVER;
2. evidence-given verification on VitaminC, PubHealth, and Climate-FEVER;
3. FEVER oracle-evidence diagnostics;
4. FEVEROUS structured-provenance characterization; and
5. ablation, bootstrap confidence-interval, and risk-coverage analyses.

## Environment setup

### Conda

```bash
conda env create -f environment.yml
conda activate rcv_py310
```

### Python virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The frozen package list records the package state of the server environment used for the experiments. Because CUDA and PyTorch installations depend on the target hardware, users may need to install a PyTorch build compatible with their local CUDA runtime.

## Data

The study uses the following public benchmark datasets:

* SciFact;
* FEVER;
* VitaminC;
* PubHealth;
* Climate-FEVER; and
* FEVEROUS.

The datasets are not redistributed in this repository. Users must obtain each dataset from its official source and comply with its original license and access conditions.

Large processed corpora, retrieval indexes, downloaded model files, cached datasets, and prediction outputs are excluded from GitHub.

## Reproduction workflow

The exact execution sequence depends on the benchmark and experiment group. The repository preserves the experiment and analysis scripts used to generate:

* retrieval outputs;
* natural language inference scores;
* support-only and support-plus-contradiction ablations;
* risk-calibrated operating points;
* bootstrap confidence intervals;
* oracle-evidence diagnostics;
* structured-provenance summaries; and
* final evaluation summaries.

Before executing a script, inspect its dataset paths, output paths, model-cache locations, and GPU settings and adapt them to the local environment.

## Evaluation conventions

* Support-only and support-plus-contradiction rows are ungated, full-coverage ablations.
* Risk-gated rows are selective operating points.
* False-verification rate is the fraction of predicted verified claims whose reference label is not verified.
* Coverage is the proportion of examples receiving a non-abstaining decision.
* Accepted accuracy is computed only over non-abstained examples.
* Macro-F1 is computed over the original task labels, with abstentions treated as non-matching predictions rather than as a fourth task class.
* FEVER oracle experiments are upper-bound diagnostics using gold evidence identifiers and are not deployable settings.
* Provenance and rule-consistency gates are modular extensions. The main experiments empirically focus on support, contradiction, and calibrated risk.

## Hardware

The main natural language inference scoring experiments were executed on NVIDIA H100 GPUs with 80 GB of memory. Preprocessing and result analysis can generally be executed on a CPU, although model inference and large-scale retrieval will be slower.

