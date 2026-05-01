# psc-final-project (DATASCI 347 Final Project)

This repo contains my DATASCI 347 final project using the UK Companies House PSC (People with Significant Control) relationship data.

The dataset is an edge table where each row is a controller → company control relationship.

## Project overview

I build two ML tasks from the same PSC edge table (`rels2.csv`):

1) Task 1 (binary classification): predict whether a control relationship ends (`ceased_on` exists).
2) Task 2 (duration bins classification): for ended relationships only, compute duration = `ceased_on - notified_on`, then predict whether duration is short / medium / long.

These two tasks are designed to be fully reproducible from the notebooks.

## Files in this repo

- `01_cease_prediction.ipynb`  
- `02_duration_bins.ipynb`  
- `requirements.txt`  
- `neo4j_import_optional/` (optional early exploration; not required to run ML tasks)

## Data

The dataset is not included in this repo.

You need a local copy of:

- `rels2.csv`

In both notebooks, update the path variable:

- `rels_path = ...`

to point to your local `rels2.csv` location.

Key fields used from `rels2.csv`:
- `natures_of_control` (control type)
- `notified_on` (relationship start/registration date)
- `ceased_on` (relationship end date, if ended)

## Task 1: Cease prediction (binary classification)

Notebook:
- `01_cease_prediction.ipynb`

Label:
- y = 1 if `ceased_on` is present
- y = 0 otherwise

Features:
- `nature_primary` extracted from `natures_of_control` (one-hot)
- `notified_year`, `notified_month` from `notified_on`

Models:
- Logistic Regression (class_weight="balanced")
- Random Forest (class_weight="balanced")

Metrics / outputs:
- ROC-AUC, PR-AUC
- F1 (including threshold tuning)
- Figures: PR curve, Threshold–F1 curve
- Tables saved under `results/tables/` and figures under `results/figures/`

## Task 2: Duration bins (short / medium / long)

Notebook:
- `02_duration_bins.ipynb`

Subset:
- Use only ended relationships (non-missing `ceased_on` and `notified_on`)
- duration_days = `ceased_on - notified_on` (in days)

Bins:
- short: ≤ 180 days
- medium: 180–730 days
- long: > 730 days

Models:
- Logistic Regression (class_weight="balanced")
- Random Forest (class_weight="balanced_subsample")

Metrics / outputs:
- Accuracy, Macro F1
- Confusion matrix
- Tables saved under `results/tables/` and figures under `results/figures/`

## How to run (reproducibility)

1) Install dependencies:
```bash
pip install -r requirements.txt
