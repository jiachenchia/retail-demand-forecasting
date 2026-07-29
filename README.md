# Sales Forecasting Project

> **End-to-end retail demand forecasting** built during an ML internship: integrating sales, store attributes, public holidays, and Open-Meteo weather into **store-level 3-week forecasts** using a multi-input LSTM with categorical embeddings, benchmarked against XGBoost, LightGBM, and feed-forward networks.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-lightgrey)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-brightgreen)
![LightGBM](https://img.shields.io/badge/LightGBM-4.x-yellowgreen)

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [What I Built & Learned](#what-i-built--learned)
4. [Techniques & Tooling](#techniques--tooling)
5. [The Modelling Pipeline](#the-modelling-pipeline)
6. [Key Decisions & Lessons](#key-decisions--lessons)
7. [Reproducibility & Data Access](#reproducibility--data-access)

---

## Overview

The goal of this project was to forecast **daily net sales (RM)** and **transaction counts (TC)** at the individual store level, using a blend of internal store data and external signals (weather, public holidays). The work spans the full pipeline: scraping and engineering an external dataset, exploratory studies on simpler models, and a flagship multi-input LSTM that delivers store-level forecasts with an interactive dashboard.

The repository is organised so that each folder maps to one stage of that pipeline — see [Repository Structure](#repository-structure) below.

> **A note on data:** this work was completed during an internship and is published with permission. Raw sales and store data are **not** included, and internal category labels have been replaced with generic names (e.g. `Segment_1`). File paths in notebooks are placeholders so reviewers can audit the engineering logic without the underlying data.

---

## Repository Structure

The repo is split into four folders, each representing a stage of the workflow. Within each folder, notebooks are named to describe exactly what they do.

```
Sales-Forecasting-Project/
├── data/            # Stage 1 — building the modelling dataset
├── notebooks/       # Stage 2 — exploratory studies + the flagship LSTM
├── experiments/     # Stage 3 — hyperparameter tuning & architecture variants
├── model/           # The final trained model, ready for inference
├── README.md
└── requirements.txt
```

### `data/` — Dataset construction

Notebooks that scrape, clean, and merge the raw sources into a single modelling table. Naming convention: `<action> <subject>` — each notebook performs one construction step. Run them roughly in the order below.

| Notebook | What it does |
| --- | --- |
| `Scrape Weather Data.ipynb` | Pulls **hourly** weather per store location from the **Open-Meteo archive API**, iterating over every store's latitude/longitude. Uses a cached, retrying HTTP session; maps WMO weather codes (0–99) to readable descriptions; outputs a timezone-aware (Asia/Singapore) hourly weather table. |
| `Create Public Holiday Calendar.ipynb` | Builds the Malaysian public-holiday calendar. Melts holidays to one row per state, splits fixed vs expandable holidays, and expands a **±7-day window** around each holiday so the model can learn pre/post-holiday effects. Adds `Public Holiday` and `Days From Holiday` flags. |
| `Create Master Merged Dataset.ipynb` | An **earlier** merge approach (weather → holidays → sales ordering), superseded by `Build LSTM Dataset` below. Kept to show the data-integration process. Applies **lifecycle filtering** (removes closed, not-yet-open, and unnamed stores) before joining sources into one master table keyed by store and date. |
| `Build LSTM Dataset.ipynb` | **The final dataset used for the flagship model.** Merges sales → weather → holidays into the flat table consumed by the sequence-prep step; adds `Day`/`Month` fields and per-store day counters. |
| `Prepare Time Series Data.ipynb` | Reshapes the merged table into sequence-model format: keeps holiday-window rows, categorises holidays (religious, regional, leaders' birthdays, etc.), de-duplicates overlapping holidays, and regroups to one row per (store, date). |

### `notebooks/` — Studies & flagship model

The core analysis. Naming convention: exploratory studies are `<Daily/Hourly> Sales Weather Study`; the flagship model and its inference companion are named directly.

| Notebook | What it does |
| --- | --- |
| `Hourly Sales Weather Study.ipynb` | Hour-by-hour study. Expands each store into its operating hours, joins hourly sales and weather, runs a correlation analysis, then benchmarks **Linear / Ridge / Lasso / ElasticNet** and **XGBoost** (with GridSearch tuning and feature-importance checks) on hourly net amount and transaction count. |
| `Daily Sales Weather Study.ipynb` | Daily-level counterpart. Collapses hourly sales to daily totals, applies **SMOTE** to address class imbalance, merges with weather, and benchmarks **XGBoost** (draft + GridSearch) and an **MLP** (scikit-learn and TensorFlow versions) for daily net amount and TC. |
| `Daily Sales Supervised ML.ipynb` | The fuller supervised sweep on the de-duplicated daily data. Runs **Polynomial Regression**, **XGBoost**, **LightGBM** (four progressive feature-drop drafts), and a **Feed-Forward NN with entity embeddings** and two task-specific heads — comparing MAE/RMSE/R² across all. |
| `LSTM Final Model.ipynb` | **Flagship.** Multi-input LSTM: categorical **embeddings** + a static-context **MLP** broadcast across timesteps, on a 14-day window with a time-aware split. Trains on both targets, reports MAE/RMSE/R², produces 200-step autoregressive forecasts with **95% confidence bands**, and saves the model to `model/`. |
| `Running LSTM Without Retraining.ipynb` | **Inference only.** Loads `model/Sales_Forecasting_LSTM_Model_Final.h5` and generates forecasts without retraining — the "handoff" notebook showing how the saved model is used in practice. |

### `experiments/` — Tuning & variants

A documented log of what was tried while developing the LSTM — including approaches that **did not** beat the tuned baseline (kept for rigor and transparency). Naming convention: `LSTM - <what was tested>`.

| Notebook | What it tested |
| --- | --- |
| `LSTM - Optuna HPO run 1.ipynb` | First Optuna hyperparameter search (20 trials) |
| `LSTM - Optuna HPO run 2.ipynb` | Second search (re-run, as the first was slow) |
| `LSTM - eval with Optuna 1 best params.ipynb` | Evaluating the model on run-1's best parameters |
| `LSTM - eval with Optuna 2 best params.ipynb` | Evaluating the model on run-2's best parameters |
| `LSTM - all features as input, no MLP.ipynb` | Feeding all variables as sequence input, without the static-context MLP |
| `LSTM - all features + MLP with Optuna 2 params.ipynb` | Best architecture (all features + MLP) using Optuna 2 parameters |
| `LSTM - drop store and segment features.ipynb` | Permutation-importance-guided feature removal |
| `LSTM - tiled static embeddings + init state.ipynb` | Tiling static embeddings across timesteps + initialising LSTM state from static features |

### `model/`

| File | Description |
| --- | --- |
| `Sales_Forecasting_LSTM_Model_Final.h5` | The trained Keras LSTM, loadable for inference (see `Running LSTM Without Retraining.ipynb`) |

---

## What I Built & Learned

- **Data integration & cleaning** — Built an ML-ready table by merging sales, store data, public holidays, and hourly weather; enforced dtypes, handled outliers (closed / unopened / unnamed stores), and standardised schemas with pandas.
- **Reproducible external data** — Implemented an Open-Meteo pipeline with on-disk caching and retries; mapped WMO weather codes to readable labels; generated timezone-aware per-store hourly series.
- **Exploration (daily & hourly)** — Quantified sales–weather relationships, handled class imbalance, and concluded that **weather alone is insufficient** to explain sales variance.
- **Supervised baselines & tuning** — Trained Polynomial/Linear models, XGBoost, LightGBM, and a Feed-Forward NN; used `GridSearchCV`, regularisation (`ReduceLROnPlateau`, `EarlyStopping`), and importance-guided feature pruning.
- **Sequence feature design** — Partitioned features into time-varying categoricals, static categoricals (entity embeddings), and continuous; applied `StandardScaler`; engineered a 14-day input window with a time-aware train/validation split.
- **LSTM forecasting (flagship)** — LSTM + 2-layer MLP for static context; 200-step autoregressive forecasts; tracked MAE/RMSE/R²; delivered store-level 3-week forecasts with interactive Plotly + ipywidgets dashboards.
- **Experiment discipline** — Optuna HPO, categorical-as-numeric trials, importance-guided column removals, tiled static embeddings, and static-initialised LSTM state — documented when they **did not** beat the tuned baseline.
- **Inference & handoff** — Saved a deployable Keras model and an inference-only notebook to generate forecasts without retraining.

---

## Techniques & Tooling

- **ML / DL**: LSTM (TensorFlow/Keras), Feed-Forward NN, XGBoost, LightGBM, Linear/Ridge/Lasso/ElasticNet, Permutation Feature Importance
- **Feature engineering**: categorical embeddings, one-hot weather codes, static vs time-varying partitions, binary event flags, sliding-window sequences
- **Training**: `EarlyStopping`, `ReduceLROnPlateau`, `GridSearchCV`, Optuna trials
- **Data**: pandas pipelines, dtype enforcement, outlier filtering, timezone conversion
- **Visualisation & UX**: Plotly charts, interactive ipywidgets dashboard

---

## The Modelling Pipeline

The flagship LSTM uses a hybrid input design:

- **Time-varying categorical** (`Name`, `Day`, `Month`) — encoded and embedded where cardinality is high
- **Static categorical** (`Store_No`, `State`, segment features) — embedded, then broadcast across timesteps via a residual MLP
- **Binary flags** (`Rain?`, `Puasa`, `Public Holiday`) — mapped to 0/1
- **Numeric** (`Net_Amount`, `TC`, `Days_after_Opening`, `Average Daily Temperature`) — min-max scaled

A **time-aware split per store** avoids leakage across dates. The model outputs both `Net_Amount` and `TC`, tracked with MAE / RMSE / R², and supports a 200-step autoregressive forecast with an optional 21-day per-store interactive dashboard.

---

## Key Decisions & Lessons

- **Standalone predictors seldom dominate the signal** — robust performance depended on multi-modal inputs; weather alone was not enough.
- **Use time-aware splits per store** for time-series modelling; random splits inflate scores via leakage.
- **Compact, focused hyperparameter searches** with learning-rate scheduling and early stopping reached good results faster than broad Optuna sweeps.
- **Permutation Feature Importance should steer selective feature removal**, not broad elimination.
- **Treat categoricals as embeddings (static) and flags (time-varying)**; keep numerics standardised; avoid aggregates computed beyond the prediction cutoff.
- **Communicate uncertainty** — 200-step autoregression shows error growth, so shorter horizons or prediction bands are preferable for decisions.
- **Simple, interactive visualisations** (store-level 3-week outlook, errors, importance) drove stakeholder adoption more than raw metrics.

---

## Reproducibility & Data Access

- **Public & reproducible**: the full weather-scraping pipeline (Open-Meteo) and all preprocessing logic are included.
- **Private data**: raw sales and store data are excluded; notebooks retain code with placeholder paths so reviewers can audit the engineering steps. Internal category labels are replaced with generic names. Some feature-importance plots were removed where their axis labels contained confidential category names — a note marks each removal.
- **Environment**: dependencies are pinned in [`requirements.txt`](requirements.txt).

```bash
pip install -r requirements.txt
```

---

*Built by [@jiachenchia](https://github.com/jiachenchia) · UCL BSc Mathematics & Statistical Science.*
