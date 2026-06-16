# Diabetes Prediction ML — Claude Context

## Project Context

- Seminar "Advanced Applied Data Science", Goethe University Frankfurt, SS 2026
- Professor: Prof. Dr. Kevin Bauer
- Methodology: CRISP-DM
- Language: German in discussion and notebook markdown; variable names and code comments in English
- **Clean slate** — no results, hyperparameters, seeds, or performance anchors from previous runs are carried over. Performance targets are determined empirically, not fixed in advance.

## Dataset

- CDC BRFSS 2015 – Diabetes Health Indicators
- UCI ML Repository, Dataset #891
- URL: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Variant used: **binary unbalanced** (~253,680 samples, 21 features)
- Target: `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes/Diabetes)
- Class distribution: ~86 % / ~14 % → no-skill baseline PR-AUC ≈ 0.14
- No missing values per UCI; verified in EDA (NB02)
- Feature types: 14 binary, 4 ordinal (GenHlth, Age, Education, Income), 2 count (MentHlth, PhysHlth), 1 continuous (BMI)
- Data access: via `ucimlrepo` package

**Important note on the label:** The BRFSS question captures diagnosis status (self-reported medical notification), not disease status. This results in asymmetric label noise on the negative class (undiagnosed individuals appear as 0). The measured precision is therefore a lower bound. See NB01 (Business Understanding) and the FP profile analysis in NB08.

## Methodological Decisions

Full reasoning in `notebooks/01_business_understanding.ipynb`. The operative consequences as implemented in the final code:

- Output: `ŷ ∈ {0, 1}`
- Primary metric: **PR-AUC** (threshold-independent, drives model selection)
- Secondary: ROC-AUC, Brier, Confusion Matrix + Precision/Recall/F1/F2 at the chosen threshold, subgroup performance
- Accuracy: deliberately not the primary metric (imbalance distorts it)
- Threshold: use-case-driven — largest threshold with OOF-Recall ≥ 0.80, determined on OOF, only applied to test (~0.138)
- Cross-validation: **StratifiedGroupKFold** (group-aware + stratified), centrally defined in `src/utils.py::make_cv`
- **Class imbalance: addressed primarily through the metric (PR-AUC) + recall-driven threshold, NOT through resampling.** SMOTE / `class_weight` were evaluated in the funnel (NB06), did not improve the final model, and were not adopted (`imbalance: none`). If resampling is tested, it must be applied exclusively within the training fold (imblearn Pipeline), never on the full dataset.
- Subgroup analysis: sex, age groups, income, education

**Optional extensions (implemented in NB08):**
- Calibration (Brier score, reliability curve; isotonic evaluated as illustration, not adopted — does not worsen Brier and lowers PR-AUC)
- Tiered/threshold analysis and illustrative group-specific thresholds (determined on TRAIN-OOF)
- False-positive profile analysis supporting the label-noise thesis

## Final Model State

- **Final model: tuned LightGBM** (primary model, deployed), LogReg-L2 retained as interpretable companion
- Feature set: `headline_pool` = raw 21 features + `BMI_squared` (22 columns)
- Tuning: `RandomizedSearchCV` (LightGBM, MLP) + `GridSearchCV` (LogReg-C) — **no Optuna, no CatBoost**
- Stacking (MLP+LightGBM) evaluated as a pre-registered secondary comparison, not promoted (gain over best single model not above 1 SE)
- Test metrics (single evaluation, NB08): PR-AUC 0.4330 (95% CI [0.4217, 0.4458]); ROC-AUC 0.8288; Brier 0.0968; Recall@T 0.789; Precision@T 0.306; OOF→Test gap −0.003
- Honest framing: the system is **data-limited, not model-limited** — the headline PR-AUC is capped by label noise, not by hyperparameters

## Notebook Pipeline (CRISP-DM)

| Phase | Notebook(s) |
|---|---|
| Business Understanding | NB01 |
| Data Understanding | NB02 |
| Data Preparation | NB03–05 |
| Modelling | NB06–07 |
| Evaluation | NB08 |
| Deployment | NB09 |
| (NB00 = Intro/Overview) | — |

All notebooks (00–09) have been executed. Order must be respected: each notebook depends on the outputs of the preceding ones.

## Project Structure

```
diabetes-prediction-ml/
├── notebooks/
│   ├── 00_introduction.ipynb
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_baseline.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modeling_funnel.ipynb
│   ├── 07_hyperparameter_tuning.ipynb
│   ├── 08_evaluation.ipynb
│   ├── 09_deployment.ipynb
│   └── assets/                              # crispdm.png
├── src/
│   ├── features.py                          # row-wise, leakage-free feature engineering
│   ├── inference.py                         # standalone scoring of the final model
│   └── utils.py                             # make_cv (CV splitter) + log_result (ledger)
├── literatur/                               # cited papers (PDF)
├── data/processed/                          # tracked; processed splits (parquet) + feature_meta.json
├── models/                                  # tracked; final LightGBM model + model_card.json
├── outputs/                                 # tracked; plots/CSVs per notebook + results.csv
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Coding Conventions

- Python 3.14+
- Variable names and code comments in English; notebook markdown in German
- Notebooks numbered sequentially: `00_`, `01_`, …
- No hardcoded absolute paths — `PROJECT_ROOT` is resolved via the first parent containing `notebooks/`
- Reproducibility: a single central `SEED = 42`, used consistently; persisted to `feature_meta.json` in NB03 and loaded by NB04+
- Pipelines via `sklearn.pipeline.Pipeline` (or `imblearn.pipeline.Pipeline` if resampling is tested)
- Train/test split *before* everything else; preprocessing/scaling exclusively in-fold — no leakage
- CV always via `src.utils.make_cv` + the same `groups` vector (raw-21 profile hash)

## Workflow Notes for Claude

- **Before each code cell:** briefly justify WHY before writing the HOW
- **No anchors** or prior assumptions from previous runs — all numbers empirically derived
- **Token-efficient** responses (user preference)
- For medical or statistical claims: **provide sources**, do not guess
- Format markdown cells sparingly — prose preferred, avoid excessive bullet/table density
- Build notebooks via Python script (clean ipynb JSON), markdown and code cells clearly separated
- Before any file creation / code: read the relevant SKILL.md under `/mnt/skills/public/` if available
