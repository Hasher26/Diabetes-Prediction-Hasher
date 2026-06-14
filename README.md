# Diabetes Risk Prediction from Health-Survey Data

**Goethe University Frankfurt · Advanced Applied Data Science · SS 2026 · Prof. Dr. Kevin Bauer**

Binary classification of diabetes risk from CDC BRFSS 2015 self-reported health-survey data
(~253,680 samples, 21 features), following the CRISP-DM process. PR-AUC is the primary metric,
chosen because the positive class makes up only ~14 % of the data; the operating threshold is
fixed by a recall constraint (recall ≥ 0.80) rather than the naive 0.5 cutoff, reflecting the
screening use case where missed cases are costlier than false alarms.

## Headline result

The final model is a tuned **LightGBM** classifier (with a LogReg-L2 model retained as an
interpretable companion). On the held-out test set, evaluated once:

| Metric | Test value | 95 % CI |
|---|---|---|
| PR-AUC | 0.4330 | [0.4217, 0.4458] |
| ROC-AUC | 0.8288 | [0.8238, 0.8336] |
| Brier (raw) | 0.0968 | — |
| Recall @ threshold 0.138 | 0.789 | [0.780, 0.799] |
| Precision @ threshold 0.138 | 0.306 | [0.299, 0.312] |

No-skill PR-AUC floor is the prevalence (0.139), so the model lifts PR-AUC ~3.1× over baseline.
The OOF→test PR-AUC gap is only −0.003, i.e. within sampling noise — the split generalises.

The honest framing of the project is that performance is **data-limited, not model-limited**:
the label records diagnosis status, not disease status, so undiagnosed positives sit in the
negative class as asymmetric label noise. The reported precision is therefore a lower bound, and
no amount of further tuning meaningfully moves the headline number.

## Method in brief

- **Leakage-aware pipeline.** The train/test split happens first; all preprocessing, scaling and
  any tested resampling live strictly inside the cross-validation folds; the test set is touched
  exactly once, in NB08.
- **Duplicate-aware splitting.** About 14 % of rows are exact duplicates — largely coincidental
  profile collisions from coarse survey coding, not data errors. Rows sharing a full 21-feature
  profile are kept entirely on one side of the partition via a group-based split (NB03) and
  `StratifiedGroupKFold` cross-validation (NB06–07), so no profile leaks across the boundary.
- **Imbalance handling.** Addressed primarily through the metric (PR-AUC) and a recall-driven
  threshold. SMOTE and `class_weight` were evaluated in the modelling funnel but did not improve
  the final model and were not adopted.
- **Diagnostics.** Bootstrap confidence intervals on all test metrics, per-subgroup performance
  (sex, age, income, education), calibration (Brier + reliability curve), SHAP global importances,
  LogReg odds ratios, and a false-positive vs. true-positive profile analysis supporting the
  label-noise thesis.

## Repository layout

```
diabetes-prediction-ml/
├── notebooks/          # one notebook per CRISP-DM step (00–08)
│   └── assets/         # figures embedded in notebooks (e.g. crispdm.png)
├── literatur/          # cited papers (PDF)
├── src/
│   ├── features.py     # row-wise, leakage-free feature engineering (NB05+)
│   ├── inference.py    # standalone scoring of the final model (NB08 hand-off)
│   └── utils.py        # shared CV splitter + results ledger
├── data/               # git-ignored; generated at runtime
├── models/             # git-ignored; written during modelling
├── outputs/            # git-ignored; plots/CSVs per notebook
├── CLAUDE.md
├── requirements.txt
├── README.md
└── .gitignore
```

`data/`, `models/`, and `outputs/` are git-ignored and generated at runtime.

## Reproducing the results

```bash
pip install -r requirements.txt
```

Python 3.10 or later. The raw data is **not** stored in the repository; it is fetched at runtime
via the `ucimlrepo` package (UCI dataset #891), so the first run needs an internet connection.
Every computational notebook fixes `SEED = 42` and passes it to all stochastic steps. Run the
numbered notebooks in order from the repository root — each depends on the outputs of the ones
before it.

A full overview (methodology, team contributions, CRISP-DM mapping) is in
[`notebooks/00_introduction.ipynb`](notebooks/00_introduction.ipynb). The written report is
submitted separately as the project thesis.

## Team

Hasher Malik (7632048) · Jan Erdorf (8748557) · Ilias El Ouali (7632585) · Sophia Schaal (7229428)
