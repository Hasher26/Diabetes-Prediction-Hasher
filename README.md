# Diabetes Prediction ML

**Goethe-Universität Frankfurt · Advanced Applied Data Science · SS 2026**
**Prof. Dr. Kevin Bauer**

Supervised-Machine-Learning-Projekt zur binären Klassifikation von Diabetes-Risiko auf Basis selbstberichteter Gesundheits- und Lifestyle-Daten des CDC-BRFSS-2015-Datensatzes. Vorgehen nach CRISP-DM.

## Datensatz

- Quelle: [UCI ML Repository · CDC Diabetes Health Indicators (#891)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- ~253.680 Samples, 21 Features
- Target: `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes/Diabetes)
- Klassenverteilung: ~86 % / ~14 % (Prävalenz 0,1393)

Die Daten stammen aus dem **Behavioral Risk Factor Surveillance System** (BRFSS), einer Telefonumfrage des CDC. Das Target ist der selbstberichtete Diagnose-Status, nicht klinisch verifiziertes Krankheits-Vorliegen — die methodischen Implikationen (Label-Noise als Performance-Obergrenze) werden im Business Understanding diskutiert.

## CRISP-DM-Pipeline

Die Notebooks bauen strikt aufeinander auf und müssen in numerischer Reihenfolge ausgeführt werden. NB03 erzeugt die Train/Test-Splits, auf die alle folgenden Notebooks zugreifen.

| Phase | Notebook | Inhalt | Status |
|---|---|---|---|
| Business Understanding | `01_business_understanding.ipynb` | Ziel, Cost-Matrix, Metrik-Wahl, Erfolgsdefinition | ✓ |
| Data Understanding | `02_data_understanding.ipynb` | EDA, Target-Verteilung, Duplikat-/Label-Analyse | ✓ |
| Data Preparation | `03_data_preparation.ipynb` | Stratified 80/20-Split, Feature-Typen, Export nach `data/processed/` | ✓ |
| Feature Diagnostics | `04_feature_diagnostics.ipynb` | IV, MI, KS, Spearman, VIF → Feature-Shortlist | ✓ |
| Feature Engineering | `05_feature_engineering.ipynb` | Composite-Features, Interaktionen, nichtlineare Transformationen | ✓ |
| Modeling | `06_modeling.ipynb` | Modellvergleich, Imbalance-Strategien, Feature-Selektion, Top-3 | ✓ |
| Model Optimization | `07_model_optimization.ipynb` | Hyperparameter-Tuning (Optuna), Stacking, finales Modell | ✓ |
| Evaluation | `08_evaluation.ipynb` | Test-Eval, Threshold, Kalibrierung, Fairness, SHAP | in Arbeit |

**Kernergebnis (Stand NB07):** Bestes Modell ist eine getunte **LightGBM** mit CV-PR-AUC ≈ **0,436** (5-Fold). Stacking brachte keinen Mehrwert gegenüber dem Einzelmodell und wurde daher verworfen (Sparsamkeit). No-Skill-Baseline: 0,14.

## Methodische Eckpunkte

- **Primärmetrik PR-AUC** (wegen ~14 % Prävalenz); ROC-AUC nur sekundär, Accuracy bewusst verworfen.
- **Strikte Leakage-Disziplin:** Das Test-Set wird in NB03 eingefroren und erst in NB08 ein einziges Mal angefasst. Scaling, SMOTE und Feature-Selektion laufen ausschließlich innerhalb von CV-Pipelines (pro Fold neu gefittet).
- **Reproduzierbarkeit:** `SEED = 42` zentral, konsistent über alle Notebooks; Split-Artefakte als Parquet persistiert.
- **Use-Case-getriebene Threshold-Wahl** (Recall-Constraint statt Default 0,5), begründet aus der Cost-Matrix in NB01.

## Setup

Voraussetzung: Python 3.10+.

### Lokal

```bash
git clone https://github.com/<your-username>/diabetes-prediction-ml.git
cd diabetes-prediction-ml

python -m venv .venv
source .venv/bin/activate          # macOS/Linux
.venv\Scripts\activate             # Windows

pip install -r requirements.txt
jupyter lab
```

### Google Colab

```python
!git clone https://github.com/<your-username>/diabetes-prediction-ml.git
%cd diabetes-prediction-ml
!pip install -r requirements.txt
```

> Einige spezialisierte Pakete (`optbinning`, `catboost`, `optuna`) werden bei Bedarf zu Beginn der jeweiligen Notebooks automatisch nachinstalliert.

### Daten und Modelle regenerieren

`data/` und `models/` sind **nicht** im Repo (gitignored). Vor dem ersten Lauf:

1. NB01–NB03 ausführen — NB03 lädt den Datensatz über `ucimlrepo` und schreibt `X_train/X_test/y_train/y_test.parquet` sowie `feature_meta.json` nach `data/processed/`.
2. Anschließend NB04 → NB07 in Reihenfolge. Jedes Notebook lädt die Artefakte des vorherigen Schritts; ohne den NB03-Export brechen NB04+ beim Parquet-Laden ab.

## Projektstruktur

```
diabetes-prediction-ml/
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_feature_diagnostics.ipynb
│   ├── 05_feature_engineering.ipynb
│   ├── 06_modeling.ipynb
│   ├── 07_model_optimization.ipynb
│   └── 08_evaluation.ipynb          # in Arbeit
├── src/
│   └── utils.py                     # wiederverwendbare Logik (build_enriched_features u. a.)
├── data/                            # nicht in Git
│   ├── raw/
│   └── processed/                   # von NB03 erzeugt
├── models/                          # nicht in Git
│   └── transformers/
├── outputs/                         # Plots, CSVs, JSON pro Notebook
├── README.md
├── requirements.txt
└── .gitignore
```

`src/utils.py` ist der gemeinsam genutzte Code (u. a. `build_enriched_features`, `cap_bmi`, `categorize_bmi`, `hurdle_encode`) und wird per `sys.path` in mehrere Notebooks importiert.

## Team

- [Name 1]
- [Name 2]
- [Name 3]
- [Name 4]

## Git-Workflow

```bash
git pull origin main
git checkout -b feature/<beschreibung>
git add .
git commit -m "Aussagekräftige Nachricht"
git push origin feature/<beschreibung>
```