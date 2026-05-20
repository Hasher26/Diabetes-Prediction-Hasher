# Diabetes Prediction ML

**Goethe Universität Frankfurt · Advanced Applied Data Science · SS 2026**
**Prof. Dr. Kevin Bauer**

Supervised Machine Learning Projekt zur binären Klassifikation von Diabetes-Risiko auf Basis selbstberichteter Gesundheits- und Lifestyle-Daten des CDC BRFSS 2015 Datensatzes. Vorgehen nach CRISP-DM.

## Dataset

- Quelle: [UCI ML Repository · CDC Diabetes Health Indicators (#891)](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)
- ~253.680 Samples, 21 Features
- Target: `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes/Diabetes)
- Klassenverteilung: ~86 % / ~14 %

Die Daten basieren auf der **Behavioral Risk Factor Surveillance System** (BRFSS) Telefonumfrage des CDC. Das Target ist selbstberichteter Diagnose-Status, nicht klinisch verifiziertes Krankheits-Vorliegen — dies hat methodische Implikationen, die im Business Understanding diskutiert werden.

## CRISP-DM Phasen

| Phase | Notebook | Status |
|---|---|---|
| 1. Business Understanding | `notebooks/01_business_understanding.ipynb` | ✓ |
| 2. Data Understanding | `notebooks/02_data_understanding.ipynb` | in Arbeit |
| 3. Data Preparation | `notebooks/03_data_preparation.ipynb` | offen |
| 4. Modeling | `notebooks/04_modeling.ipynb` | offen |
| 5. Evaluation | `notebooks/05_evaluation.ipynb` | offen |
| 6. Deployment | `notebooks/06_deployment.ipynb` | offen |

## Setup

### Google Colab

```python
!git clone https://github.com/[USERNAME]/diabetes-prediction-ml.git
%cd diabetes-prediction-ml
!pip install -r requirements.txt
```

### Lokal

```bash
git clone https://github.com/[USERNAME]/diabetes-prediction-ml.git
cd diabetes-prediction-ml

python -m venv venv
source venv/bin/activate          # macOS/Linux
venv\Scripts\activate             # Windows

pip install -r requirements.txt
jupyter lab
```

## Projektstruktur

```
diabetes-prediction-ml/
├── notebooks/        # Jupyter Notebooks pro CRISP-DM Phase
├── src/              # wiederverwendbare Python-Module
├── data/             # Datensätze (nicht in Git)
├── results/          # Modelle und Plots (nicht in Git)
├── CLAUDE.md         # Kontext für Claude
├── README.md
├── requirements.txt
└── .gitignore
```

## Team

- [Name 1]
- [Name 2]
- [Name 3]
- [Name 4]

## Git Workflow

```bash
# aktuellen Stand holen
git pull origin main

# Branch für eigene Änderungen
git checkout -b feature/[beschreibung]

# committen
git add .
git commit -m "Aussagekräftige Nachricht"

# pushen
git push origin feature/[beschreibung]
```
