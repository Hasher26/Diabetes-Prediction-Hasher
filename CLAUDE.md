# Diabetes Prediction ML — Claude Context

## Projektkontext

- Seminar "Advanced Applied Data Science", Goethe Universität Frankfurt, SS 2026
- Professor: Prof. Dr. Kevin Bauer
- Vorgehen: CRISP-DM
- Sprache: Deutsch in Diskussion und Notebook-Markdown; Variablennamen und Code-Kommentare auf Englisch
- **Sauberer Neustart** — keine Ergebnisse, Hyperparameter, Seeds oder Performance-Anker aus früheren Versuchen werden übernommen. Performance-Ziele werden empirisch ermittelt, nicht vorab fixiert.

## Datensatz

- CDC BRFSS 2015 – Diabetes Health Indicators
- UCI ML Repository, Dataset #891
- URL: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators
- Genutzte Variante: **binary unbalanced** (~253.680 Samples, 21 Features)
- Target: `Diabetes_binary` (0 = No Diabetes, 1 = Prediabetes/Diabetes)
- Klassenverteilung: ~86 % / ~14 % → No-Skill-Baseline PR-AUC ≈ 0,14
- Keine fehlenden Werte laut UCI; in der EDA (NB02) verifiziert
- Feature-Typen: 14 binär, 4 ordinal (GenHlth, Age, Education, Income), 2 count (MentHlth, PhysHlth), 1 kontinuierlich (BMI)
- Datenzugriff: via `ucimlrepo`-Paket

**Wichtig zum Label:** Die BRFSS-Frage erfasst Diagnose-Status (Selbstauskunft über ärztliche Mitteilung), nicht Krankheits-Status. Daraus resultiert asymmetrisches Label-Noise auf der negativen Klasse (Undiagnostizierte erscheinen als 0). Die gemessene Precision ist damit eine Untergrenze. Siehe NB01 (Business Understanding) und die FP-Profilanalyse in NB08.

## Methodische Entscheidungen

Vollständige Argumentation in `notebooks/01_business_understanding.ipynb`. Hier die operativen Konsequenzen, wie im finalen Code umgesetzt:

- Output: `ŷ ∈ {0, 1}`
- Primärmetrik: **PR-AUC** (schwellenunabhängig, steuert die Modellauswahl)
- Sekundär: ROC-AUC, Brier, Confusion Matrix + Precision/Recall/F1/F2 am gewählten Threshold, Subgruppen-Performance
- Accuracy: bewusst nicht als Hauptmetrik (Imbalance verzerrt)
- Threshold: use-case-getrieben — größter Threshold mit OOF-Recall ≥ 0,80, auf OOF bestimmt, am Test nur angewendet (~0,138)
- Cross-Validation: **StratifiedGroupKFold** (group-aware + stratifiziert), zentral definiert in `src/utils.py::make_cv`
- **Class Imbalance: primär über Metrik (PR-AUC) + recall-getriebenen Threshold gelöst, NICHT über Resampling.** SMOTE / `class_weight` wurden im Funnel (NB06) evaluiert, haben das finale Modell nicht verbessert und wurden nicht übernommen (`imbalance: none`). Falls Resampling getestet wird, ausschließlich im Trainingsfold (imblearn Pipeline), nie auf gesamten Daten.
- Subgruppen-Auswertung: Geschlecht, Altersgruppen, Einkommen, Bildung

**Optionale Erweiterungen (umgesetzt in NB08):**
- Kalibrierung (Brier-Score, Reliability-Curve; Isotonic als Illustration geprüft, nicht übernommen — verschlechtert Brier nicht und senkt PR-AUC)
- Tier-/Threshold-Betrachtung und illustrative gruppenspezifische Schwellen (auf TRAIN-OOF bestimmt)
- False-Positive-Profilanalyse zur Stützung der Label-Noise-These

## Finaler Modellstand

- **Finales Modell: tuned LightGBM** (Primärmodell, deployt), LogReg-L2 als interpretierbarer Begleiter
- Feature-Set: `headline_pool` = raw 21 Features + `BMI_squared` (22 Spalten)
- Tuning: `RandomizedSearchCV` (LightGBM, MLP) + `GridSearchCV` (LogReg-C) — **kein Optuna, kein CatBoost**
- Stacking (MLP+LightGBM) als sekundärer, vorab registrierter Vergleich geprüft, aber nicht promotet (Gewinn über bestes Einzelmodell nicht über 1 SE)
- Test-Kennzahlen (einmalige Evaluation, NB08): PR-AUC 0,4330 (95% CI [0,4217, 0,4458]); ROC-AUC 0,8288; Brier 0,0968; Recall@T 0,789; Precision@T 0,306; OOF→Test-Gap −0,003
- Ehrlicher Rahmen: System ist **daten-, nicht modell-limitiert** — die Headline-PR-AUC ist durch Label-Noise gedeckelt, nicht durch Hyperparameter

## Notebook-Pipeline (CRISP-DM)

| Phase | Notebook(s) |
|---|---|
| Business Understanding | NB01 |
| Data Understanding | NB02 |
| Data Preparation | NB03–05 |
| Modelling | NB06–07 |
| Evaluation | NB08 |
| (NB00 = Intro/Overview) | — |

Alle Notebooks (00–08) sind ausgeführt. Reihenfolge muss eingehalten werden: jedes Notebook hängt an den Outputs der vorherigen.

## Projektstruktur

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
│   └── assets/                              # crispdm.png
├── src/
│   ├── features.py                          # row-wise, leakage-freies Feature-Engineering
│   ├── inference.py                         # standalone Scoring des finalen Modells
│   └── utils.py                             # make_cv (CV-Splitter) + log_result (Ledger)
├── literatur/                               # zitierte Paper (PDF)
├── data/                                    # nicht in Git; zur Laufzeit erzeugt
├── models/                                  # nicht in Git
├── outputs/                                 # nicht in Git; Plots/CSVs je Notebook
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Coding-Konventionen

- Python 3.10+
- Variablennamen und Code-Kommentare auf Englisch; Notebook-Markdown auf Deutsch
- Notebooks numerisch nummeriert: `00_`, `01_`, …
- Keine hardcoded absoluten Pfade — `PROJECT_ROOT` wird über das erste Parent mit `notebooks/` aufgelöst
- Reproduzierbarkeit: ein zentraler `SEED = 42`, konsistent verwendet; in NB03 in `feature_meta.json` persistiert und von NB04+ geladen
- Pipelines mit `sklearn.pipeline.Pipeline` (bzw. `imblearn.pipeline.Pipeline`, falls Resampling getestet wird)
- Train/Test-Split *vor* allem anderen; Preprocessing/Scaling ausschließlich in-fold — kein Leakage
- CV immer über `src.utils.make_cv` + denselben `groups`-Vektor (raw-21-Profil-Hash)

## Workflow-Hinweise für Claude

- **Vor jeder Code-Zelle:** kurz inhaltlich begründen WARUM, bevor das WIE geschrieben wird
- **Keine Anker** oder Vorab-Annahmen aus früheren Versuchen — alle Zahlen empirisch
- **Token-effizient** antworten (User-Präferenz)
- Bei medizinischen oder statistischen Aussagen: **Quellen** liefern, nicht aus dem Bauch
- Markdown-Cells sparsam formatieren — Prosa bevorzugt, keine übermäßige Bullet-/Tabellen-Lastigkeit
- Notebooks via Python-Skript bauen (sauberes ipynb-JSON), Markdown- und Code-Cells klar getrennt
- Vor jedem File-Create / Code: relevante SKILL.md unter `/mnt/skills/public/` lesen, falls verfügbar
