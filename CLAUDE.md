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
- Keine fehlenden Werte laut UCI; trotzdem in EDA verifizieren
- Feature-Typen: 14 binär, 4 ordinal (GenHlth, Age, Education, Income), 3 numerisch (BMI, MentHlth, PhysHlth)
- Datenzugriff: via `ucimlrepo`-Paket

**Wichtig zum Label:** BRFSS-Frage erfasst Diagnose-Status (Selbstauskunft über ärztliche Mitteilung), nicht Krankheits-Status. Daraus resultiert asymmetrisches Label-Noise auf der negativen Klasse. Siehe Section 4 in `notebooks/01_business_understanding.ipynb`.

## Methodische Entscheidungen

Vollständige Argumentation in `notebooks/01_business_understanding.ipynb`. Hier nur die operativen Konsequenzen:

**Pflicht-Teil (binäre Klassifikation):**
- Output: `ŷ ∈ {0, 1}`
- Primärmetrik: **PR-AUC** (schwellenunabhängig, Modellauswahl)
- Sekundär: ROC-AUC, Confusion Matrix + Precision/Recall/F1 am gewählten Threshold, Subgruppen-Performance
- Accuracy: nicht als Hauptmetrik (Imbalance verzerrt)
- Threshold: use-case-getrieben (Recall-Constraint, z. B. Recall ≥ 0,80), nicht naiv 0,5
- Cross-Validation: Stratified K-Fold
- Class Imbalance: SMOTE oder `class_weight` — ausschließlich im Trainingsfold (imblearn Pipeline), nie auf gesamten Daten
- Subgruppen-Auswertung: über Geschlecht, Altersgruppen, Einkommen

**Optionale Erweiterungen** (Section 10 im Business Understanding):
- 10.1 Kalibrierung (Brier-Score, Calibration Plot, ggf. Platt/Isotonic)
- 10.2 Tier-basiertes Deployment (zwei Thresholds, dreistufige Empfehlung)
- 10.3 False-Positive-Profilanalyse zur Label-Noise-Stützung

## Stand des Projekts

**Abgeschlossen:**
- `notebooks/01_business_understanding.ipynb` — CRISP-DM Phase 1 (Business Understanding)

**Als Nächstes:**
- `notebooks/02_data_understanding.ipynb` — CRISP-DM Phase 2 (Data Understanding / EDA)
  1. Datensatz laden, Grundstruktur, Datentypen
  2. Target-Verteilung + No-Skill-Baselines
  3. Duplikat- und Label-Konsistenz-Analyse (empirisches irreduzibles Rauschen)
  4. Univariate Verteilungen, Outlier-Inspektion (besonders BMI, MentHlth, PhysHlth)
  5. Bivariate Analyse: Target-Prävalenz pro Feature-Ausprägung
  6. Korrelationen unter Features
  7. Class-Overlap (PCA/UMAP)

## Projektstruktur

```
diabetes-prediction-ml/
├── notebooks/
│   ├── 01_business_understanding.ipynb     # ✓ abgeschlossen
│   ├── 02_data_understanding.ipynb         # nächster Schritt
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment.ipynb
├── src/                                    # wiederverwendbare Python-Module
├── data/                                   # nicht in Git
│   ├── raw/
│   └── processed/
├── results/                                # nicht in Git
│   ├── models/
│   └── plots/
├── CLAUDE.md
├── README.md
├── requirements.txt
└── .gitignore
```

## Coding-Konventionen

- Python 3.10+
- Variablennamen und Code-Kommentare auf Englisch
- Notebook-Markdown auf Deutsch (Diskussion, Begründung, Interpretation)
- Notebooks numerisch nummeriert: `01_`, `02_`, …
- Keine hardcoded absoluten Pfade — nur relative Pfade vom Repo-Root
- Reproduzierbarkeit: einen Seed an einer zentralen Stelle festlegen und konsistent verwenden (Konvention: am Notebook-Anfang als `SEED = …`)
- Pipelines mit `sklearn.pipeline.Pipeline` oder `imblearn.pipeline.Pipeline` (für SMOTE)
- Train/Val/Test-Split *vor* allem anderen
- Preprocessing/SMOTE/Scaling ausschließlich innerhalb der Pipeline auf Trainingsfolds — kein Leakage

## Workflow-Hinweise für Claude

- **Vor jeder Code-Zelle:** kurz inhaltlich begründen WARUM, bevor das WIE geschrieben wird
- **Keine Anker** oder Vorab-Annahmen aus früheren Versuchen — alle Zahlen empirisch
- **Token-effizient** antworten (User-Präferenz)
- Bei medizinischen oder statistischen Aussagen: **Quellen** liefern, nicht aus dem Bauch
- Markdown-Cells sparsam formatieren — Prosa bevorzugt, keine übermäßige Bullet-/Tabellen-Lastigkeit
- Notebooks via Python-Skript bauen (sauberes ipynb-JSON), Markdown-Cells für Erklärungen, Code-Cells klar getrennt
- Vor jedem File-Create / Code: relevante SKILL.md unter `/mnt/skills/public/` lesen, falls verfügbar
