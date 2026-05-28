"""Generates notebooks/05_feature_engineering.ipynb"""
import json, uuid, pathlib

def uid(): return str(uuid.uuid4())

def md(*lines):
    src = "\n".join(lines)
    return {"cell_type": "markdown", "id": uid(), "metadata": {}, "source": [src]}

def code(*lines):
    src = "\n".join(lines)
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": uid(),
        "metadata": {},
        "outputs": [],
        "source": [src],
    }

cells = []

# ─── Titel ────────────────────────────────────────────────────────────────────
cells.append(md(
    "# 05 Feature Engineering",
    "",
    "In CRISP-DM Phase 3b werden auf Basis der NB04-Diagnostik neue Features konstruiert,",
    "die das Modell in NB06 nutzen kann. Ziel ist eine Anreicherung des Feature-Sets mit",
    "klinisch begründeten Kombinationsmerkmalen, Interaktionstermen und nichtlinearen",
    "Transformationen — ohne Datenleck in das Test-Set.",
    "",
    "**Eiserne Regeln:**",
    "- Test-Set bleibt tabu bis NB09.",
    "- Kein Resampling in diesem Notebook.",
    "- Transformer werden ausschließlich auf `X_train` gefittet.",
    "",
    "**Struktur:**",
    "1. Setup",
    "2. Daten laden",
    "3. Features droppen + Transformationen anwenden",
    "4. Composite Features",
    "5. Interaktionsterme",
    "6. Polynomterme und nichtlineare Transformationen",
    "7. IV/MI Vorher-Nachher-Vergleich",
    "8. Export",
    "9. Zusammenfassung",
))

# ─── 1. Setup ─────────────────────────────────────────────────────────────────
cells.append(md(
    "## 1. Setup",
    "",
    "Alle Bibliotheken werden zentral importiert. `optbinning` wird für den IV-Vergleich",
    "in Abschnitt 7 benötigt. Der globale Seed `SEED = 42` wird konsistent aus NB03 übernommen.",
))

cells.append(code(
    "import subprocess, sys, importlib",
    "",
    "for pkg in ['optbinning']:",
    "    if importlib.util.find_spec(pkg) is None:",
    "        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])",
    "",
    "import warnings",
    "warnings.filterwarnings('ignore')",
    "",
    "import numpy as np",
    "import pandas as pd",
    "import json",
    "from pathlib import Path",
    "import matplotlib.pyplot as plt",
    "import seaborn as sns",
    "from sklearn.feature_selection import mutual_info_classif",
    "from sklearn.preprocessing import PowerTransformer",
    "import joblib",
    "from optbinning import OptimalBinning",
    "",
    "SEED = 42",
    "np.random.seed(SEED)",
    "",
    "PROCESSED_DIR = Path('../data/processed')",
    "MODELS_DIR    = Path('../models/transformers')",
    "OUTPUTS_DIR   = Path('../outputs/05_feature_engineering')",
    "for d in [MODELS_DIR, OUTPUTS_DIR]:",
    "    d.mkdir(parents=True, exist_ok=True)",
    "",
    "print('Setup abgeschlossen.')",
    "print(f'  PROCESSED_DIR : {PROCESSED_DIR.resolve()}')",
    "print(f'  MODELS_DIR    : {MODELS_DIR.resolve()}')",
    "print(f'  OUTPUTS_DIR   : {OUTPUTS_DIR.resolve()}')",
))

# ─── 2. Daten laden ───────────────────────────────────────────────────────────
cells.append(md(
    "## 2. Daten laden",
    "",
    "Die in NB03 erzeugten Parquet-Dateien werden direkt geladen. Der Split ist bereits",
    "fixiert — kein neuerlicher `train_test_split`. `y_test` wird nur der Vollständigkeit",
    "halber geladen, aber bis NB09 nicht ausgewertet.",
))

cells.append(code(
    "X_train = pd.read_parquet(PROCESSED_DIR / 'X_train.parquet')",
    "X_test  = pd.read_parquet(PROCESSED_DIR / 'X_test.parquet')",
    "y_train = pd.read_parquet(PROCESSED_DIR / 'y_train.parquet').squeeze()",
    "y_test  = pd.read_parquet(PROCESSED_DIR / 'y_test.parquet').squeeze()",
    "",
    "with open(PROCESSED_DIR / 'feature_meta.json') as f:",
    "    meta = json.load(f)",
    "",
    "BINARY_COLS  = meta['binary_cols']",
    "ORDINAL_COLS = meta['ordinal_cols']",
    "COUNT_COLS   = meta['count_cols']",
    "NUMERIC_COLS = meta['numeric_cols']",
    "",
    "print(f'X_train : {X_train.shape}')",
    "print(f'X_test  : {X_test.shape}')",
    "print(f'y_train Prävalenz: {y_train.mean():.4f}')",
    "print(f'y_test  Prävalenz: {y_test.mean():.4f}')",
    "print(f'\\nFeature-Gruppen:')",
    "print(f'  BINARY  : {len(BINARY_COLS)}')",
    "print(f'  ORDINAL : {len(ORDINAL_COLS)}')",
    "print(f'  COUNT   : {len(COUNT_COLS)}')",
    "print(f'  NUMERIC : {len(NUMERIC_COLS)}')",
))

# ─── Hilfsfunktion IV/MI ──────────────────────────────────────────────────────
cells.append(code(
    "# Hilfsfunktion: IV und MI für ein einzelnes Feature berechnen",
    "def compute_iv(feat_series, y_series, feature_name, is_continuous=False):",
    "    \"\"\"IV via OptimalBinning, MI via sklearn.\"\"\"",
    "    X_arr = feat_series.values.astype(float)",
    "    y_arr = y_series.values.astype(int)",
    "    try:",
    "        ob = OptimalBinning(name=feature_name, dtype='numerical',",
    "                            solver='cp', max_n_bins=10)",
    "        ob.fit(X_arr, y_arr)",
    "        iv = ob.iv",
    "    except Exception:",
    "        iv = np.nan",
    "    return iv",
    "",
    "def compute_mi(feat_series, y_series, is_discrete=True):",
    "    X_arr = feat_series.values.reshape(-1, 1)",
    "    mi = mutual_info_classif(",
    "        X_arr, y_series.values,",
    "        discrete_features=[is_discrete],",
    "        random_state=SEED,",
    "    )[0]",
    "    return round(mi, 4)",
    "",
    "# NB04-Referenzwerte (aus outputs/04_diagnostics/feature_diagnostics.csv)",
    "NB04_IV = {",
    "    'GenHlth': 0.7905, 'HighBP': 0.6095, 'BMI': 0.4694, 'Age': 0.3970,",
    "    'HighChol': 0.3389, 'DiffWalk': 0.3170, 'Income': 0.2281, 'PhysHlth': 0.2194,",
    "    'HeartDiseaseorAttack': 0.1939, 'Education': 0.1226, 'PhysActivity': 0.1041,",
    "    'MentHlth': 0.0402, 'HvyAlcoholConsump': 0.0379, 'Smoker': 0.0312,",
    "    'Veggies': 0.0234, 'Fruits': 0.0154, 'Stroke': 0.0, 'CholCheck': 0.0,",
    "}",
    "NB04_MI = {",
    "    'GenHlth': 0.0441, 'HighBP': 0.0351, 'BMI': 0.0285, 'Age': 0.0203,",
    "    'HighChol': 0.0200, 'DiffWalk': 0.0202, 'Income': 0.0136, 'PhysHlth': 0.0141,",
    "    'HeartDiseaseorAttack': 0.0126, 'Education': 0.0075, 'PhysActivity': 0.0064,",
    "    'MentHlth': 0.0026, 'HvyAlcoholConsump': 0.0020, 'Smoker': 0.0019,",
    "    'Veggies': 0.0014, 'Fruits': 0.0009, 'Stroke': 0.0045, 'CholCheck': 0.0031,",
    "}",
    "print('Referenzwerte geladen.')",
))

# ─── 3. Drop + Transformationen ───────────────────────────────────────────────
cells.append(md(
    "## 3. Features droppen + Transformationen anwenden",
    "",
    "### 3.1 Ausschluss diagnostisch schwacher Features",
    "",
    "NB04 dokumentiert für die folgenden drei Features Unterschreitung aller vier",
    "Selektionsschwellen (IV, MI, KS, Spearman) oder reinen Selektionsbias:",
    "",
    "| Feature | IV | MI | in_shortlist | Begründung |",
    "|---|---|---|---|---|",
    "| `AnyHealthcare` | 0.000 | 0.0001 | False | Nahezu konstant (>95 % positiv), kein Informationsgehalt |",
    "| `NoDocbcCost` | 0.008 | 0.0005 | False | Alle Metriken unter Schwelle |",
    "| `Sex` | 0.008 | 0.0005 | False | Alle Metriken unter Schwelle |",
    "",
    "Diese Features werden direkt per `drop` entfernt — keine separate Funktion erforderlich.",
))

cells.append(code(
    "DROP_COLS = ['AnyHealthcare', 'NoDocbcCost', 'Sex']",
    "",
    "X_train = X_train.drop(columns=DROP_COLS)",
    "X_test  = X_test.drop(columns=DROP_COLS)",
    "",
    "print(f'Spalten nach Drop: {X_train.shape[1]}  (vorher: 21)')",
    "print(f'Entfernte Spalten : {DROP_COLS}')",
    "assert X_train.shape[1] == 18, 'Unerwartete Spaltenanzahl nach Drop!'",
    "assert not any(c in X_train.columns for c in DROP_COLS), 'Drop unvollständig!'",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 3.2 BMI-Capping → `BMI_capped`",
    "",
    "**Formel:** `BMI_capped = clip(BMI, 18, 50)`",
    "",
    "Capping auf [18, 50] orientiert sich an klinischen WHO-Schwellen (BMI < 18,5 = Untergewicht;",
    "BMI ≥ 50 = Super-Adipositas) und den P1/P99-Perzentilen aus NB02.",
    "Der Originalwert `BMI` bleibt erhalten, da er als Basis für nachfolgende Transformationen",
    "(BMI_squared, BMI_yj in Abschnitt 6) benötigt wird.",
    "",
    "**Quelle:** WHO Expert Consultation 2004, *Lancet*, 363(9403):157–163.",
))

cells.append(code(
    "def cap_bmi(X, lower=18, upper=50):",
    "    X = X.copy()",
    "    X['BMI_capped'] = X['BMI'].clip(lower=lower, upper=upper)",
    "    return X",
    "",
    "X_train = cap_bmi(X_train)",
    "X_test  = cap_bmi(X_test)",
    "",
    "n_affected_train = ((X_train['BMI'] < 18) | (X_train['BMI'] > 50)).sum()",
    "print(f'BMI_capped Bereich: [{X_train[\"BMI_capped\"].min():.1f}, {X_train[\"BMI_capped\"].max():.1f}]')",
    "print(f'Gecappte Samples   : {n_affected_train} ({n_affected_train/len(X_train):.2%})')",
    "print(f'NaN in BMI_capped  : {X_train[\"BMI_capped\"].isna().sum()}')",
    "assert X_train['BMI_capped'].between(18, 50).all(), 'BMI_capped außerhalb [18, 50]!'",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 3.3 BMI-Kategorisierung → `BMI_cat`",
    "",
    "**Formel:** Sieben WHO-Klassen nach Binning auf `BMI` (Original, nicht gecappt).",
    "",
    "| Klasse | Label | BMI-Bereich |",
    "|---|---|---|",
    "| 0 | Untergewicht | < 18.5 |",
    "| 1 | Normalgewicht | 18.5 – 25 |",
    "| 2 | Übergewicht | 25 – 30 |",
    "| 3 | Adipositas I | 30 – 35 |",
    "| 4 | Adipositas II | 35 – 40 |",
    "| 5 | Adipositas III | 40 – 50 |",
    "| 6 | Super-Adipositas | > 50 |",
    "",
    "Das ordinale `BMI_cat` ist für baumbasierte Modelle geeignet ohne Skalierung",
    "und wird in Abschnitt 4 (Composite Features) verwendet.",
    "",
    "**Quelle:** WHO Expert Consultation 2004, *Lancet*, 363(9403):157–163.",
))

cells.append(code(
    "def categorize_bmi(X):",
    "    X = X.copy()",
    "    bins   = [0, 18.5, 25, 30, 35, 40, 50, 999]",
    "    labels = [0, 1, 2, 3, 4, 5, 6]",
    "    X['BMI_cat'] = pd.cut(X['BMI'], bins=bins, labels=labels).astype(int)",
    "    return X",
    "",
    "X_train = categorize_bmi(X_train)",
    "X_test  = categorize_bmi(X_test)",
    "",
    "print('BMI_cat Häufigkeiten (X_train):')",
    "cat_labels = {0:'Untergewicht', 1:'Normalgewicht', 2:'Übergewicht',",
    "              3:'Adipositas I', 4:'Adipositas II', 5:'Adipositas III', 6:'Super'}",
    "counts = X_train['BMI_cat'].value_counts().sort_index()",
    "for k, v in counts.items():",
    "    print(f'  {k} ({cat_labels[k]:>14}) : {v:>7,} ({v/len(X_train):.1%})')",
    "print(f'\\nNaN in BMI_cat: {X_train[\"BMI_cat\"].isna().sum()}')",
    "assert set(X_train['BMI_cat'].unique()).issubset({0,1,2,3,4,5,6}), 'Unerwartete Kategorie!'",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 3.4 Hurdle-Encoding → `MentHlth_any`, `MentHlth_days`, `PhysHlth_any`, `PhysHlth_days`",
    "",
    "**Formel:** Für jede Count-Variable `col`:",
    "- `col_any = (col > 0).astype(int)` — Binärindikator",
    "- `col_days = col` — Originalanzahl (umbenannt)",
    "- Original `col` wird entfernt.",
    "",
    "Begründung (NB02): Beide Variablen sind stark zero-inflated (MentHlth: 69 %, PhysHlth: 63 %",
    "Nullen). Ein einzelner Zahlenwert vermischt zwei inhaltlich verschiedene Informationen.",
    "Das Hurdle-Encoding trennt diese explizit und erlaubt dem Modell, beide Anteile",
    "unabhängig zu gewichten.",
    "",
    "**Quelle:** Mullahy J. 1986, *Journal of Econometrics*, 33(3):341–365.",
))

cells.append(code(
    "def hurdle_encode(X, cols=None):",
    "    if cols is None:",
    "        cols = ['MentHlth', 'PhysHlth']",
    "    X = X.copy()",
    "    for col in cols:",
    "        X[f'{col}_any']  = (X[col] > 0).astype(int)",
    "        X[f'{col}_days'] = X[col].astype(int)",
    "        X = X.drop(columns=[col])",
    "    return X",
    "",
    "X_train = hurdle_encode(X_train)",
    "X_test  = hurdle_encode(X_test)",
    "",
    "hurdle_cols = ['MentHlth_any', 'MentHlth_days', 'PhysHlth_any', 'PhysHlth_days']",
    "print('Neue Spalten nach hurdle_encode:')",
    "print(X_train[hurdle_cols].describe().T[['min','max','mean']].round(3))",
    "",
    "assert (X_train['MentHlth_any'] == (X_train['MentHlth_days'] > 0)).all()",
    "assert (X_train['PhysHlth_any'] == (X_train['PhysHlth_days'] > 0)).all()",
    "assert 'MentHlth' not in X_train.columns and 'PhysHlth' not in X_train.columns",
    "print(f'\\nSpalten gesamt: {X_train.shape[1]}')",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(code(
    "print(f'X_train Shape nach allen Schritt-3-Transformationen: {X_train.shape}')",
    "print(f'X_test  Shape: {X_test.shape}')",
    "print(f'Fehlende Werte (X_train): {X_train.isna().sum().sum()}')",
    "print(f'Aktuelle Spalten:')",
    "for c in X_train.columns:",
    "    print(f'  {c}')",
))

# ─── 4. Composite Features ────────────────────────────────────────────────────
cells.append(md(
    "## 4. Composite Features",
    "",
    "Composite Features fassen mehrere klinisch verwandte Einzel-Features zu einer",
    "Summenvariablen zusammen. Jedes Feature ist durch eine publizierte Risiko-Skala",
    "oder einen epidemiologischen Rahmen begründet.",
    "",
    "Alle Composites werden auf Kopien der DataFrames gebildet und danach an",
    "`X_train` bzw. `X_test` angehängt. Die Einzel-Features bleiben erhalten —",
    "welche Darstellung das Modell bevorzugt, entscheidet die Feature-Selektion in NB06.",
))

# --- cardio_comorbidity ---
cells.append(md(
    "### 4.1 `cardio_comorbidity`",
    "",
    "**Formel:**",
    "```",
    "cardio_comorbidity = HighBP + HighChol + HeartDiseaseorAttack + Stroke + DiffWalk",
    "```",
    "Wertebereich: 0 – 5 (ganzzählig).",
    "",
    "Zählt kardiovaskuläre und mobilitätsbezogene Komorbiditäten. Jede Komponente",
    "ist ein etablierter Risikofaktor im Framingham-Kontext.",
    "",
    "**Quelle:** D'Agostino et al. 2008, *Circulation*, 117(6):743–753 (Framingham Risk Score).",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['cardio_comorbidity'] = (",
    "        df['HighBP'] + df['HighChol'] + df['HeartDiseaseorAttack']",
    "        + df['Stroke'] + df['DiffWalk']",
    "    )",
    "",
    "print('cardio_comorbidity (X_train):')",
    "print(X_train['cardio_comorbidity'].value_counts().sort_index().to_string())",
    "print(f'NaN: {X_train[\"cardio_comorbidity\"].isna().sum()}')",
    "assert X_train['cardio_comorbidity'].between(0, 5).all()",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# --- allostatic_load ---
cells.append(md(
    "### 4.2 `allostatic_load`",
    "",
    "**Formel:**",
    "```",
    "allostatic_load = HighBP + HighChol + BMI_cat + PhysHlth_any + MentHlth_any",
    "```",
    "Wertebereich: 0 – 10 (abhängig von BMI_cat-Maximum 6).",
    "",
    "Allostatic Load bezeichnet die kumulative physiologische Belastung durch chronischen",
    "Stress. Die Kombination aus kardiometabolischen Markern (HighBP, HighChol), körperlicher",
    "Konstitution (BMI_cat) und subjektivem Gesundheitserleben (PhysHlth_any, MentHlth_any)",
    "folgt dem Operationalisierungsrahmen von McEwen.",
    "",
    "**Quelle:** McEwen & Stellar 1993, *Archives of Internal Medicine*, 153(18):2093–2101.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['allostatic_load'] = (",
    "        df['HighBP'] + df['HighChol'] + df['BMI_cat']",
    "        + df['PhysHlth_any'] + df['MentHlth_any']",
    "    )",
    "",
    "print('allostatic_load Verteilung (X_train):')",
    "print(X_train['allostatic_load'].describe().round(2))",
    "print(f'NaN: {X_train[\"allostatic_load\"].isna().sum()}')",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# --- healthy_lifestyle ---
cells.append(md(
    "### 4.3 `healthy_lifestyle`",
    "",
    "**Formel:**",
    "```",
    "healthy_lifestyle = PhysActivity + Fruits + Veggies + (1 - Smoker) + (1 - HvyAlcoholConsump)",
    "```",
    "Wertebereich: 0 – 5.",
    "",
    "Anlehnung an AHA's *Life's Simple 7*: physische Aktivität, Ernährung",
    "(Obst + Gemüse) und Rauchverzicht gelten als zentrale verhaltensbasierte",
    "Schutzfaktoren. Höhere Werte bedeuten gesündere Lebensweise.",
    "",
    "**Quelle:** Lloyd-Jones et al. 2010, *Circulation*, 121(4):586–613.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['healthy_lifestyle'] = (",
    "        df['PhysActivity'] + df['Fruits'] + df['Veggies']",
    "        + (1 - df['Smoker']) + (1 - df['HvyAlcoholConsump'])",
    "    )",
    "",
    "print('healthy_lifestyle Verteilung (X_train):')",
    "print(X_train['healthy_lifestyle'].value_counts().sort_index().to_string())",
    "assert X_train['healthy_lifestyle'].between(0, 5).all()",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# --- metsyn_proxy ---
cells.append(md(
    "### 4.4 `metsyn_proxy`",
    "",
    "**Formel:**",
    "```",
    "metsyn_proxy = HighBP + HighChol + (BMI_capped >= 30).astype(int) + HeartDiseaseorAttack",
    "```",
    "Wertebereich: 0 – 4.",
    "",
    "Approximiert das Metabolische Syndrom mit BRFSS-verfügbaren Variablen.",
    "BMI ≥ 30 entspricht dem IDF-Kriterium für zentrales Übergewicht als Kerndefinition.",
    "Glukose, Triglyceride und HDL liegen im BRFSS nicht vor.",
    "",
    "**Quelle:** International Diabetes Federation 2005,",
    "*The IDF consensus worldwide definition of the metabolic syndrome*.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['metsyn_proxy'] = (",
    "        df['HighBP'] + df['HighChol']",
    "        + (df['BMI_capped'] >= 30).astype(int)",
    "        + df['HeartDiseaseorAttack']",
    "    )",
    "",
    "print('metsyn_proxy Verteilung (X_train):')",
    "print(X_train['metsyn_proxy'].value_counts().sort_index().to_string())",
    "assert X_train['metsyn_proxy'].between(0, 4).all()",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# --- ses_index ---
cells.append(md(
    "### 4.5 `ses_index`",
    "",
    "**Formel:**",
    "```",
    "ses_index = Education + Income",
    "```",
    "Wertebereich: 2 – 14 (Education: 1–6, Income: 1–8).",
    "",
    "Sozialer Status ist ein starker Diabetesprädiktor über multiple Mechanismen",
    "(Ernährungszugang, Stresslevel, Gesundheitskompetenz). Education und Income",
    "sind die einzigen SES-Proxies im BRFSS-Datensatz und weisen laut NB04 eine",
    "moderate Korrelation auf (Cluster: SES).",
    "",
    "**Quelle:** Adler et al. 1994, *JAMA*, 269(24):3140–3145.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['ses_index'] = df['Education'] + df['Income']",
    "",
    "print('ses_index Verteilung (X_train):')",
    "print(X_train['ses_index'].describe().round(2))",
    "print(f'Bereich: {X_train[\"ses_index\"].min()} – {X_train[\"ses_index\"].max()}')",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# --- mental_physical_burden ---
cells.append(md(
    "### 4.6 `mental_physical_burden`",
    "",
    "**Formel:**",
    "```",
    "mental_physical_burden = MentHlth_days + PhysHlth_days",
    "```",
    "Wertebereich: 0 – 60.",
    "",
    "Addiert die Anzahl der mental und physisch eingeschränkten Tage zum Gesamtmaß",
    "der subjektiven Gesundheitsbeeinträchtigung. Entspricht dem Summary-Score-Ansatz",
    "des SF-36-Instruments zur gesundheitsbezogenen Lebensqualität.",
    "",
    "**Quelle:** Ware & Sherbourne 1992, *Medical Care*, 30(6):473–483 (SF-36).",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['mental_physical_burden'] = df['MentHlth_days'] + df['PhysHlth_days']",
    "",
    "print('mental_physical_burden Verteilung (X_train):')",
    "print(X_train['mental_physical_burden'].describe().round(2))",
    "assert X_train['mental_physical_burden'].between(0, 60).all()",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# --- findrisc_lite ---
cells.append(md(
    "### 4.7 `findrisc_lite`",
    "",
    "**Formel:**",
    "```",
    "findrisc_lite = BMI_cat + Age + (1 - PhysActivity) + (1 - Fruits) + (1 - Veggies)",
    "```",
    "Wertebereich: 0 – 14 (BMI_cat 0–6, Age 1–13).",
    "",
    "Vereinfachte Adaptation des Finnish Diabetes Risk Score (FINDRISC).",
    "Höhere Werte bedeuten höheres Diabetesrisiko. FINDRISC ist ein validierter",
    "Fragebogen-basierter Screening-Score für Typ-2-Diabetes.",
    "",
    "**Quelle:** Lindström & Tuomilehto 2003, *Diabetes Care*, 26(3):725–731.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['findrisc_lite'] = (",
    "        df['BMI_cat'] + df['Age']",
    "        + (1 - df['PhysActivity']) + (1 - df['Fruits']) + (1 - df['Veggies'])",
    "    )",
    "",
    "print('findrisc_lite Verteilung (X_train):')",
    "print(X_train['findrisc_lite'].describe().round(2))",
    "print(f'Bereich: {X_train[\"findrisc_lite\"].min()} – {X_train[\"findrisc_lite\"].max()}')",
    "print('Sanity-Check bestanden.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# --- ascvd_proxy ---
cells.append(md(
    "### 4.8 `ascvd_proxy`",
    "",
    "**Formel:**",
    "```",
    "ascvd_proxy = Age + HighBP + HighChol + Smoker + HeartDiseaseorAttack",
    "```",
    "Wertebereich: 1 – 17 (Age 1–13).",
    "",
    "Approximiert das arteriosklerotische Kardiovaskuläre Erkrankungsrisiko",
    "mit den im BRFSS verfügbaren Risikofaktoren der ACC/AHA Pooled Cohort Equations.",
    "Alter ist der dominante Koeffizient, ergänzt durch binäre Risikofaktoren.",
    "",
    "**Quelle:** Goff et al. 2014, *JACC*, 63(25 Pt B):2935–2959 (ACC/AHA PCE).",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['ascvd_proxy'] = (",
    "        df['Age'] + df['HighBP'] + df['HighChol']",
    "        + df['Smoker'] + df['HeartDiseaseorAttack']",
    "    )",
    "",
    "print('ascvd_proxy Verteilung (X_train):')",
    "print(X_train['ascvd_proxy'].describe().round(2))",
    "print(f'Bereich: {X_train[\"ascvd_proxy\"].min()} – {X_train[\"ascvd_proxy\"].max()}')",
    "print('Sanity-Check bestanden.')",
    "",
    "COMPOSITE_FEATURES = [",
    "    'cardio_comorbidity', 'allostatic_load', 'healthy_lifestyle', 'metsyn_proxy',",
    "    'ses_index', 'mental_physical_burden', 'findrisc_lite', 'ascvd_proxy',",
    "]",
    "print(f'\\nShape X_train nach Section 4: {X_train.shape}')",
))

# ─── 5. Interaktionsterme ─────────────────────────────────────────────────────
cells.append(md(
    "## 5. Interaktionsterme",
    "",
    "Interaktionsterme modellieren nicht-additive Effekte zweier Features.",
    "Sie werden als einfache Produkte berechnet — Baummodelle können solche",
    "Interaktionen grundsätzlich selbst lernen, aber explizite Terme beschleunigen",
    "die Konvergenz und erhöhen die Interpretierbarkeit linearer Ableitungen.",
    "",
    "Alle vier Terme sind epidemiologisch motiviert.",
))

cells.append(md(
    "### 5.1 `BMI_x_Age`",
    "",
    "**Formel:** `BMI_x_Age = BMI_capped × Age`",
    "",
    "Der Effekt von Übergewicht auf Diabetes ist altersabhängig: Bei jüngeren Personen",
    "ist der BMI-Effekt stärker ausgeprägt als bei älteren, wo Sarkopenie den BMI-Wert",
    "verzerren kann.",
    "",
    "**Quelle:** Janssen et al. 2005, *Obesity Research*, 13(12):2072–2079.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['BMI_x_Age'] = df['BMI_capped'] * df['Age']",
    "",
    "print('BMI_x_Age (X_train):')",
    "print(X_train['BMI_x_Age'].describe().round(2))",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 5.2 `BMI_x_HighBP`",
    "",
    "**Formel:** `BMI_x_HighBP = BMI_capped × HighBP`",
    "",
    "Hypertonie und Adipositas potenzieren sich gegenseitig im Diabetesrisiko;",
    "der kombinierte Effekt übertrifft die Summe der Einzeleffekte.",
    "",
    "**Quelle:** Landsberg et al. 2013, *Journal of Clinical Hypertension*, 15(1):14–33.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['BMI_x_HighBP'] = df['BMI_capped'] * df['HighBP']",
    "",
    "print('BMI_x_HighBP (X_train):')",
    "print(X_train['BMI_x_HighBP'].describe().round(2))",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 5.3 `Age_x_GenHlth`",
    "",
    "**Formel:** `Age_x_GenHlth = Age × GenHlth`",
    "",
    "GenHlth (allgemeiner Gesundheitszustand) und Alter sind die beiden stärksten",
    "Einzelprädiktoren (NB04: IV = 0.79 bzw. 0.40). Ihr Produkt erfasst, dass schlechter",
    "Gesundheitszustand im Alter besonders stark mit Diabetes assoziiert ist.",
    "",
    "**Quelle:** Hu et al. 2001, BRFSS Diabetes Predictors Study.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['Age_x_GenHlth'] = df['Age'] * df['GenHlth']",
    "",
    "print('Age_x_GenHlth (X_train):')",
    "print(X_train['Age_x_GenHlth'].describe().round(2))",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 5.4 `HighBP_x_HighChol`",
    "",
    "**Formel:** `HighBP_x_HighChol = HighBP × HighChol`",
    "",
    "Das gleichzeitige Vorliegen von Hypertonie und Hypercholesterinämie ist ein",
    "Kernkriterium des Metabolischen Syndroms. Das Produkt ist 1 nur wenn beide",
    "Bedingungen zutreffen (logisches AND bei Binärvariablen).",
    "",
    "**Quelle:** International Diabetes Federation 2005,",
    "*The IDF consensus worldwide definition of the metabolic syndrome*.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['HighBP_x_HighChol'] = df['HighBP'] * df['HighChol']",
    "",
    "print('HighBP_x_HighChol (X_train):')",
    "print(X_train['HighBP_x_HighChol'].value_counts().sort_index().to_string())",
    "",
    "INTERACTION_FEATURES = ['BMI_x_Age', 'BMI_x_HighBP', 'Age_x_GenHlth', 'HighBP_x_HighChol']",
    "print(f'\\nShape X_train nach Section 5: {X_train.shape}')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# ─── 6. Nichtlineare Transformationen ─────────────────────────────────────────
cells.append(md(
    "## 6. Polynomterme und nichtlineare Transformationen",
    "",
    "BMI zeigt in epidemiologischen Studien einen nichtlinearen Zusammenhang mit",
    "dem Diabetesrisiko (J- bzw. U-Kurve im unteren Bereich, steile Kurve im oberen).",
    "Für die Count-Variablen (MentHlth_days, PhysHlth_days) nivelliert die log-Transformation",
    "die zero-inflated Rechtsschiefe für lineare Modelle.",
))

cells.append(md(
    "### 6.1 `BMI_squared`",
    "",
    "**Formel:** `BMI_squared = BMI_capped²`",
    "",
    "Das Quadrat von BMI ermöglicht linearen Modellen, den überproportionalen Anstieg",
    "des Diabetesrisikos bei hohen BMI-Werten zu modellieren.",
    "",
    "**Quelle:** Tirosh et al. 2011, *NEJM*, 364(15):1315–1325.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['BMI_squared'] = df['BMI_capped'] ** 2",
    "",
    "print('BMI_squared (X_train):')",
    "print(X_train['BMI_squared'].describe().round(2))",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 6.2 `BMI_yj` — Yeo-Johnson-Transformation",
    "",
    "**Formel:** `BMI_yj = PowerTransformer(method='yeo-johnson').fit_transform(BMI_capped)`",
    "",
    "Die Yeo-Johnson-Transformation (Verallgemeinerung von Box-Cox auf negative Werte)",
    "normalisiert die BMI-Verteilung. Der Transformer wird **ausschließlich auf X_train**",
    "gefittet und dann auf X_test angewendet — kein Datenleck.",
    "",
    "Das gefittete `PowerTransformer`-Objekt wird unter `models/transformers/power_transformer.pkl`",
    "gespeichert und in NB07 (Pipelines) wiederverwendet.",
    "",
    "**Quelle:** Yeo & Johnson 2000, *Biometrika*, 87(4):954–959.",
))

cells.append(code(
    "pt = PowerTransformer(method='yeo-johnson', standardize=True)",
    "",
    "# Fit nur auf X_train!",
    "X_train['BMI_yj'] = pt.fit_transform(X_train[['BMI_capped']]).ravel()",
    "X_test['BMI_yj']  = pt.transform(X_test[['BMI_capped']]).ravel()",
    "",
    "print('PowerTransformer Lambda:', pt.lambdas_)",
    "print('BMI_yj (X_train):')",
    "print(X_train['BMI_yj'].describe().round(4))",
    "print(f'NaN in BMI_yj: {X_train[\"BMI_yj\"].isna().sum()}')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

cells.append(md(
    "### 6.3 `MentHlth_log` und `PhysHlth_log`",
    "",
    "**Formel:** `log1p(MentHlth_days)`, `log1p(PhysHlth_days)`",
    "",
    "`log1p(x) = ln(1 + x)` ist für `x = 0` definiert (→ 0) und behandelt damit die",
    "vielen Nullen in den Count-Variablen korrekt. Empfohlen für zero-inflated Zähldaten",
    "in linearen Modellen.",
    "",
    "**Quelle:** Mullahy J. 1986, *Journal of Econometrics*, 33(3):341–365.",
))

cells.append(code(
    "for df in [X_train, X_test]:",
    "    df['MentHlth_log'] = np.log1p(df['MentHlth_days'])",
    "    df['PhysHlth_log'] = np.log1p(df['PhysHlth_days'])",
    "",
    "print('MentHlth_log (X_train):')",
    "print(X_train['MentHlth_log'].describe().round(4))",
    "print('\\nPhysHlth_log (X_train):')",
    "print(X_train['PhysHlth_log'].describe().round(4))",
    "",
    "NONLINEAR_FEATURES = ['BMI_squared', 'BMI_yj', 'MentHlth_log', 'PhysHlth_log']",
    "print(f'\\nShape X_train nach Section 6: {X_train.shape}')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# ─── 7. IV/MI Vorher-Nachher ──────────────────────────────────────────────────
cells.append(md(
    "## 7. IV/MI Vorher-Nachher-Vergleich",
    "",
    "Für jedes neue Feature werden IV (OptimalBinning) und MI",
    "(mutual_info_classif) berechnet und mit dem jeweils stärksten Einzel-Feature",
    "unter den Komponenten verglichen (Werte aus NB04).",
    "",
    "**Entscheidungsregel:** Ein neues Feature wird beibehalten, wenn",
    "`IV_neu > IV_alt ODER MI_neu > MI_alt`.",
    "",
    "Diese Regel ist liberal — endgültige Selektion erfolgt in NB06 über",
    "Permutation Importance und/oder SHAP.",
))

cells.append(code(
    "# Referenz: bestes Einzelfeature pro neuem Feature",
    "REFERENCE = {",
    "    # Composite",
    "    'cardio_comorbidity':      ('HighBP',               False),",
    "    'allostatic_load':         ('HighBP',               False),",
    "    'healthy_lifestyle':       ('PhysActivity',         False),",
    "    'metsyn_proxy':            ('HighBP',               False),",
    "    'ses_index':               ('Income',               False),",
    "    'mental_physical_burden':  ('PhysHlth',             False),",
    "    'findrisc_lite':           ('BMI',                  False),",
    "    'ascvd_proxy':             ('HighBP',               False),",
    "    # Interaktionen",
    "    'BMI_x_Age':               ('BMI',                  True),",
    "    'BMI_x_HighBP':            ('HighBP',               True),",
    "    'Age_x_GenHlth':           ('GenHlth',              True),",
    "    'HighBP_x_HighChol':       ('HighBP',               False),",
    "    # Nichtlinear",
    "    'BMI_squared':             ('BMI',                  True),",
    "    'BMI_yj':                  ('BMI',                  True),",
    "    'MentHlth_log':            ('MentHlth',             True),",
    "    'PhysHlth_log':            ('PhysHlth',             True),",
    "}",
    "",
    "NEW_FEATURES = COMPOSITE_FEATURES + INTERACTION_FEATURES + NONLINEAR_FEATURES",
    "",
    "print(f'Neue Features gesamt: {len(NEW_FEATURES)}')",
    "print('Starte IV/MI-Berechnung ...')",
))

cells.append(code(
    "results = []",
    "for feat in NEW_FEATURES:",
    "    ref_feat, is_cont = REFERENCE[feat]",
    "    iv_neu = compute_iv(X_train[feat], y_train, feat, is_continuous=is_cont)",
    "    mi_neu = compute_mi(X_train[feat], y_train, is_discrete=not is_cont)",
    "    iv_alt = NB04_IV.get(ref_feat, np.nan)",
    "    mi_alt = NB04_MI.get(ref_feat, np.nan)",
    "    behalten = (iv_neu > iv_alt) or (mi_neu > mi_alt)",
    "    results.append({",
    "        'Feature':            feat,",
    "        'IV_neu':             round(iv_neu, 4),",
    "        'MI_neu':             round(mi_neu, 4),",
    "        'Bestes_Einzelfeat':  ref_feat,",
    "        'IV_alt':             round(iv_alt, 4),",
    "        'MI_alt':             round(mi_alt, 4),",
    "        'behalten':           behalten,",
    "    })",
    "    print(f'  {feat:<28} IV={iv_neu:.4f}  MI={mi_neu:.4f}  {'✓' if behalten else '✗'}')",
    "",
    "comparison_df = pd.DataFrame(results)",
    "print(f'\\nFertig. {comparison_df[\"behalten\"].sum()} / {len(comparison_df)} Features beibehalten.')",
))

cells.append(code(
    "print(comparison_df.to_string(index=False))",
))

cells.append(code(
    "# Balkendiagramm IV Vorher-Nachher",
    "fig, axes = plt.subplots(1, 2, figsize=(14, 6))",
    "",
    "x = range(len(comparison_df))",
    "labels = comparison_df['Feature'].tolist()",
    "colors = ['#2ecc71' if b else '#e74c3c' for b in comparison_df['behalten']]",
    "",
    "# IV",
    "axes[0].bar([i - 0.2 for i in x], comparison_df['IV_neu'], width=0.4,",
    "            label='IV neu', color=colors, alpha=0.85)",
    "axes[0].bar([i + 0.2 for i in x], comparison_df['IV_alt'], width=0.4,",
    "            label='IV Referenz', color='#3498db', alpha=0.5)",
    "axes[0].set_xticks(list(x))",
    "axes[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)",
    "axes[0].set_title('IV — neue Features vs. Referenz')",
    "axes[0].set_ylabel('Information Value')",
    "axes[0].legend()",
    "",
    "# MI",
    "axes[1].bar([i - 0.2 for i in x], comparison_df['MI_neu'], width=0.4,",
    "            label='MI neu', color=colors, alpha=0.85)",
    "axes[1].bar([i + 0.2 for i in x], comparison_df['MI_alt'], width=0.4,",
    "            label='MI Referenz', color='#3498db', alpha=0.5)",
    "axes[1].set_xticks(list(x))",
    "axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=8)",
    "axes[1].set_title('MI — neue Features vs. Referenz')",
    "axes[1].set_ylabel('Mutual Information')",
    "axes[1].legend()",
    "",
    "fig.suptitle('IV/MI Vorher-Nachher-Vergleich — neue Features', fontsize=13)",
    "plt.tight_layout()",
    "plt.savefig(OUTPUTS_DIR / 'iv_mi_comparison.png', dpi=150, bbox_inches='tight')",
    "plt.show()",
))

cells.append(code(
    "# Features droppen, die weder IV noch MI-Kriterium erfüllen",
    "DROP_ENGINEERED = comparison_df.loc[~comparison_df['behalten'], 'Feature'].tolist()",
    "KEEP_ENGINEERED = comparison_df.loc[comparison_df['behalten'],  'Feature'].tolist()",
    "",
    "if DROP_ENGINEERED:",
    "    print(f'Werden entfernt ({len(DROP_ENGINEERED)}): {DROP_ENGINEERED}')",
    "    X_train = X_train.drop(columns=DROP_ENGINEERED)",
    "    X_test  = X_test.drop(columns=DROP_ENGINEERED)",
    "else:",
    "    print('Alle neuen Features beibehalten.')",
    "",
    "print(f'Shape X_train nach Filterung: {X_train.shape}')",
    "comparison_df.to_csv(OUTPUTS_DIR / 'iv_mi_comparison.csv', index=False)",
    "print('Vergleichstabelle gespeichert.')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# ─── 8. Export ────────────────────────────────────────────────────────────────
cells.append(md(
    "## 8. Export",
    "",
    "Die angereicherten Datensätze werden als Parquet-Dateien gespeichert.",
    "Zusätzlich werden die Feature-Metadaten (Spaltengruppen) und der gefittete",
    "`PowerTransformer` persistiert.",
    "",
    "**Dateien:**",
    "- `data/processed/X_train_enriched.parquet`",
    "- `data/processed/X_test_enriched.parquet`",
    "- `data/processed/feature_meta_enriched.json`",
    "- `models/transformers/power_transformer.pkl`",
))

cells.append(code(
    "# Parquet-Export",
    "X_train.to_parquet(PROCESSED_DIR / 'X_train_enriched.parquet', index=True)",
    "X_test.to_parquet(PROCESSED_DIR  / 'X_test_enriched.parquet',  index=True)",
    "",
    "print(f'X_train_enriched : {X_train.shape}')",
    "print(f'X_test_enriched  : {X_test.shape}')",
))

cells.append(code(
    "# Feature-Metadaten",
    "base_cols = [c for c in X_train.columns if c in (",
    "    BINARY_COLS + ORDINAL_COLS + ['BMI']",
    ") and c not in ['MentHlth', 'PhysHlth'] + DROP_COLS]",
    "",
    "meta_enriched = {",
    "    'seed': SEED,",
    "    'base_features': base_cols,",
    "    'bmi_features':  ['BMI', 'BMI_capped', 'BMI_cat', 'BMI_squared', 'BMI_yj'],",
    "    'hurdle_features': ['MentHlth_any', 'MentHlth_days', 'MentHlth_log',",
    "                        'PhysHlth_any',  'PhysHlth_days',  'PhysHlth_log'],",
    "    'composite_features': [f for f in COMPOSITE_FEATURES if f in X_train.columns],",
    "    'interaction_features': [f for f in INTERACTION_FEATURES if f in X_train.columns],",
    "    'nonlinear_features':   [f for f in NONLINEAR_FEATURES   if f in X_train.columns],",
    "    'all_features': list(X_train.columns),",
    "    'dropped_original':   DROP_COLS,",
    "    'dropped_engineered': DROP_ENGINEERED,",
    "}",
    "",
    "with open(PROCESSED_DIR / 'feature_meta_enriched.json', 'w') as f:",
    "    json.dump(meta_enriched, f, indent=2)",
    "",
    "print('feature_meta_enriched.json gespeichert.')",
    "print(f'Gesamte Feature-Anzahl: {len(meta_enriched[\"all_features\"])}')",
    "for group, cols in meta_enriched.items():",
    "    if isinstance(cols, list) and group != 'all_features':",
    "        print(f'  {group:<24}: {len(cols)}')",
))

cells.append(code(
    "# PowerTransformer speichern",
    "joblib.dump(pt, MODELS_DIR / 'power_transformer.pkl')",
    "print(f'PowerTransformer gespeichert: {(MODELS_DIR / \"power_transformer.pkl\").resolve()}')",
    "",
    "# Verifikation: Laden und kurzer Check",
    "pt_loaded = joblib.load(MODELS_DIR / 'power_transformer.pkl')",
    "test_val = pt_loaded.transform([[30.0]])[0][0]",
    "print(f'Verifikation Laden: pt.transform([[30.0]]) = {test_val:.4f}')",
))

cells.append(md("### Interpretation\n*Wird nach Ausführung ergänzt.*"))

# ─── 9. Zusammenfassung ───────────────────────────────────────────────────────
cells.append(md(
    "## 9. Zusammenfassung",
    "",
    "Dieses Notebook hat das Feature-Set in vier Schritten angereichert:",
    "",
    "**Schritt 1 — Bereinigung:** Drei Features mit nachgewiesener Schwäche in allen vier",
    "NB04-Metriken wurden entfernt (`AnyHealthcare`, `NoDocbcCost`, `Sex`).",
    "",
    "**Schritt 2 — Basisransformationen:** BMI wurde gecappt (`BMI_capped`) und kategorisiert",
    "(`BMI_cat`). `MentHlth` und `PhysHlth` wurden per Hurdle-Encoding in je zwei Features",
    "aufgetrennt.",
    "",
    "**Schritt 3 — Composite Features:** Acht klinisch begründete Summenskalen wurden",
    "konstruiert (FINDRISC, Allostatic Load, Metabolisches Syndrom-Proxy u. a.).",
    "",
    "**Schritt 4 — Interaktionen und Nichtlinearitäten:** Vier Interaktionsterme und vier",
    "nichtlineare Transformationen (BMI²,  Yeo-Johnson, log1p) wurden ergänzt.",
    "",
    "**IV/MI-Filter:** Features, die weder IV noch MI über dem Referenz-Einzelfeature liegen,",
    "wurden entfernt.",
    "",
    "**Gespeicherte Artefakte:**",
    "- `data/processed/X_train_enriched.parquet`",
    "- `data/processed/X_test_enriched.parquet`",
    "- `data/processed/feature_meta_enriched.json`",
    "- `models/transformers/power_transformer.pkl`",
    "- `outputs/05_feature_engineering/iv_mi_comparison.csv`",
    "- `outputs/05_feature_engineering/iv_mi_comparison.png`",
    "",
    "**Nächster Schritt:** NB06 — Baseline-Modellierung auf `X_train_enriched` mit",
    "Stratified K-Fold Cross-Validation (PR-AUC als Primärmetrik).",
))

# ─── Notebook bauen ───────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "cells": cells,
}

out = pathlib.Path("notebooks/05_feature_engineering.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Geschrieben: {out}  ({len(cells)} Zellen)")
