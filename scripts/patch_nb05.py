"""
Applies all four patch points to notebooks/05_feature_engineering.ipynb,
creates src/utils.py, and updates notebooks/03_data_preparation.ipynb.
"""
import json, pathlib, sys, copy

# ─── helpers ──────────────────────────────────────────────────────────────────
def load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

def save(nb, path):
    pathlib.Path(path).write_text(
        json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8"
    )

def src(cell):
    """Join list-of-lines source into one string."""
    return "".join(cell["source"]) if cell["source"] else ""

def set_src(cell, text):
    """Store text back as a list of lines (standard .ipynb format)."""
    if not text:
        cell["source"] = []
        return
    lines = text.split('\n')
    result = [line + '\n' for line in lines[:-1]]
    if lines[-1]:
        result.append(lines[-1])
    cell["source"] = result

def find_cell(cells, snippet, cell_type="code"):
    """Return index of first cell whose source contains snippet."""
    for i, c in enumerate(cells):
        if c["cell_type"] == cell_type and snippet in src(c):
            return i
    raise ValueError(f"Cell not found: {snippet!r}")

# ═══════════════════════════════════════════════════════════════════════════════
# 1.  src/utils.py
# ═══════════════════════════════════════════════════════════════════════════════
UTILS_SRC = '''\
import pandas as pd


def cap_bmi(X, lower=18, upper=50):
    """Fügt BMI_capped hinzu (Originalwert BMI bleibt erhalten)."""
    X = X.copy()
    X["BMI_capped"] = X["BMI"].clip(lower=lower, upper=upper)
    return X


def categorize_bmi(X):
    """Fügt BMI_cat hinzu (7 WHO-Klassen 0-6; Originalwert BMI bleibt erhalten)."""
    X = X.copy()
    bins   = [0, 18.5, 25, 30, 35, 40, 50, 999]
    labels = [0, 1, 2, 3, 4, 5, 6]
    X["BMI_cat"] = pd.cut(X["BMI"], bins=bins, labels=labels).astype(int)
    return X


def hurdle_encode(X, cols=None):
    """Hurdle-Encoding: ersetzt jede Zählvariable durch _any + _days."""
    if cols is None:
        cols = ["MentHlth", "PhysHlth"]
    X = X.copy()
    for col in cols:
        X[f"{col}_any"]  = (X[col] > 0).astype(int)
        X[f"{col}_days"] = X[col].astype(int)
        X = X.drop(columns=[col])
    return X
'''

src_dir = pathlib.Path("src")
src_dir.mkdir(exist_ok=True)
init = src_dir / "__init__.py"
if not init.exists():
    init.write_text("", encoding="utf-8")
(src_dir / "utils.py").write_text(UTILS_SRC, encoding="utf-8")
print("src/utils.py geschrieben.")

# ═══════════════════════════════════════════════════════════════════════════════
# 2.  NB05 patches
# ═══════════════════════════════════════════════════════════════════════════════
nb05 = load("notebooks/05_feature_engineering.ipynb")
cells = nb05["cells"]

# ── 2a. Setup-Zelle: sys.path + Import ───────────────────────────────────────
idx = find_cell(cells, "PROCESSED_DIR = Path('../data/processed')")
old = src(cells[idx])
# Prepend path-setup and import after the optbinning install block
insert = (
    "\nimport sys\n"
    "sys.path.insert(0, str(pathlib.Path('..').resolve()))\n"
    "from src.utils import cap_bmi, categorize_bmi, hurdle_encode\n"
)
# Insert after the optbinning install block (after the blank line before warnings)
set_src(cells[idx], old.replace(
    "import warnings\nwarnings.filterwarnings('ignore')",
    "import warnings\nwarnings.filterwarnings('ignore')" + insert,
    1
))
print(f"  NB05 Setup-Zelle: sys.path + Import ergänzt (Zelle {idx}).")

# ── 2b. cap_bmi-Zelle: def entfernen ─────────────────────────────────────────
idx = find_cell(cells, "def cap_bmi(")
old = src(cells[idx])
# Remove the function definition block (up to the blank line before the call)
new = "\n".join(
    line for line in old.splitlines()
    if not line.startswith("def cap_bmi") and
       not line.startswith("    X = X.copy()") and
       not (line.startswith("    X['BMI_capped']") and "clip" in line) and
       not line.startswith("    return X")
).lstrip("\n")
# Cleaner: just hardcode the application-only source
set_src(cells[idx],
    "X_train = cap_bmi(X_train)\n"
    "X_test  = cap_bmi(X_test)\n"
    "\n"
    "n_affected_train = ((X_train['BMI'] < 18) | (X_train['BMI'] > 50)).sum()\n"
    "print(f'BMI_capped Bereich: [{X_train[\"BMI_capped\"].min():.1f}, {X_train[\"BMI_capped\"].max():.1f}]')\n"
    "print(f'Gecappte Samples   : {n_affected_train} ({n_affected_train/len(X_train):.2%})')\n"
    "print(f'NaN in BMI_capped  : {X_train[\"BMI_capped\"].isna().sum()}')\n"
    "assert X_train['BMI_capped'].between(18, 50).all(), 'BMI_capped außerhalb [18, 50]!'\n"
    "print('Sanity-Check bestanden.')"
)
print(f"  NB05 cap_bmi-Zelle: def entfernt (Zelle {idx}).")

# ── 2c. categorize_bmi-Zelle: def entfernen ──────────────────────────────────
idx = find_cell(cells, "def categorize_bmi(")
set_src(cells[idx],
    "X_train = categorize_bmi(X_train)\n"
    "X_test  = categorize_bmi(X_test)\n"
    "\n"
    "print('BMI_cat Häufigkeiten (X_train):')\n"
    "cat_labels = {0:'Untergewicht', 1:'Normalgewicht', 2:'Übergewicht',\n"
    "              3:'Adipositas I', 4:'Adipositas II', 5:'Adipositas III', 6:'Super'}\n"
    "counts = X_train['BMI_cat'].value_counts().sort_index()\n"
    "for k, v in counts.items():\n"
    "    print(f'  {k} ({cat_labels[k]:>14}) : {v:>7,} ({v/len(X_train):.1%})')\n"
    "print(f'\\nNaN in BMI_cat: {X_train[\"BMI_cat\"].isna().sum()}')\n"
    "assert set(X_train['BMI_cat'].unique()).issubset({0,1,2,3,4,5,6}), 'Unerwartete Kategorie!'\n"
    "print('Sanity-Check bestanden.')"
)
print(f"  NB05 categorize_bmi-Zelle: def entfernt (Zelle {idx}).")

# ── 2d. hurdle_encode-Zelle: def entfernen ───────────────────────────────────
idx = find_cell(cells, "def hurdle_encode(")
set_src(cells[idx],
    "X_train = hurdle_encode(X_train)\n"
    "X_test  = hurdle_encode(X_test)\n"
    "\n"
    "hurdle_cols = ['MentHlth_any', 'MentHlth_days', 'PhysHlth_any', 'PhysHlth_days']\n"
    "print('Neue Spalten nach hurdle_encode:')\n"
    "print(X_train[hurdle_cols].describe().T[['min','max','mean']].round(3))\n"
    "\n"
    "assert (X_train['MentHlth_any'] == (X_train['MentHlth_days'] > 0)).all()\n"
    "assert (X_train['PhysHlth_any'] == (X_train['PhysHlth_days'] > 0)).all()\n"
    "assert 'MentHlth' not in X_train.columns and 'PhysHlth' not in X_train.columns\n"
    "print(f'\\nSpalten gesamt: {X_train.shape[1]}')\n"
    "print('Sanity-Check bestanden.')"
)
print(f"  NB05 hurdle_encode-Zelle: def entfernt (Zelle {idx}).")

# ── 2e. compute_iv: MIP-Fallback + Logging ───────────────────────────────────
idx = find_cell(cells, "def compute_iv(")
set_src(cells[idx],
    "# Hilfsfunktion: IV und MI für ein einzelnes Feature berechnen\n"
    "_iv_nan_log = []  # sammelt Features, bei denen OptimalBinning fehlschlug\n"
    "\n"
    "def compute_iv(feat_series, y_series, feature_name, is_continuous=False):\n"
    "    \"\"\"IV via OptimalBinning (cp → mip Fallback). NaN wird geloggt.\"\"\"\n"
    "    X_arr = feat_series.values.astype(float)\n"
    "    y_arr = y_series.values.astype(int)\n"
    "    for solver in ('cp', 'mip'):\n"
    "        try:\n"
    "            ob = OptimalBinning(name=feature_name, dtype='numerical',\n"
    "                                solver=solver, max_n_bins=10)\n"
    "            ob.fit(X_arr, y_arr)\n"
    "            iv = ob.iv\n"
    "            if not (iv != iv):  # nicht NaN\n"
    "                return iv\n"
    "        except Exception:\n"
    "            pass\n"
    "    # Beide Solver gescheitert\n"
    "    _iv_nan_log.append(feature_name)\n"
    "    return float('nan')\n"
    "\n"
    "def compute_mi(feat_series, y_series, is_discrete=True):\n"
    "    X_arr = feat_series.values.reshape(-1, 1)\n"
    "    mi = mutual_info_classif(\n"
    "        X_arr, y_series.values,\n"
    "        discrete_features=[is_discrete],\n"
    "        random_state=SEED,\n"
    "    )[0]\n"
    "    return round(mi, 4)\n"
    "\n"
    "# NB04-Referenzwerte (aus outputs/04_diagnostics/feature_diagnostics.csv)\n"
    "NB04_IV = {\n"
    "    'GenHlth': 0.7905, 'HighBP': 0.6095, 'BMI': 0.4694, 'Age': 0.3970,\n"
    "    'HighChol': 0.3389, 'DiffWalk': 0.3170, 'Income': 0.2281, 'PhysHlth': 0.2194,\n"
    "    'HeartDiseaseorAttack': 0.1939, 'Education': 0.1226, 'PhysActivity': 0.1041,\n"
    "    'MentHlth': 0.0402, 'HvyAlcoholConsump': 0.0379, 'Smoker': 0.0312,\n"
    "    'Veggies': 0.0234, 'Fruits': 0.0154, 'Stroke': 0.0, 'CholCheck': 0.0,\n"
    "}\n"
    "NB04_MI = {\n"
    "    'GenHlth': 0.0441, 'HighBP': 0.0351, 'BMI': 0.0285, 'Age': 0.0203,\n"
    "    'HighChol': 0.0200, 'DiffWalk': 0.0202, 'Income': 0.0136, 'PhysHlth': 0.0141,\n"
    "    'HeartDiseaseorAttack': 0.0126, 'Education': 0.0075, 'PhysActivity': 0.0064,\n"
    "    'MentHlth': 0.0026, 'HvyAlcoholConsump': 0.0020, 'Smoker': 0.0019,\n"
    "    'Veggies': 0.0014, 'Fruits': 0.0009, 'Stroke': 0.0045, 'CholCheck': 0.0031,\n"
    "}\n"
    "print('Referenzwerte geladen.')"
)
print(f"  NB05 compute_iv: MIP-Fallback + Logging ergänzt (Zelle {idx}).")

# ── 2f. REFERENCE-Zelle: MentHlth_days/PhysHlth_days + NaN-Report ────────────
idx = find_cell(cells, "REFERENCE = {")
set_src(cells[idx],
    "# Referenzwerte für hurdle-encodierte Zählfeatures live berechnen\n"
    "# (MentHlth/PhysHlth existieren nach hurdle_encode nicht mehr im DataFrame)\n"
    "_hurdle_refs = ['MentHlth_days', 'PhysHlth_days']\n"
    "for _col in _hurdle_refs:\n"
    "    NB04_IV[_col] = compute_iv(X_train[_col], y_train, _col)\n"
    "    NB04_MI[_col] = compute_mi(X_train[_col], y_train, is_discrete=True)\n"
    "    print(f'{_col}: IV={NB04_IV[_col]:.4f}  MI={NB04_MI[_col]:.4f}')\n"
    "\n"
    "# Referenz: bestes Einzelfeature pro neuem Feature\n"
    "REFERENCE = {\n"
    "    # Composite\n"
    "    'cardio_comorbidity':      ('HighBP',               False),\n"
    "    'allostatic_load':         ('HighBP',               False),\n"
    "    'healthy_lifestyle':       ('PhysActivity',         False),\n"
    "    'metsyn_proxy':            ('HighBP',               False),\n"
    "    'ses_index':               ('Income',               False),\n"
    "    'mental_physical_burden':  ('PhysHlth',             False),\n"
    "    'findrisc_lite':           ('BMI',                  False),\n"
    "    'ascvd_proxy':             ('HighBP',               False),\n"
    "    # Interaktionen\n"
    "    'BMI_x_Age':               ('BMI',                  True),\n"
    "    'BMI_x_HighBP':            ('HighBP',               True),\n"
    "    'Age_x_GenHlth':           ('GenHlth',              True),\n"
    "    'HighBP_x_HighChol':       ('HighBP',               False),\n"
    "    # Nichtlinear — Referenz sind die hurdle-encodierten _days-Features\n"
    "    'BMI_squared':             ('BMI',                  True),\n"
    "    'BMI_yj':                  ('BMI',                  True),\n"
    "    'MentHlth_log':            ('MentHlth_days',        True),\n"
    "    'PhysHlth_log':            ('PhysHlth_days',        True),\n"
    "}\n"
    "\n"
    "NEW_FEATURES = COMPOSITE_FEATURES + INTERACTION_FEATURES + NONLINEAR_FEATURES\n"
    "\n"
    "print(f'Neue Features gesamt: {len(NEW_FEATURES)}')\n"
    "print('Starte IV/MI-Berechnung ...')"
)
print(f"  NB05 REFERENCE-Zelle: MentHlth_days/PhysHlth_days + live MI (Zelle {idx}).")

# ── 2g. Nach der Ergebnis-Schleife: NaN-Report-Zelle einfügen ────────────────
# Finde die Zelle mit dem Ergebnis-Print und füge danach den NaN-Report ein
loop_idx = find_cell(cells, "comparison_df = pd.DataFrame(results)")
loop_src = src(cells[loop_idx])
# Append NaN report to the loop cell
if "_iv_nan_log" not in loop_src:
    set_src(cells[loop_idx], loop_src + "\n\n"
        "# NaN-Report: Features, bei denen IV nicht berechnet werden konnte\n"
        "if _iv_nan_log:\n"
        "    print(f'\\nIV-NaN ({len(_iv_nan_log)} Features) — Entscheidung basiert ausschließlich auf MI:')\n"
        "    for _f in _iv_nan_log:\n"
        "        _row = comparison_df[comparison_df['Feature'] == _f].iloc[0]\n"
        "        _decision = 'beibehalten (MI)' if _row['behalten'] else 'entfernt (MI)'\n"
        "        print(f'  {_f:<28} MI_neu={_row[\"MI_neu\"]:.4f}  MI_alt={_row[\"MI_alt\"]:.4f}  → {_decision}')\n"
        "else:\n"
        "    print('IV für alle Features erfolgreich berechnet.')"
    )
    print(f"  NB05 Loop-Zelle: NaN-Report angehängt (Zelle {loop_idx}).")

# ── 2h. bmi_features: list comprehension statt hardcoded ─────────────────────
idx = find_cell(cells, "'bmi_features':")
old = src(cells[idx])
set_src(cells[idx], old.replace(
    "    'bmi_features':  ['BMI', 'BMI_capped', 'BMI_cat', 'BMI_squared', 'BMI_yj'],",
    "    'bmi_features':  [f for f in ['BMI', 'BMI_capped', 'BMI_cat', 'BMI_squared', 'BMI_yj'] if f in X_train.columns],",
    1
))
print(f"  NB05 meta_enriched: bmi_features als list comprehension (Zelle {idx}).")

save(nb05, "notebooks/05_feature_engineering.ipynb")
print("NB05 gespeichert.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 3.  NB03 patches
# ═══════════════════════════════════════════════════════════════════════════════
nb03 = load("notebooks/03_data_preparation.ipynb")
cells3 = nb03["cells"]

# ── 3a. Setup-Zelle: sys.path ergänzen ───────────────────────────────────────
idx3 = find_cell(cells3, "SEED = 42")
old3 = src(cells3[idx3])
if "sys.path" not in old3:
    set_src(cells3[idx3], old3 + (
        "\n"
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path('..').resolve()))\n"
    ))
    print(f"  NB03 Setup-Zelle: sys.path ergänzt (Zelle {idx3}).")

# ── 3b. Drei Funktionsdefinitionen durch einen Import ersetzen ────────────────
# Ersetze cap_bmi-Zelle durch den gemeinsamen Import
cap_idx = find_cell(cells3, "def cap_bmi(")
set_src(cells3[cap_idx],
    "from src.utils import cap_bmi, categorize_bmi, hurdle_encode\n"
    "print('Transformationsfunktionen aus src.utils importiert.')"
)
print(f"  NB03 cap_bmi-Zelle: durch Import ersetzt (Zelle {cap_idx}).")

# categorize_bmi und hurdle_encode zu No-Op-Zellen (leer wäre ungültig → Kommentar)
cat_idx = find_cell(cells3, "def categorize_bmi(")
set_src(cells3[cat_idx], "# categorize_bmi wird aus src.utils importiert (siehe Zelle oben).")
print(f"  NB03 categorize_bmi-Zelle: zu Kommentar-Zelle (Zelle {cat_idx}).")

hurdle_idx = find_cell(cells3, "def hurdle_encode(")
set_src(cells3[hurdle_idx], "# hurdle_encode wird aus src.utils importiert (siehe Zelle oben).")
print(f"  NB03 hurdle_encode-Zelle: zu Kommentar-Zelle (Zelle {hurdle_idx}).")

# ── 3c. Sanity-Check-Plot-Zelle: BMI → BMI_capped / BMI → BMI_cat ───────────
plot_idx = find_cell(cells3, "X_capped = cap_bmi(X_train)")
old_plot = src(cells3[plot_idx])
new_plot = (old_plot
    # Plot 2 uses X_capped["BMI"] for histogram
    .replace('X_capped["BMI"].plot.hist', 'X_capped["BMI_capped"].plot.hist')
    # n_affected title line unchanged (uses X_train["BMI"] directly — OK)
    # Plot 3: X_cat["BMI"].value_counts()
    .replace('counts = X_cat["BMI"].value_counts()', 'counts = X_cat["BMI_cat"].value_counts()')
    # Print min/max after capping
    .replace("X_capped['BMI'].min():.1f} / {X_capped['BMI'].max():.1f}",
             "X_capped['BMI_capped'].min():.1f} / {X_capped['BMI_capped'].max():.1f}")
)
set_src(cells3[plot_idx], new_plot)
print(f"  NB03 Plot-Zelle: BMI_capped/BMI_cat referenziert (Zelle {plot_idx}).")

# ── 3d. Section-5-Markdown anpassen ──────────────────────────────────────────
for i, c in enumerate(cells3):
    if c["cell_type"] == "markdown" and "Die folgenden Funktionen kapseln" in src(c):
        old_md = src(c)
        set_src(cells3[i], old_md.replace(
            "Die folgenden Funktionen kapseln alle in Notebook 02 identifizierten\n"
            "Transformationen. Sie werden hier nur definiert — die Anwendung und der\n"
            "Vergleich der Varianten erfolgt in Notebook 04.\n"
            "\n"
            "Alle Funktionen arbeiten auf einer Kopie des DataFrames und geben einen\n"
            "neuen DataFrame zurück.",
            "Die Transformationsfunktionen `cap_bmi`, `categorize_bmi` und `hurdle_encode`\n"
            "sind in `src/utils.py` definiert und werden hier importiert. Dadurch können\n"
            "NB05 (Feature Engineering) und zukünftige Notebooks dieselben Implementierungen\n"
            "ohne Redundanz nutzen.\n"
            "\n"
            "Alle Funktionen arbeiten auf einer Kopie des DataFrames und geben einen\n"
            "neuen DataFrame zurück.",
        ))
        print(f"  NB03 Section-5-Markdown angepasst (Zelle {i}).")
        break

save(nb03, "notebooks/03_data_preparation.ipynb")
print("NB03 gespeichert.\n")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Abschlussprüfung
# ═══════════════════════════════════════════════════════════════════════════════
import ast

errors = []
for nb_path in ["notebooks/05_feature_engineering.ipynb",
                "notebooks/03_data_preparation.ipynb"]:
    nb = load(nb_path)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            try:
                ast.parse(src(c))
            except SyntaxError as e:
                errors.append(f"{nb_path}: {e}  in: {src(c)[:60]!r}")

if errors:
    print("SYNTAXFEHLER:")
    for e in errors:
        print(" ", e)
    sys.exit(1)
else:
    print("Syntaxprüfung bestanden — alle Code-Zellen fehlerfrei.")

# Prüfe, dass def-Blöcke aus NB05 entfernt wurden
nb05_src = pathlib.Path("notebooks/05_feature_engineering.ipynb").read_text(encoding="utf-8")
for fn in ["def cap_bmi", "def categorize_bmi", "def hurdle_encode"]:
    if fn in nb05_src:
        print(f"WARNUNG: {fn!r} noch in NB05 vorhanden!")
    else:
        print(f"OK: {fn!r} nicht mehr in NB05.")

# Prüfe import in NB05
if "from src.utils import" in nb05_src:
    print("OK: 'from src.utils import' in NB05 vorhanden.")
else:
    print("WARNUNG: Import fehlt in NB05!")

print("\nFertig.")
