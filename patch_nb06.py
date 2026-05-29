import json

with open('notebooks/06_modeling.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']

# ---------------------------------------------------------------
# 1 & 2: Patch code-model-compare (cell index 24)
# ---------------------------------------------------------------
c24 = cells[24]
assert c24['id'] == 'code-model-compare', c24['id']
src = ''.join(c24['source'])

# Fix 1: remove eval_metric='PRAUC' from CatBoost
old_cat = "    ('CatBoost', CatBoostClassifier(\n        scale_pos_weight=spw, eval_metric='PRAUC',\n        verbose=0, random_state=SEED\n    )),"
new_cat = "    ('CatBoost', CatBoostClassifier(\n        scale_pos_weight=spw, verbose=0, random_state=SEED\n    )),"
assert old_cat in src, f"CatBoost pattern not found in:\n{src[src.find('CatBoost'):src.find('CatBoost')+200]}"
src = src.replace(old_cat, new_cat)
print("Fix 1 (CatBoost eval_metric removed): OK")

# Fix 2: sharper diversity filter — max 1 Boosting
old_filter = (
    "# Top-3 nach PR-AUC (Diversität: nicht mehr als 2 aus derselben Familie)\n"
    "top3_candidates = df_ok.sort_values('mean PR-AUC', ascending=False)\n"
    "top3 = []\n"
    "family_count: dict = {}\n"
    "for _, r in top3_candidates.iterrows():\n"
    "    fam = r['Familie']\n"
    "    if family_count.get(fam, 0) < 2:\n"
    "        top3.append(r['Modell'])\n"
    "        family_count[fam] = family_count.get(fam, 0) + 1\n"
    "    if len(top3) == 3:\n"
    "        break"
)
new_filter = (
    "# Top-3 nach PR-AUC (Diversität: max. 1 Boosting-Modell, Rest aus anderen Familien)\n"
    "# Ziel: bestes Boosting + bestes lineares Modell + beste dritte Familie\n"
    "top3_candidates = df_ok.sort_values('mean PR-AUC', ascending=False)\n"
    "top3 = []\n"
    "boosting_added = False\n"
    "family_seen: dict = {}\n"
    "for _, r in top3_candidates.iterrows():\n"
    "    fam = r['Familie']\n"
    "    if fam == 'Boosting':\n"
    "        if not boosting_added:\n"
    "            top3.append(r['Modell'])\n"
    "            boosting_added = True\n"
    "    else:\n"
    "        if family_seen.get(fam, 0) < 1:\n"
    "            top3.append(r['Modell'])\n"
    "            family_seen[fam] = 1\n"
    "    if len(top3) == 3:\n"
    "        break"
)
assert old_filter in src, f"Filter pattern not found. Snippet:\n{src[src.find('Top-3 nach'):src.find('Top-3 nach')+500]}"
src = src.replace(old_filter, new_filter)
print("Fix 2 (Diversity filter sharpened): OK")

# Write back, clear outputs
c24['source'] = list(src)  # store as single string in list (valid ipynb)
c24['outputs'] = []
c24['execution_count'] = None

# ---------------------------------------------------------------
# 3. Insert new interpretation markdown cell after md-compare-interp (index 25)
# ---------------------------------------------------------------
assert cells[25]['id'] == 'md-compare-interp', cells[25]['id']

new_md_source = (
    "**Einordnung der Ergebnisse nach Sektion 8:**\n"
    "\n"
    "Die vier besten Modelle (LightGBM 0,4327 / HistGradBoost 0,4322 / MLP 0,4277 / LogReg 0,4229) liegen "
    "innerhalb eines Bandes von ca. 0,01 PR-AUC-Einheiten. Bei einer CV-Streuung von std ≈ 0,005–"
    "0,007 pro Fold überlappen die Konfidenzintervalle stark — **aktuell ist kein Modell statistisch "
    "signifikant besser als ein anderes**. Die endgültige Auswahl kann erst nach Hyperparameter-Tuning "
    "in NB07 gesichert werden.\n"
    "\n"
    "Bagging (Random Forest 0,3457 / Extra Trees 0,3130) liegt trotz höchster Laufzeit (188 s / 237 s) "
    "deutlich zurück. Das bestätigt den in der Literatur bekannten Befund: Boosting schlägt Bagging auf "
    "strukturierten Tabellendaten mit Klassenungleichgewicht, weil sequentielles Residuallernen gezielter "
    "auf schwer klassifizierbare Samples fokussiert.\n"
    "\n"
    "CatBoost ist nach dem Fix (eval\\_metric entfernt) mit einem echten PR-AUC-Wert ausgewiesen und wird "
    "in der sortierten Tabelle eingeordnet. Der Diversitätsfilter begrenzt die Boosting-Familie auf einen "
    "Kandidaten — CatBoost gelangt also nur in die Top-3, falls es den besten Boosting-Wert erreicht."
)

new_md_cell = {
    'cell_type': 'markdown',
    'id': 'md-sec8-postfix',
    'metadata': {},
    'source': [new_md_source]
}

cells.insert(26, new_md_cell)
print("Fix 3 (new markdown cell inserted at index 26): OK")

# After insertion indices shift by 1
# code-summary is now at index 35
assert cells[35]['id'] == 'code-summary', f"Expected code-summary at 35, got {cells[35]['id']}"

# ---------------------------------------------------------------
# 4. Update Section 11 summary (code-summary, index 35)
# ---------------------------------------------------------------
c_sum = cells[35]
src_sum = ''.join(c_sum['source'])

# Update Top-3 diversity annotation
old_anm = (
    "    ('Top-3 Familien für NB07',                  top3_str,\n"
    "     'Diversitäts-Filter: max. 2 pro Familie — für Stacking-Ensemble'),"
)
new_anm = (
    "    ('Top-3 Familien für NB07',                  top3_str,\n"
    "     'Diversitäts-Filter: max. 1 Boosting + beste andere Familien — für Stacking'),"
)
if old_anm in src_sum:
    src_sum = src_sum.replace(old_anm, new_anm)
    print("Fix 4a (Top-3 annotation updated): OK")
else:
    print(f"WARNING: Top-3 annotation not found. Snippet:\n{src_sum[src_sum.find('Top-3 Familien'):src_sum.find('Top-3 Familien')+200]}")

# Update next steps block
old_steps = (
    "print(f'\\nNächste Schritte:')\n"
    "print('  - NB07: Hyperparameter-Tuning der Top-3 Familien:')\n"
    "print(f'          Kandidaten: {top3}')\n"
    "print('          LightGBM/XGBoost: num_leaves, min_child_samples, learning_rate,')\n"
    "print('                             n_estimators, subsample, colsample_bytree')\n"
    "print('          CatBoost:          depth, learning_rate, l2_leaf_reg, iterations')"
)
new_steps = (
    "print(f'\\nNächste Schritte:')\n"
    "print('  - NB07: Hyperparameter-Tuning der Top-3 Kandidaten:')\n"
    "print(f'          {top3}')\n"
    "print('          LightGBM: num_leaves, min_child_samples, learning_rate,')\n"
    "print('                    n_estimators, subsample, colsample_bytree')\n"
    "print('          MLP:      hidden_layer_sizes, alpha, learning_rate_init')\n"
    "print('          LogReg:   C, l1_ratio (ElasticNet)')"
)
if old_steps in src_sum:
    src_sum = src_sum.replace(old_steps, new_steps)
    print("Fix 4b (next steps updated): OK")
else:
    print(f"WARNING: next-steps pattern not found. Snippet around 'chste Schritte':")
    idx = src_sum.find('chste Schritte')
    if idx >= 0:
        print(repr(src_sum[max(0,idx-10):idx+350]))

c_sum['source'] = [src_sum]
c_sum['outputs'] = []
c_sum['execution_count'] = None

# ---------------------------------------------------------------
# Save
# ---------------------------------------------------------------
with open('notebooks/06_modeling.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("\nNotebook saved successfully.")
