import csv
import datetime
import json
from pathlib import Path
import numpy as np
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


# ---------------------------------------------------------------------------
# CANONICAL FEATURE ENGINEERING PIPELINE (NB05 §3-§8, hard-coded decisions)
# ---------------------------------------------------------------------------

# IV/MI-filter: 8 engineered features dropped in NB05 §7 (hard-coded decision)
_DROP_ENGINEERED = [
    "healthy_lifestyle", "mental_physical_burden", "ascvd_proxy",
    "healthcare_access_index", "ses_x_access",
    "HighBP_x_HighChol", "BMI_squared", "BMI_yj",
]
# Columns dropped from raw feature set (need AnyHealthcare/NoDocbcCost for
# healthcare_access_index before dropping; Sex has no discriminatory power)
_MODEL_DROP_COLS = ["AnyHealthcare", "NoDocbcCost", "Sex"]

# Final column order for bmi_encoding='raw' (matches X_train_enriched.parquet)
_BASE_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "GenHlth", "DiffWalk", "Age", "Education", "Income",
]


def build_enriched_features(df, bmi_encoding="raw", use_hurdle=True):
    """
    Applies the full NB05 feature-engineering pipeline to a raw 21-feature
    BRFSS DataFrame and returns the enriched feature matrix.

    With the defaults (bmi_encoding='raw', use_hurdle=True) the output is
    column- and value-identical to data/processed/X_train_enriched.parquet
    (202944 × 30 on the training split).

    Parameters
    ----------
    df : pd.DataFrame
        Raw 21-feature BRFSS split produced by NB03.
    bmi_encoding : {'raw', 'capped', 'categorical', 'raw_squared'}
        BMI representation placed in the final feature set.
        'raw'         – raw BMI column; interactions use raw BMI.  [default]
        'capped'      – BMI clipped to [18, 50]; interactions follow.
        'categorical' – 7-class WHO ordinal; interactions follow.
        'raw_squared' – raw BMI + BMI_squared (31 columns instead of 30).
    use_hurdle : bool
        True  – MentHlth/PhysHlth → _any + _days + _log (default).
        False – keep MentHlth/PhysHlth as raw count features.

    Returns
    -------
    pd.DataFrame  (index preserved from df)
    """
    X = df.copy()

    # ------------------------------------------------------------------
    # Intermediate BMI series – always needed by composites regardless of
    # which encoding ends up in the final set.
    # ------------------------------------------------------------------
    _bmi_capped = X["BMI"].clip(18, 50)
    _bmi_cat = pd.cut(
        X["BMI"],
        bins=[0, 18.5, 25, 30, 35, 40, 50, 999],
        labels=[0, 1, 2, 3, 4, 5, 6],
    ).astype(int)

    # ------------------------------------------------------------------
    # Hurdle encoding (or keep raw counts)
    # ------------------------------------------------------------------
    if use_hurdle:
        X["MentHlth_any"]  = (X["MentHlth"] > 0).astype(int)
        X["MentHlth_days"] = X["MentHlth"].astype(int)
        X["PhysHlth_any"]  = (X["PhysHlth"] > 0).astype(int)
        X["PhysHlth_days"] = X["PhysHlth"].astype(int)
        X = X.drop(columns=["MentHlth", "PhysHlth"])
        _phys_any = X["PhysHlth_any"]
        _ment_any = X["MentHlth_any"]
    else:
        _phys_any = (X["PhysHlth"] > 0).astype(int)
        _ment_any = (X["MentHlth"] > 0).astype(int)

    # ------------------------------------------------------------------
    # Drop columns that were removed from the feature set in NB05.
    # AnyHealthcare and NoDocbcCost are no longer needed after this point
    # (healthcare_access_index was dropped by IV/MI filter and is not built).
    # ------------------------------------------------------------------
    X = X.drop(columns=_MODEL_DROP_COLS)

    # ------------------------------------------------------------------
    # Composite features – 5 retained after IV/MI filter (NB05 §7)
    # ------------------------------------------------------------------
    X["cardio_comorbidity"] = (
        X["HighBP"] + X["HighChol"] + X["HeartDiseaseorAttack"]
        + X["Stroke"] + X["DiffWalk"]
    )
    # allostatic_load uses BMI_cat (ordinal 0-6) and binary health indicators
    X["allostatic_load"] = (
        X["HighBP"] + X["HighChol"] + _bmi_cat + _phys_any + _ment_any
    )
    # metsyn_proxy uses BMI_capped threshold at 30 (IDF obesity criterion)
    X["metsyn_proxy"] = (
        X["HighBP"] + X["HighChol"]
        + (_bmi_capped >= 30).astype(int) + X["HeartDiseaseorAttack"]
    )
    X["ses_index"]     = X["Education"] + X["Income"]
    # findrisc_lite uses BMI_cat (FINDRISC-adapted diabetes risk score)
    X["findrisc_lite"] = (
        _bmi_cat + X["Age"]
        + (1 - X["PhysActivity"]) + (1 - X["Fruits"]) + (1 - X["Veggies"])
    )

    # ------------------------------------------------------------------
    # BMI series used in interaction terms – depends on chosen encoding
    # ------------------------------------------------------------------
    if bmi_encoding == "raw":
        _bmi_inter = X["BMI"]
    elif bmi_encoding == "capped":
        _bmi_inter = _bmi_capped
    elif bmi_encoding == "categorical":
        _bmi_inter = _bmi_cat
    elif bmi_encoding == "raw_squared":
        _bmi_inter = X["BMI"]
    else:
        raise ValueError(
            f"Unknown bmi_encoding {bmi_encoding!r}. "
            "Choose from 'raw', 'capped', 'categorical', 'raw_squared'."
        )

    # ------------------------------------------------------------------
    # Interaction features – 3 retained after IV/MI filter
    # ------------------------------------------------------------------
    X["Age_x_GenHlth"] = X["Age"]  * X["GenHlth"]
    X["BMI_x_Age"]     = _bmi_inter * X["Age"]
    X["BMI_x_HighBP"]  = _bmi_inter * X["HighBP"]

    # ------------------------------------------------------------------
    # Log transforms (hurdle-path only)
    # ------------------------------------------------------------------
    if use_hurdle:
        X["MentHlth_log"] = np.log1p(X["MentHlth_days"])
        X["PhysHlth_log"] = np.log1p(X["PhysHlth_days"])

    # ------------------------------------------------------------------
    # BMI columns in the final feature set
    # ------------------------------------------------------------------
    if bmi_encoding == "raw":
        pass  # BMI already in X; no additional BMI column needed
    elif bmi_encoding == "capped":
        X["BMI_capped"] = _bmi_capped
        X = X.drop(columns=["BMI"])
    elif bmi_encoding == "categorical":
        X["BMI_cat"] = _bmi_cat
        X = X.drop(columns=["BMI"])
    elif bmi_encoding == "raw_squared":
        X["BMI_squared"] = X["BMI"] ** 2

    # ------------------------------------------------------------------
    # Select and order final columns
    # ------------------------------------------------------------------
    hurdle_cols = (
        ["MentHlth_any", "MentHlth_days", "PhysHlth_any", "PhysHlth_days"]
        if use_hurdle else ["MentHlth", "PhysHlth"]
    )
    composite_cols = [
        "cardio_comorbidity", "allostatic_load", "metsyn_proxy",
        "ses_index", "findrisc_lite",
    ]
    inter_log_cols = ["Age_x_GenHlth"]
    if use_hurdle:
        inter_log_cols += ["MentHlth_log", "PhysHlth_log"]

    if bmi_encoding == "raw":
        bmi_block = ["BMI", "BMI_x_Age", "BMI_x_HighBP"]
    elif bmi_encoding == "capped":
        bmi_block = ["BMI_capped", "BMI_x_Age", "BMI_x_HighBP"]
    elif bmi_encoding == "categorical":
        bmi_block = ["BMI_cat", "BMI_x_Age", "BMI_x_HighBP"]
    elif bmi_encoding == "raw_squared":
        bmi_block = ["BMI", "BMI_squared", "BMI_x_Age", "BMI_x_HighBP"]

    final_cols = _BASE_COLS + hurdle_cols + composite_cols + inter_log_cols + bmi_block
    return X[final_cols]


def log_result(results_path, notebook, model, pr_auc_mean, pr_auc_std,
               roc_auc_mean=None, feature_set="raw21", imbalance="class_weight",
               n_folds=5, seed=42, pr_auc_folds=None, roc_auc_folds=None):
    p = Path(results_path); p.parent.mkdir(parents=True, exist_ok=True)
    key = {"notebook": notebook, "model": model, "feature_set": feature_set,
           "imbalance": imbalance, "seed": seed}
    row = {"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           **key, "n_folds": n_folds,
           "pr_auc_mean": round(float(pr_auc_mean), 4),
           "pr_auc_std": round(float(pr_auc_std), 4),
           "roc_auc_mean": round(float(roc_auc_mean), 4) if roc_auc_mean is not None else None,
           "pr_auc_folds": json.dumps([round(float(v), 6) for v in pr_auc_folds]) if pr_auc_folds is not None else None,
           "roc_auc_folds": json.dumps([round(float(v), 6) for v in roc_auc_folds]) if roc_auc_folds is not None else None}
    df = pd.read_csv(p) if (p.exists() and p.stat().st_size > 0) else pd.DataFrame()
    if not df.empty:
        m = pd.Series(True, index=df.index)
        for k, v in key.items():
            m &= df[k].astype(str) == str(v)
        df = df[~m]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)
