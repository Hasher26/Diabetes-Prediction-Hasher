"""
Feature engineering for the diabetes-prediction project (NB05+).

All functions are row-wise deterministic — computed per row with no target and
no fitted statistic — so they carry no leakage and may be applied before the
train/val/test split.  The only learnable step (in-fold feature scaling) lives
in the evaluation pipeline, not here.

NB08 imports build_candidates from this module to apply the identical,
deterministic transform to the frozen test set.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# WHO BMI classification constants (TRS 894, 2000)
# 0 underweight, 1 normal, 2 overweight, 3 obese-I, 4 obese-II,
# 5 obese-III (40-50), 6 super-obese (>=50; WHO class III subdivided at 50).
# ---------------------------------------------------------------------------
WHO_BINS   = [0, 18.5, 25, 30, 35, 40, 50, 999]
WHO_LABELS = [0, 1, 2, 3, 4, 5, 6]


def _bmi_cat(bmi):
    """Map continuous BMI values to integer WHO category (0–6), left-closed [lo, hi)."""
    return pd.cut(bmi, bins=WHO_BINS, labels=WHO_LABELS, right=False).astype(int)

def add_bmi_features(X):
    """
    Add BMI re-encodings (additive — nothing dropped).

   BMI_squared : additive non-linearity capturing the convex (accelerating)
                  risk rise at high BMI (Tirosh et al. 2011, NEJM).
    BMI_capped  : clip to [18, 50] — fixed WHO clinical constants, NOT
                  data-derived percentiles, hence leakage-free and not an
                  in-fold step.
    BMI_cat     : integer WHO class (building block for composites).
    BMI_class_* : one-hot of WHO classes; normal weight (class 1) is the
                  reference category and is dropped.  Columns are reindexed
                  to a fixed expected set so the output is deterministic even
                  if an extreme BMI class is absent in a small fold.
    """
    X = X.copy()
    X["BMI_squared"] = X["BMI"] ** 2           # additive non-linearity (Tirosh 2011)
    X["BMI_capped"]  = X["BMI"].clip(18, 50)   # swap alternative (WHO TRS 894, fixed bounds)
    X["BMI_cat"]     = _bmi_cat(X["BMI"])      # swap alternative / composite building block
    # One-hot of WHO classes; class 1 (normal) dropped as reference.
    dummies = pd.get_dummies(X["BMI_cat"], prefix="BMI_class").astype(int)
    # Reindex to the fixed expected column set so output is deterministic
    # even when a class is absent in a small subset/fold.
    expected = [f"BMI_class_{i}" for i in WHO_LABELS if i != 1]  # 0,2,3,4,5,6
    dummies = dummies.reindex(columns=expected, fill_value=0)
    X = pd.concat([X, dummies], axis=1)
    return X


def add_hurdle_features(X):
    """
    Zero-inflation transforms for MentHlth and PhysHlth (Mullahy 1986).

    *_any : hurdle indicator — any bad days vs. none.
    *_log : log1p of the count — compresses right-skew for the linear model.
    """
    X = X.copy()
    for col in ["MentHlth", "PhysHlth"]:
        X[f"{col}_any"] = (X[col] > 0).astype(int)   # hurdle indicator
        X[f"{col}_log"] = np.log1p(X[col])            # log of the count
    return X


def add_composites(X):
    """
    Row-wise sums on the BRFSS-available subset. findrisc_lite, ada_risk_proxy
    and healthy_lifestyle adapt validated scores; comorbidity_count, ses_index
    and healthcare_access_index are rationale-based equal-weighted sums (not
    published scores).  Each is linearly redundant for Logistic Regression and
    marginal for trees — kept for screening and interpretive value.

    comorbidity_count       : cardiometabolic burden (Muhammad et al. 2025)
    findrisc_lite           : FINDRISC-adapted diabetes risk (Lindström & Tuomilehto 2003)
    ada_risk_proxy          : ADA Diabetes Risk Test subset (Bang et al. 2009)
    healthy_lifestyle       : AHA Life's Simple 7 behaviours (Lloyd-Jones et al. 2010)
    ses_index               : socioeconomic position (Agardh et al. 2011)
    healthcare_access_index : system-contact / diagnosis-probability proxy
                              (Andersen & Newman 1973)
    """
    X = X.copy()
    bmi_cat = X["BMI_cat"] if "BMI_cat" in X.columns else _bmi_cat(X["BMI"])
    X["comorbidity_count"] = (
        X["HighBP"] + X["HighChol"] + X["Stroke"] + X["HeartDiseaseorAttack"]
    )
    X["findrisc_lite"] = (
        bmi_cat + X["Age"]
        + (1 - X["PhysActivity"]) + (1 - X["Fruits"]) + (1 - X["Veggies"])
    )
    X["ada_risk_proxy"] = (
        X["Age"] + bmi_cat + X["HighBP"] + (1 - X["PhysActivity"]) + X["Sex"]
    )
    X["healthy_lifestyle"] = (
        X["PhysActivity"] + X["Fruits"] + X["Veggies"]
        + (1 - X["Smoker"]) + (1 - X["HvyAlcoholConsump"])
    )
    X["ses_index"] = X["Education"] + X["Income"]
    X["healthcare_access_index"] = (
        X["CholCheck"] + X["AnyHealthcare"] + (1 - X["NoDocbcCost"])
    )
    return X


def add_interactions(X):
    """
    Explicit products giving the linear model interaction structure that trees
    find natively (expected tree gain ≈ 0).

    BMI_x_HighBP      : obesity × hypertension synergy (Landsberg et al. 2013)
    BMI_x_Age         : age-modified adiposity effect (Janssen et al. 2005)
    Age_x_HighBP      : age-dependent hypertension effect
    HighBP_x_HighChol : metabolic-syndrome co-occurrence / logical AND (IDF 2006)
    """
    X = X.copy()
    X["BMI_x_HighBP"]      = X["BMI"] * X["HighBP"]
    X["BMI_x_Age"]         = X["BMI"] * X["Age"]
    X["Age_x_HighBP"]      = X["Age"] * X["HighBP"]
    X["HighBP_x_HighChol"] = X["HighBP"] * X["HighChol"]   # logical AND on binaries
    return X


def add_flags(X):
    """
    poor_health : fair/poor self-rated health (GenHlth >= 4).
    Validated predictor of morbidity/mortality (Idler & Benyamini 1997).
    Threshold dummy for the linear model; redundant for trees.
    """
    X = X.copy()
    X["poor_health"] = (X["GenHlth"] >= 4).astype(int)
    return X


def build_candidates(X):
    """
    Apply every feature-engineering block additively.

    Returns the raw 21 features + all engineered candidates — nothing dropped —
    so the selection step can choose any subset.  Output is deterministic and
    leakage-free (row-wise, no fitted statistics).
    """
    X = add_bmi_features(X)
    X = add_hurdle_features(X)
    X = add_composites(X)
    X = add_interactions(X)
    X = add_flags(X)
    return X
