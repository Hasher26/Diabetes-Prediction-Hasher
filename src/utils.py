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
