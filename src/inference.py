"""Standalone inference for the final diabetes-risk model (NB08 hand-off).

Place this file at ``src/inference.py``.

It depends ONLY on numpy, pandas, joblib and lightgbm — NOT on the rest of the
project (no ``src.features``, no notebooks, no data). That is possible because
the model's only engineered feature is ``BMI_squared = BMI ** 2``. A third party
can therefore score raw 21-feature BRFSS rows using just the files in ``models/``:

    models/nb08_model_card.json        (feature order, threshold, metadata)
    models/nb08_final_lightgbm.txt     (version-robust native LightGBM booster)
    models/nb08_final_lightgbm.joblib  (fitted sklearn pipeline; convenient path)

Usage
-----
    import pandas as pd
    from src.inference import predict

    df = pd.read_csv("my_rows.csv")          # 21 raw BRFSS feature columns
    out = predict(df)                        # -> DataFrame[proba, label]

`predict` tries the joblib pipeline first and silently falls back to the native
text booster if joblib fails (e.g. a scikit-learn / LightGBM version mismatch on
another machine). The decision threshold defaults to the frozen operating point
stored in the model card; pass ``threshold=...`` to override.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _models_dir(models_dir=None) -> Path:
    """Resolve the models directory (default: <repo>/models next to src/)."""
    if models_dir is not None:
        return Path(models_dir)
    return Path(__file__).resolve().parent.parent / "models"


def load_card(models_dir=None) -> dict:
    """Load the model card (feature order, threshold, metrics, versions)."""
    with open(_models_dir(models_dir) / "nb08_model_card.json") as f:
        return json.load(f)


def _prepare(df_raw, card: dict) -> pd.DataFrame:
    """Validate raw inputs and build the exact column matrix the model expects."""
    df = pd.DataFrame(df_raw)
    missing = [c for c in card["raw_features_expected"] if c not in df.columns]
    if missing:
        raise ValueError(f"input is missing required raw features: {missing}")
    X = df.copy()
    # the only engineered column the model was trained on
    if "BMI_squared" in card["model_input_columns"] and "BMI_squared" not in X.columns:
        X["BMI_squared"] = X["BMI"] ** 2
    return X[card["model_input_columns"]]   # enforce exact order


def predict(df_raw, models_dir=None, threshold=None) -> pd.DataFrame:
    """Score raw 21-feature BRFSS rows.

    Returns a DataFrame with columns ``proba`` (P(diabetes)) and ``label``
    (1 if proba >= threshold else 0), indexed like the input.
    """
    mdir = _models_dir(models_dir)
    card = load_card(models_dir)
    thr = float(card["operating_threshold"]) if threshold is None else float(threshold)
    X = _prepare(df_raw, card)

    try:                                       # convenient path
        import joblib
        model = joblib.load(mdir / "nb08_final_lightgbm.joblib")
        proba = model.predict_proba(X)[:, 1]
    except Exception as exc:                   # version-robust fallback
        print(f"[inference] joblib path failed ({type(exc).__name__}); "
              f"using native LightGBM booster.")
        import lightgbm as lgb
        booster = lgb.Booster(model_file=str(mdir / "nb08_final_lightgbm.txt"))
        proba = booster.predict(X.values)

    proba = np.asarray(proba, dtype=float)
    return pd.DataFrame({"proba": proba, "label": (proba >= thr).astype(int)},
                        index=X.index)


if __name__ == "__main__":
    # tiny self-check: scores a single all-zero row (sanity only, not meaningful)
    card = load_card()
    demo = pd.DataFrame([{c: 0 for c in card["raw_features_expected"]}])
    demo["BMI"] = 28
    print(predict(demo))
