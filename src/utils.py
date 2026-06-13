"""
Shared utilities for the diabetes-prediction project.

Deliberately minimal. This module holds ONLY the cross-cutting plumbing that
every notebook (NB04-NB08) must use identically:
  * make_cv      - the canonical cross-validation splitter (one source of truth)
  * log_result   - the central results ledger writer (outputs/results.csv)

All feature-engineering and modelling logic lives in the notebooks themselves,
so nothing in here can silently pre-decide feature selection. (The previous
IV/MI-filtered `build_enriched_features` pipeline was intentionally removed.)
"""

import datetime
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def make_cv(n_splits=5, seed=42):
    """
    Return the project's canonical cross-validation splitter.

    This is the ONE place the CV parameters are defined, so every notebook
    (NB05+) uses an identical splitter and the per-fold scores stay pairwise
    comparable across models and notebooks.

    StratifiedGroupKFold gives two guarantees at once:
      * group-aware - rows sharing an identical 21-feature profile stay in the
        same fold, so duplicate profiles cannot leak across the
        train/validation boundary;
      * stratified  - the ~14% positive prevalence is preserved in every fold.

    Parameters match NB04's baseline CV (n_splits=5, shuffle=True,
    random_state=42). Pass the SEED loaded from feature_meta.json at the call
    site, e.g. `make_cv(seed=SEED)`, and supply the SAME `groups` array to
    `cross_validate(..., groups=groups)`.
    """
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def log_result(results_path, notebook, model, pr_auc_mean, pr_auc_std,
               roc_auc_mean=None, feature_set="raw21", imbalance="class_weight",
               n_folds=5, seed=42, pr_auc_folds=None, roc_auc_folds=None):
    """
    Append (or overwrite) one result row in the shared ledger (results.csv).

    Rows are keyed by (notebook, model, feature_set, imbalance, seed): logging
    the same combination again OVERWRITES the previous row, so re-running a
    notebook never leaves duplicate entries.

    The per-fold lists are stored as JSON strings (one value per fold, in
    cross_validate order). Each value is iterated and cast to float, so the raw
    numpy arrays returned by cross_validate (e.g. scores["test_average_precision"])
    can be passed directly - no manual conversion needed. These fold lists are
    what enables the paired, fold-wise model comparisons in NB05+.
    """
    p = Path(results_path)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Identity of this result - used to de-duplicate on re-runs.
    key = {"notebook": notebook, "model": model, "feature_set": feature_set,
           "imbalance": imbalance, "seed": seed}

    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        **key,
        "n_folds": n_folds,
        "pr_auc_mean": round(float(pr_auc_mean), 4),
        "pr_auc_std": round(float(pr_auc_std), 4),
        "roc_auc_mean": round(float(roc_auc_mean), 4) if roc_auc_mean is not None else None,
        # iterate + float() so numpy arrays from cross_validate serialise cleanly
        "pr_auc_folds": json.dumps([round(float(v), 6) for v in pr_auc_folds])
        if pr_auc_folds is not None else None,
        "roc_auc_folds": json.dumps([round(float(v), 6) for v in roc_auc_folds])
        if roc_auc_folds is not None else None,
    }

    # Load the existing ledger (if any), drop the old row with the same key,
    # then append the new one. Last write wins per key.
    df = pd.read_csv(p) if (p.exists() and p.stat().st_size > 0) else pd.DataFrame()
    if not df.empty:
        mask = pd.Series(True, index=df.index)
        for k, v in key.items():
            mask &= df[k].astype(str) == str(v)
        df = df[~mask]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(p, index=False)


def paired_delta(a, b, sign_min=4, c_practical=0.005):
    """
    Paired per-fold comparison: a minus b (both must be same-length arrays).

    Returns a dict with per-fold deltas, mean Δ, std, standard error (se=sd/√n,
    optimistic under fold dependence — Bengio & Grandvalet 2004), sign count,
    and a retain flag requiring both the sign rule (n_pos >= sign_min) and the
    practical-significance bar (mean >= 2·se AND mean >= c_practical).

    sign_min     : folds where a > b needed to satisfy the sign rule (default 4/5).
    c_practical  : minimum mean Δ to clear the practical-significance bar.
    """
    d = np.asarray(a, float) - np.asarray(b, float)
    mean = float(d.mean()); sd = float(d.std(ddof=1)); se = sd / np.sqrt(len(d))
    n_pos = int((d > 0).sum())
    real  = (n_pos >= sign_min) and (mean >= 2 * se)
    worth = mean >= c_practical
    return {"deltas": d, "mean": mean, "sd": sd, "se": se, "n_pos": n_pos,
            "real": bool(real), "worth": bool(worth), "retain": bool(real and worth)}