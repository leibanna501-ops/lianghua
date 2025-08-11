import pandas as pd
from typing import Dict, Any
from ..registry import REGISTRY
from ..utils.legacy import load_legacy
from ..utils.compat import call_legacy
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

@REGISTRY.register("calibrator", "platt")
def platt(prob: pd.Series, y: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    """Pure Platt scaling using logistic regression.

    Drops NaNs for fitting to avoid sklearn errors, then writes calibrated
    values back to the original index. Falls back to identity if fitting fails.
    """
    out = prob.copy()
    mask = prob.notna() & y.notna()
    if mask.sum() == 0:
        return out.clip(0, 1)
    try:
        lr = LogisticRegression(**cfg)
        lr.fit(prob[mask].values.reshape(-1, 1), y[mask].values)
        out.loc[mask] = lr.predict_proba(prob[mask].values.reshape(-1, 1))[:, 1]
        return out.clip(0, 1)
    except Exception:
        return out.clip(0, 1)


@REGISTRY.register("calibrator", "isotonic")
def isotonic(prob: pd.Series, y: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    """Pure isotonic regression calibrator.

    Drops NaNs for fitting to avoid sklearn errors, then writes calibrated
    values back to the original index. Falls back to identity if fitting fails.
    """
    out = prob.copy()
    mask = prob.notna() & y.notna()
    if mask.sum() == 0:
        return out.clip(0, 1)
    try:
        iso = IsotonicRegression(out_of_bounds="clip", **cfg)
        iso.fit(prob[mask].values, y[mask].values)
        out.loc[mask] = iso.predict(prob[mask].values)
        return out.clip(0, 1)
    except Exception:
        return out.clip(0, 1)

@REGISTRY.register("calibrator","legacy_platt_isotonic")
def legacy_platt_isotonic(prob: pd.Series, y: pd.Series, cfg: Dict[str, Any]) -> pd.Series:
    """Fit-on-OOF calibrator via legacy fit_calibrator if available; else identity.
    - Drops rows where prob or y is NaN for training to avoid sklearn errors.
    - Writes calibrated values back to those rows; keeps others as original (or NaN).
    """
    legacy = load_legacy()
    # Default: start from original probability
    out = prob.copy()
    mask = prob.notna() & y.notna()
    if mask.sum() == 0:
        return out.clip(0,1)
    if legacy and hasattr(legacy, "fit_calibrator"):
        try:
            sub = pd.Series(
                call_legacy(legacy.fit_calibrator, prob[mask].values, y[mask].values, **cfg),
                index=prob.index[mask]
            )
            out.loc[mask] = sub.values
            return out.clip(0,1)
        except Exception:
            # fall back to identity
            return out.clip(0,1)
    return out.clip(0,1)
