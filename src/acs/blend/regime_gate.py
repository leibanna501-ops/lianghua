import pandas as pd
from typing import Dict, Any
from ..registry import REGISTRY
from ..utils.legacy import load_legacy
from ..utils.compat import call_legacy

@REGISTRY.register("blender","regime_gate_oscillator")
def regime_gate_oscillator(p_sup: pd.Series, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """
    Blend supervised prob with unsupervised oscillator prob from legacy.
    New option:
      - invert_sup: bool = False  # if True, use 1 - p_sup before blending
      - w_unsup: float = 0.3      # weight of unsupervised gate
    """
    legacy = load_legacy()
    p_sup = p_sup.astype(float).clip(0,1)
    if bool(cfg.get("invert_sup", False)):
        p_sup = 1.0 - p_sup

    if legacy and hasattr(legacy, "unsup_oscillator_prob"):
        try:
            p_unsup = pd.Series(call_legacy(legacy.unsup_oscillator_prob, df, **cfg), index=df.index).clip(0,1)
        except Exception:
            p_unsup = pd.Series(0.5, index=df.index)
    else:
        p_unsup = pd.Series(0.5, index=df.index)

    # neutral fill to avoid NaNs
    p_sup = p_sup.fillna(0.5)
    p_unsup = p_unsup.fillna(0.5)

    w = float(cfg.get("w_unsup", 0.3))
    return (1-w)*p_sup + w*p_unsup

@REGISTRY.register("blender","invert")
def invert(p_sup: pd.Series, df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    """Pure inversion blender: return 1 - p_sup."""
    return (1.0 - pd.Series(p_sup, index=df.index).astype(float)).clip(0, 1)
