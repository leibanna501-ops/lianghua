import pandas as pd
from typing import Dict, Any
from ..registry import REGISTRY

@REGISTRY.register("label","weak_label_v1")
def weak_label_v1(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.Series:
    horizon = int(cfg.get("horizon", 5))
    th = float(cfg.get("threshold", 0.03))
    # Avoid FutureWarning by disabling fill_method explicitly
    fwd = df["close"].pct_change(horizon, fill_method=None).shift(-horizon)
    y = (fwd > th).astype("float32")
    y[fwd.isna()] = float("nan")
    return y
