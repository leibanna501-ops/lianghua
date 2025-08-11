import pandas as pd
from typing import Dict, Any
from ..registry import REGISTRY

@REGISTRY.register("feature","ma_converge")
def ma_converge(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    win_s = int(cfg.get("ma_short", 20))
    win_l = int(cfg.get("ma_long", 60))
    ma_s = df["close"].rolling(win_s, min_periods=max(5,win_s//4)).mean()
    ma_l = df["close"].rolling(win_l, min_periods=max(5,win_l//4)).mean()
    z = (ma_s - ma_l) / (ma_l.abs() + 1e-8)
    return pd.DataFrame({f"ma_converge_{win_s}_{win_l}": z.astype("float32")})
