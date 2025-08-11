
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Tuple
import math
import pandas as pd
import numpy as np

from .config import load_cfg
from .registry import REGISTRY
from . import discover

def _latest_csv_for_symbol(out_dir: Path, symbol: str) -> Optional[Path]:
    # Only consider plain prediction CSVs like 000001_xxx.csv; skip calibration/metrics
    cands = sorted(out_dir.glob(f"{symbol}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    def _is_pred(p: Path) -> bool:
        name = p.name.lower()
        if "calibration" in name or name.endswith("_metrics.csv") or name.endswith("_metrics.json"):
            return False
        return True
    cands = [p for p in cands if _is_pred(p)]
    return cands[0] if cands else None

def describe_csv(csv_path: Path) -> dict:
    df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
    stats = df.describe().to_dict()
    tail = df.tail(5).to_dict(orient="split")
    return {"path": str(csv_path), "stats": stats, "tail": tail}

def _roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    order = np.argsort(y_score)
    y = y_true[order]
    n1 = y.sum()
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = np.arange(1, len(y)+1)
    rank_sum_pos = ranks[y == 1].sum()
    auc = (rank_sum_pos - n1*(n1+1)/2) / (n0*n1)
    return float(auc)

def _pick_prob_column(df: pd.DataFrame) -> pd.Series:
    cols = [c.lower() for c in df.columns]
    # prefer p_final, then p_sup
    if "p_final" in cols:
        return df.iloc[:, cols.index("p_final")]
    if "p_sup" in cols:
        return df.iloc[:, cols.index("p_sup")]
    # If looks like a calibration CSV, raise a friendly error
    if {"p_mean", "y_rate"}.issubset(set(cols)):
        raise ValueError("Got a calibration CSV (contains p_mean/y_rate). Please pass a prediction CSV like outputs/000001_xxx.csv.")
    # fallback: first numeric column
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        raise ValueError("No numeric prediction column found in CSV.")
    return num.iloc[:, 0]

def evaluate(cfg_path: str, csv_path: Optional[str]=None, out_dir: Optional[str]=None) -> Tuple[dict, Path]:
    discover()
    cfg = load_cfg(cfg_path)
    out_dir = Path(out_dir or cfg.run.out_dir)
    if csv_path is None:
        csv = _latest_csv_for_symbol(out_dir, cfg.data.symbol)
        if csv is None:
            raise FileNotFoundError(f"No CSV found for symbol {cfg.data.symbol} in {out_dir}")
        csv_path = str(csv)
    csv_p = Path(csv_path)

    pred = pd.read_csv(csv_p, parse_dates=[0], index_col=0)
    p = _pick_prob_column(pred)

    fetcher = REGISTRY.get("fetcher", cfg.data.fetcher)
    df = fetcher(cfg.data.symbol, cfg.data.start, cfg.data.end).sort_index()
    label_fn = REGISTRY.get("label", cfg.label.name)
    y = label_fn(df, cfg.label.params).reindex(df.index)

    al = pd.concat([p.rename("p"), y.rename("y")], axis=1).dropna()
    cov = len(al) / len(p) if len(p) else 0.0

    y_true = al["y"].astype(float).values
    y_score = al["p"].astype(float).values
    auc = _roc_auc_score(y_true, y_score)
    brier = float(np.mean((y_score - y_true)**2)) if len(al) else float("nan")

    bins = np.linspace(0, 1, 11)
    al["bin"] = np.digitize(al["p"].values, bins, right=True)
    cal = al.groupby("bin").agg(
        p_mean=("p", "mean"),
        y_rate=("y", "mean"),
        cnt=("p", "size")
    ).reset_index()
    cal_path = out_dir / f"{cfg.data.symbol}_calibration.csv"
    cal.to_csv(cal_path, index=False, encoding="utf-8" )

    png_path = out_dir / f"{cfg.data.symbol}_calibration.png"
    try:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot([0,1],[0,1], linestyle="--")
        plt.plot(cal["p_mean"], cal["y_rate"], marker="o")
        plt.xlabel("Predicted prob"); plt.ylabel("Empirical rate"); plt.title(f"Calibration {cfg.data.symbol}")
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        png = str(png_path)
    except Exception:
        png = None

    metrics = {
        "symbol": cfg.data.symbol,
        "csv": str(csv_p),
        "coverage": round(float(cov), 6),
        "auc": None if (isinstance(auc,float) and math.isnan(auc)) else round(float(auc), 6),
        "brier": None if (isinstance(brier,float) and math.isnan(brier)) else round(float(brier), 6),
        "calibration_csv": str(cal_path),
        "calibration_png": png
    }
    metrics_path = out_dir / f"{cfg.data.symbol}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics, metrics_path
