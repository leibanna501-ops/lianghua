# -*- coding: utf-8 -*-
"""
acs_panel_unified.py  (v1.4.2)
- 修复网络错误：akshare 取数支持“分段抓取 + 自动重试（指数回退+抖动）”
- 概率校准：默认优先 Platt（在 logit 上做 LR，可外推到更小/更大概率），若样本不足再回退 Isotonic
- 解决“概率有地板”的问题：移除 Isotonic 的硬 clip 效应，尾部用温度缩放做轻度拉伸
- 新增：无监督“累积 vs. 出货”振荡器（oscillator），与监督概率加权混合（动态权重随行情 Regime）
- 新增：sell_flag（疑似“出货完成”）：概率低位 + 概率回撤深 + 出货压力强 + 跌破 VWAP/事件 AVWAP
- 评估面板沿用 v1.3.0，主图增加 sell_flag 高亮条
"""

import os, sys, glob, argparse, traceback, time, random, math
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import calendar

# Dash / Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output, State, no_update

try:
    from dash import ctx as dash_ctx
except Exception:
    dash_ctx = None
from dash import callback_context as cb_ctx

DEBUG_VERSION = "v1.4.2"

# ------------------------------
# 工具：CSV 搜索与读取
# ------------------------------
PREFER_KEYWORDS = ("acs_prob", "prob", "acs")

def list_csvs(root: str, code: Optional[str] = None) -> List[str]:
    if code and str(code).strip():
        pattern = os.path.join(root, "**", f"*{code}*.csv")
    else:
        pattern = os.path.join(root, "**", f"*acs*.csv")
    files = glob.glob(pattern, recursive=True)
    files = [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith("~")]
    return files

def _score_csv_path(path: str) -> Tuple[int, int, float]:
    base = os.path.basename(path).lower()
    kw_hits = sum(1 for kw in PREFER_KEYWORDS if kw in base)
    depth = -len(os.path.normpath(path).split(os.sep))
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0.0
    return (kw_hits, depth, mtime)

def pick_best_csv(files: List[str]) -> Optional[str]:
    if not files:
        return None
    return sorted(files, key=_score_csv_path, reverse=True)[0]

def read_csv_smart(path: str) -> pd.DataFrame:
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err

# ------------------------------
# 计算核心（取数+特征+概率）
# ------------------------------
try:
    import akshare as ak
except Exception:
    ak = None

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.isotonic import IsotonicRegression
    SK_OK = True
except Exception:
    SK_OK = False

EPS = 1e-12

def _to_float(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)

def _zscore(x: pd.Series, win: int) -> pd.Series:
    m = x.rolling(win, min_periods=max(10, win // 5)).mean()
    s = x.rolling(win, min_periods=max(10, win // 5)).std()
    return (x - m) / (s.replace(0, np.nan))

def _rolling_rank(x: pd.Series, win: int) -> pd.Series:
    def _rank_last(a: np.ndarray):
        if len(a) <= 1 or np.all(np.isnan(a)):
            return np.nan
        last = a[-1]
        arr = a[:-1]; arr = arr[~np.isnan(arr)]
        if len(arr) == 0: return np.nan
        return (arr < last).sum() / max(len(arr), 1)
    return x.rolling(win, min_periods=max(20, win // 4)).apply(_rank_last, raw=True)

def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / (b.replace(0, np.nan))

def _avwap(close: pd.Series, volume: pd.Series) -> pd.Series:
    cum_v = volume.cumsum()
    cum_pv = (close * volume).cumsum()
    return _safe_div(cum_pv, cum_v)

def _to_datestr_yyyymmdd(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return pd.Timestamp.today().strftime("%Y%m%d")
    try:
        ts = pd.to_datetime(x)
    except Exception:
        s = str(x)
        return s.replace("-", "").replace("/", "")
    return ts.strftime("%Y%m%d")

def _to_datestr_iso(x) -> str:
    try:
        return pd.to_datetime(x).strftime("%Y-%m-%d")
    except Exception:
        s = str(x).replace("/", "-")
        if len(s) == 8 and s.isdigit():
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"
        return s

# ---------- 取数（分段抓取 + 自动重试） ----------
def _date_chunks(start_date: pd.Timestamp, end_date: pd.Timestamp, span_days=120):
    cur = start_date
    while cur <= end_date:
        nxt = min(cur + pd.Timedelta(days=span_days - 1), end_date)
        yield cur, nxt
        cur = nxt + pd.Timedelta(days=1)

def fetch_daily(symbol: str, start, end, debug_log: list,
                chunk_days: int = 120, max_retries: int = 4, base_sleep: float = 0.8) -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("未安装 akshare，请先：pip install -U akshare")

    s = pd.to_datetime(start); e = pd.to_datetime(end)
    debug_log.append(f"[fetch] symbol={symbol}  span={_to_datestr_iso(s)}..{_to_datestr_iso(e)}  chunks={chunk_days}d")

    dfs = []
    for (cs, ce) in _date_chunks(s, e, chunk_days):
        s8, e8 = cs.strftime("%Y%m%d"), ce.strftime("%Y%m%d")
        last_err = None
        for k in range(1, max_retries + 1):
            try:
                df = ak.stock_zh_a_hist(symbol=symbol, period="daily", start_date=s8, end_date=e8, adjust="qfq")
                if df is None or df.empty:
                    raise RuntimeError(f"区间 {s8}-{e8} 返回空")
                dfs.append(df)
                debug_log.append(f"[fetch] ok {s8}-{e8}  rows={len(df)}  (try#{k})")
                break
            except Exception as e1:
                last_err = e1
                msg = repr(e1)
                if ("10054" in msg) or ("Connection aborted" in msg) or ("Read timed out" in msg) or ("Max retries exceeded" in msg):
                    sleep = base_sleep * (2 ** (k - 1)) * (1 + 0.2 * random.random())
                    debug_log.append(f"[fetch] retry {s8}-{e8}  try#{k}  sleep={sleep:.2f}s  err={str(e1)[:100]}")
                    time.sleep(sleep)
                    continue
                else:
                    raise
        else:
            raise RuntimeError(f"网络多次失败：{s8}-{e8}｜最后错误：{last_err}")

    df = pd.concat(dfs, ignore_index=True)

    # 列名兼容
    rename_map = {"日期": "date","开盘": "open","收盘": "close","最高": "high","最低": "low","成交量": "volume","成交额": "amount","均价": "avg","涨跌幅": "pct_chg"}
    for zh, en in rename_map.items():
        if zh in df.columns:
            df[en] = df[zh]

    keep = [c for c in ["date","open","high","low","close","volume","amount","avg"] if c in df.columns]
    if "date" not in keep:
        raise RuntimeError(f"取数后缺少日期列，返回列={list(df.columns)}")

    df = df[keep].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open","high","low","close","volume","amount","avg"]:
        if c in df.columns: df[c] = _to_float(df[c])

    df = df.sort_values("date").drop_duplicates("date").set_index("date")

    # volume/avg 兜底
    if (("volume" not in df.columns) or df["volume"].isna().all()) and ("avg" in df.columns):
        df["volume"] = _safe_div(df["amount"], df["avg"])
    if (("avg" not in df.columns) or df["avg"].isna().all()) and ("volume" in df.columns):
        df["avg"] = _safe_div(df["amount"], df["volume"])
    if "volume" not in df.columns or df["volume"].isna().all():
        df["volume"] = _safe_div(df["amount"], df["close"])

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open","high","low","close","amount","volume"])
    debug_log.append(f"[fetch] merged rows={len(df)}  cols={list(df.columns)}")
    return df

def build_features(
    df: pd.DataFrame,
    win_fast: int = 10,
    win_mid: int = 20,
    win_slow: int = 60,
    win_year: int = 250,
    kyle_win: int = 60,
    ud_ratio_win: int = 40,
    box_win: int = 40,
    evt_ret_th: float = 0.09,
    evt_pos_th: float = 0.70,
):
    out = df.copy().sort_index()
    open_, high, low, close, volume, amount = [out[c].astype(float) for c in ["open","high","low","close","volume","amount"]]
    ret = close.pct_change().fillna(0.0)
    out["ret"] = ret

    out["pos_1y"] = _rolling_rank(close, win_year)
    out["vol_20"] = close.pct_change().rolling(20).std()
    out["vol_drop"] = -_zscore(out["vol_20"], win_slow)

    ma_fast = close.rolling(win_fast).mean()
    ma_mid = close.rolling(win_mid).mean()
    ma_slow = close.rolling(win_slow).mean()
    out["ma_converge"] = -( _zscore((ma_fast - ma_mid).abs() / close, win_slow) + _zscore((ma_mid - ma_slow).abs() / close, win_slow) )
    out["slope_mid"] = (ma_mid - ma_mid.shift(win_fast)) / (ma_mid.shift(win_fast) + EPS)

    up_mask = close >= close.shift(1)
    up_median = amount.where(up_mask, np.nan).rolling(ud_ratio_win).median()
    dn_median = amount.where(~up_mask, np.nan).rolling(ud_ratio_win).median()
    out["ud_amt_ratio_rob"] = _safe_div(up_median, (dn_median + EPS))
    out["ud_amt_ratio_z"] = _zscore(np.log1p(out["ud_amt_ratio_rob"]), win_slow)

    tr_raw = (high - low).abs()
    tr_alt = pd.concat([(close - close.shift(1)).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    tr = tr_raw.where(tr_raw > tr_alt, tr_alt).fillna(tr_raw)
    dir_coeff = ((close - open_) / (tr + EPS)).clip(-3, 3)
    dir_coeff = np.tanh(dir_coeff)
    signed_vol = dir_coeff * volume
    out["signed_vol_z"] = _zscore(signed_vol, win_slow)

    amihud = (ret.abs() / (amount.replace(0, np.nan))).rolling(win_mid).mean()
    out["amihud_z"] = -_zscore(np.log(amihud.replace(0, np.nan)), win_slow)
    cov = ret.rolling(kyle_win).cov(signed_vol)
    var = signed_vol.rolling(kyle_win).var()
    out["kyle_lambda"] = cov / (var + EPS)
    out["kyle_z"] = -_zscore(out["kyle_lambda"], win_slow)

    vwap = _avwap(close, volume)
    out["vwap_dist"] = (close - vwap) / (vwap + EPS)
    out["box_width"] = (high.rolling(box_win).max() - low.rolling(box_win).min()) / (close + EPS)
    out["box_tight_z"] = -_zscore(out["box_width"], win_slow)

    evt = (close.pct_change() >= evt_ret_th) & (((close - low) / (high - low + EPS)) > evt_pos_th)
    seg = evt.astype(int).cumsum()
    days_since_evt = seg.groupby(seg).cumcount()
    days_since_evt[seg == 0] = np.nan
    pv_cum = (close * volume).groupby(seg).cumsum()
    v_cum = (volume).groupby(seg).cumsum()
    avwap_evt = pv_cum / (v_cum + EPS)
    avwap_evt[seg == 0] = np.nan
    out["below_avwap_evt"] = ((close - avwap_evt) / (avwap_evt + EPS)).clip(-1, 1)
    out["cooldown_penalty"] = -(((days_since_evt <= 20) & (out["below_avwap_evt"] < -0.01))).astype(float)

    rs = close / (close.rolling(60).mean() + EPS)
    out["rs_z"] = _zscore(rs, win_slow)

    ma200 = close.rolling(200).mean()
    out["mkt_ma200_slope"] = (ma200 - ma200.shift(20)) / (ma200.shift(20) + EPS)
    vol20 = ret.rolling(20).std()
    out["mkt_vol_z"] = _zscore(vol20, win_slow)
    reg_score = 1.2*out["mkt_ma200_slope"].fillna(0) + 0.6*(out["pos_1y"].fillna(0.5)-0.5) - 0.3*out["mkt_vol_z"].fillna(0)
    out["regime_prob"] = 1.0 / (1.0 + np.exp(-reg_score))

    atr = tr.rolling(14).mean()
    out["atr_14"] = atr

    features = [
        "pos_1y","vol_drop","ma_converge","slope_mid","box_tight_z","vwap_dist",
        "signed_vol_z","amihud_z","kyle_z","ud_amt_ratio_z","below_avwap_evt",
        "cooldown_penalty","rs_z","mkt_ma200_slope","mkt_vol_z","regime_prob"
    ]
    return out, features

# ---------- 前视指标 & 标签 ----------
def forward_window_stats(df: pd.DataFrame, horizon: int) -> Tuple[pd.Series, pd.Series]:
    high_fwd = df["high"].shift(-horizon).rolling(horizon, min_periods=1).max()
    low_fwd  = df["low"].shift(-horizon).rolling(horizon,  min_periods=1).min()
    ret_up = high_fwd / df["close"] - 1.0
    dd_min = low_fwd  / df["close"] - 1.0
    ret_up[high_fwd.isna()] = np.nan
    dd_min[low_fwd.isna()]  = np.nan
    return ret_up, dd_min

def build_weak_labels(df: pd.DataFrame, horizon: int = 20, min_up: float = 0.06, max_dd: float = 0.12) -> pd.DataFrame:
    ret_up, dd_min = forward_window_stats(df, horizon)
    close = df["close"]
    vol20 = close.pct_change().rolling(20).std()
    vol_ref = vol20.rolling(250, min_periods=40).median()
    vol_scale = (vol20 / (vol_ref + EPS)).clip(0.7, 1.5).fillna(1.0)

    pos_1y = _rolling_rank(close, 250)
    bull = (pos_1y >= 0.60).astype(float); bear = (pos_1y <= 0.40).astype(float)
    up_adj = 1.0 - 0.05*bull + 0.05*bear
    dd_adj = 1.0 - 0.10*bull + 0.10*bear

    min_up_s = (min_up * vol_scale * up_adj).astype(float)
    max_dd_s = (max_dd * vol_scale * dd_adj).astype(float)

    label = ((ret_up >= min_up_s) & (dd_min >= -max_dd_s)).astype(float)
    lab_df = pd.DataFrame({"label_weak": label, "ret_up": ret_up, "dd_min": dd_min}, index=df.index)
    return lab_df

# ---------- 无监督“累积 vs. 出货”振荡器 ----------
def _sigmoid(x): return 1.0 / (1.0 + np.exp(-x))

def unsup_oscillator_prob(feat_df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    Z = feat_df.replace([np.inf, -np.inf], np.nan)

    # 累积侧（与原 fallback 类似）
    acc = (
        0.9 * Z.get("signed_vol_z", 0).fillna(0)
        + 0.8 * Z.get("amihud_z", 0).fillna(0)
        + 0.8 * Z.get("kyle_z", 0).fillna(0)
        + 0.6 * Z.get("ud_amt_ratio_z", 0).fillna(0)
        + 0.5 * Z.get("box_tight_z", 0).fillna(0)
        + 0.4 * Z.get("ma_converge", 0).fillna(0)
        + 0.3 * Z.get("vol_drop", 0).fillna(0)
        + 0.2 * Z.get("vwap_dist", 0).fillna(0)
        + 0.2 * Z.get("slope_mid", 0).fillna(0)
        + 0.1 * Z.get("pos_1y", 0).fillna(0)
        + 0.2 * Z.get("regime_prob", 0).fillna(0)
        + 0.2 * Z.get("mkt_ma200_slope", 0).fillna(0)
    )

    # 出货侧（体现在：下跌日体量占优、价格跌破 VWAP/事件 AVWAP、RS 走弱、箱体变宽、signed_vol 为负等）
    dist = (
        0.9 * (-Z.get("signed_vol_z", 0).fillna(0))
        + 0.7 * (-Z.get("ud_amt_ratio_z", 0).fillna(0))
        + 0.6 * (-Z.get("rs_z", 0).fillna(0))
        + 0.5 * (-Z.get("box_tight_z", 0).fillna(0))
        + 0.4 * (-Z.get("vwap_dist", 0).fillna(0))
        + 0.4 * (-Z.get("below_avwap_evt", 0).fillna(0))
        + 0.3 * ( Z.get("mkt_vol_z", 0).fillna(0))
        + 0.2 * (-Z.get("ma_converge", 0).fillna(0))
    )

    # 标准化平衡一下两侧量纲
    acc_z = (acc - acc.mean()) / (acc.std() + EPS)
    dist_z = (dist - dist.mean()) / (dist.std() + EPS)

    osc = acc_z - dist_z
    prob_osc = pd.Series(_sigmoid(1.15 * osc), index=feat_df.index)  # 轻度增益，拉开尾部
    return prob_osc, dist_z

# ---------- 概率校准器（优先 Platt，必要时 Isotonic） ----------
def _logit(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.log(p / (1-p))

class PlattOnLogit:
    def __init__(self):
        self.lr = LogisticRegression(max_iter=1000, C=1e6)
    def fit(self, p, y):
        X = _logit(np.asarray(p)).reshape(-1,1)
        self.lr.fit(X, y.astype(int))
        return self
    def predict(self, p_new):
        X = _logit(np.asarray(p_new)).reshape(-1,1)
        return self.lr.predict_proba(X)[:,1]

def fit_calibrator(p_valid: np.ndarray, y_valid: np.ndarray, debug_log: list):
    best_name = "platt"
    cal_platt = PlattOnLogit().fit(p_valid, y_valid)
    p1 = cal_platt.predict(p_valid)
    brier_platt = float(np.mean((p1 - y_valid)**2))

    cal_iso = None; brier_iso = np.inf; floor_iso = None
    try:
        cal_iso = IsotonicRegression(out_of_bounds="clip").fit(p_valid, y_valid)
        p2 = cal_iso.predict(p_valid)
        brier_iso = float(np.mean((p2 - y_valid)**2))
        floor_iso = float(cal_iso.predict([p_valid.min()])[0])
    except Exception:
        pass

    # 决策：若 Isotonic 明显更好且没有“高地板”，用 Isotonic；否则用 Platt
    if (cal_iso is not None) and (brier_iso + 1e-4 < brier_platt) and (floor_iso is not None) and (floor_iso <= 0.15):
        debug_log.append(f"[calib] choose Isotonic  brier_iso={brier_iso:.6f}  floor≈{floor_iso:.3f}")
        return cal_iso, "isotonic"
    debug_log.append(f"[calib] choose Platt     brier_platt={brier_platt:.6f}")
    return cal_platt, "platt"

def temperature_stretch(p: pd.Series, T: float = 0.9) -> pd.Series:
    """尾部轻度拉伸（T<1 拉开，两端更接近 0/1；仅在最终输出上做小幅处理）"""
    z = _logit(np.clip(p.values, 1e-6, 1-1e-6)) / max(T, 1e-3)
    return pd.Series(_sigmoid(z), index=p.index)

# ---------- 监督学习概率（OOF+Refit+校准） ----------
def fit_probability(
    feat_df: pd.DataFrame,
    features: List[str],
    label: Optional[pd.Series],
    n_splits: int = 5,
    C: float = 1.5,
    seed: int = 42,
    min_col_coverage: float = 0.50,
    min_rows_oof: int = 200,
    min_rows_insample: int = 40,
    cv_gap: int = 5,
    debug_log: Optional[list] = None,
):
    debug_log = debug_log if debug_log is not None else []
    metrics = {"auc": None, "mode": "fallback", "calibration": "none"}

    X_all = feat_df[features].replace([np.inf, -np.inf], np.nan)
    cov = X_all.notna().mean()
    keep_cols = cov[cov >= min_col_coverage].index.tolist()
    if len(keep_cols) < 5:
        keep_cols = cov.sort_values(ascending=False).head(8).index.tolist()
    X_df = X_all[keep_cols].copy()

    mask_lbl = (label.notna() if label is not None else pd.Series(False, index=X_df.index))

    # 路径1：OOF + 校准 + Refit
    if SK_OK and mask_lbl.sum() >= min_rows_oof:
        idx_lbl = label.loc[mask_lbl].index
        X_lbl = X_df.loc[idx_lbl].values
        y_lbl = label.loc[idx_lbl].astype(int).values
        w_all = np.linspace(0.7, 1.0, len(idx_lbl))

        tscv = TimeSeriesSplit(n_splits=n_splits)
        proba_oof = pd.Series(np.nan, index=feat_df.index)

        for tr, te in tscv.split(X_lbl):
            if cv_gap and len(tr) > 0:
                te_start = te[0]
                tr = tr[tr < max(0, te_start - cv_gap)]
            if len(tr)==0 or len(te)==0: continue

            model = make_pipeline(
                SimpleImputer(strategy="median"),
                StandardScaler(),
                LogisticRegression(max_iter=300, C=C, class_weight="balanced", random_state=seed),
            )
            model.fit(X_lbl[tr], y_lbl[tr], logisticregression__sample_weight=w_all[tr])
            p = model.predict_proba(X_lbl[te])[:, 1]
            proba_oof.loc[idx_lbl[te]] = p

        valid_mask = proba_oof.loc[idx_lbl].notna().values
        y_valid = y_lbl[valid_mask]
        p_valid = proba_oof.loc[idx_lbl].values[valid_mask]
        if len(p_valid) < 50 or np.unique(y_valid).size < 2:
            debug_log.append("[calib] OOF 样本不足，跳过校准，直接 refit")
            calibrator, name = None, "none"
        else:
            calibrator, name = fit_calibrator(p_valid, y_valid, debug_log)
            proba_oof.loc[idx_lbl[valid_mask]] = (
                calibrator.predict(p_valid) if hasattr(calibrator, "predict") else calibrator.transform(p_valid)
            )

        try:
            auc = roc_auc_score(y_valid, proba_oof.loc[idx_lbl].values[valid_mask])
            metrics["auc"] = float(auc)
        except Exception:
            pass

        final_model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=300, C=C, class_weight="balanced", random_state=seed),
        )
        final_model.fit(X_lbl, y_lbl, logisticregression__sample_weight=w_all)
        proba_all = pd.Series(final_model.predict_proba(X_df.values)[:, 1], index=X_df.index)

        if calibrator is not None and name != "none":
            try:
                proba_all = pd.Series(calibrator.predict(proba_all.values), index=proba_all.index)
            except Exception:
                proba_all = pd.Series(calibrator.transform(proba_all.values), index=proba_all.index)

        metrics["mode"] = f"logreg_oof+refit+{name}[{len(keep_cols)}cols]"
        metrics["calibration"] = name
        return proba_all, metrics

    # 路径2：有限样本 in-sample
    if SK_OK and mask_lbl.sum() >= min_rows_insample:
        X_tr = X_df.loc[mask_lbl].values
        y_tr = label.loc[mask_lbl].astype(int).values
        w_all = np.linspace(0.8, 1.0, len(X_tr))
        model = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=300, C=C, class_weight="balanced", random_state=seed),
        )
        model.fit(X_tr, y_tr, logisticregression__sample_weight=w_all)
        proba_all = pd.Series(model.predict_proba(X_df.values)[:, 1], index=X_df.index)
        metrics["mode"] = f"logreg_insample[{len(keep_cols)}cols]"
        return proba_all, metrics

    # 路径3：无监督 fallback
    fallback, _ = unsup_oscillator_prob(feat_df)
    metrics["mode"] = "unsup_oscillator(fallback)"
    return fallback, metrics

# ------------------------------
# 可视化：主图（含 sell_flag 高亮）
# ------------------------------
def _hide_non_trading(fig, idx):
    trading = pd.DatetimeIndex(pd.to_datetime(idx).normalize().unique())
    if len(trading) == 0: return
    all_days = pd.date_range(trading.min(), trading.max(), freq="D")
    non_trading_days = all_days.difference(trading)
    fig.update_xaxes(rangebreaks=[dict(values=non_trading_days.tolist())])

def make_figure(
    df: pd.DataFrame,
    title: str,
    visible_set: set,
    separate_prob_panel: bool,
    show_volume: bool,
    smooth_win: int,
    threshold: float,
    date_start: Optional[str],
    date_end: Optional[str],
):
    data = df.copy()
    cols = {c.lower().strip(): c for c in data.columns}
    need = ["date","open","high","low","close","acs_prob"]
    missing = [c for c in need if c not in cols]
    if missing:
        raise ValueError(f"CSV 缺少必要列：{missing}；实际列={list(data.columns)}")
    data = data.rename(columns={cols[c]: c for c in need if c in cols})
    if "sell_flag" in cols: data = data.rename(columns={cols["sell_flag"]: "sell_flag"})
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").set_index("date")
    if date_start: data = data.loc[data.index >= pd.to_datetime(date_start)]
    if date_end:   data = data.loc[data.index <= pd.to_datetime(date_end)]
    if data.empty:
        fig = go.Figure(); fig.update_layout(title="所选时间段无数据"); return fig

    prob_raw = pd.to_numeric(data["acs_prob"], errors="coerce").clip(0, 1)
    prob = prob_raw.rolling(max(1, int(smooth_win)), min_periods=1).mean()

    ma_windows = [5, 10, 20, 30, 60, 120]
    ma_map = {w: prob.rolling(w, min_periods=max(1, w // 2)).mean() for w in ma_windows}

    rows = 1; row_heights = [1.0]; specs = [[{"secondary_y": True}]]
    if separate_prob_panel:
        rows = 2; row_heights = [0.66, 0.34]; specs.append([{"secondary_y": False}])
    if show_volume:
        rows += 1; row_heights = [0.58, 0.24, 0.18] if separate_prob_panel else [0.72, 0.28]; specs.append([{}])

    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=row_heights, vertical_spacing=0.04, specs=specs)

    fig.add_trace(
        go.Candlestick(x=data.index, open=data["open"], high=data["high"], low=data["low"], close=data["close"],
                       increasing=dict(line=dict(color="#2ca02c", width=0.8), fillcolor="#2ca02c"),
                       decreasing=dict(line=dict(color="#ef5350", width=0.8), fillcolor="#ef5350"),
                       name="K线", showlegend=False),
        row=1, col=1, secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=data.index, y=prob, name="吸筹概率", mode="lines",
                   line=dict(color="#5B8FF9", width=1.2),
                   hovertemplate="日期=%{x|%Y-%m-%d}<br>概率=%{y:.2f}<extra></extra>",
                   legendgroup="prob", visible=("吸筹概率" in visible_set)),
        row=1, col=1, secondary_y=True,
    )

    if separate_prob_panel:
        fig.add_trace(
            go.Scatter(x=data.index, y=prob, name="吸筹概率(面板)", mode="lines",
                       line=dict(color="#5B8FF9", width=0.8), opacity=0.5, legendgroup="prob",
                       visible=("吸筹概率" in visible_set)),
            row=2, col=1, secondary_y=False,
        )

    ma_palette = ["#5B8FF9","#61DDAA","#65789B","#F6BD16","#7262fd","#78D3F8"]
    for i, w in enumerate(ma_windows):
        name = f"MA{w}"
        trace = go.Scatter(x=data.index, y=ma_map[w], name=name, mode="lines",
                           line=dict(color=ma_palette[i % len(ma_palette)], width=1.0),
                           opacity=0.95, legendgroup="prob_ma", visible=(name in visible_set))
        if separate_prob_panel: fig.add_trace(trace, row=2, col=1, secondary_y=False)
        else:                   fig.add_trace(trace, row=1, col=1, secondary_y=True)

    if threshold is not None:
        hi = (prob >= float(threshold)).astype(int)
        if hi.any():
            idx = hi.index.to_list(); on=False; start_t=None
            for i, (t, v) in enumerate(hi.items()):
                if v==1 and not on: on=True; start_t=t
                if v==0 and on:
                    on=False
                    fig.add_vrect(x0=start_t, x1=idx[i-1], fillcolor="#FF6B6B", opacity=0.08, line_width=0, layer="below", row=1, col=1)
            if on: fig.add_vrect(x0=start_t, x1=idx[-1], fillcolor="#FF6B6B", opacity=0.08, line_width=0, layer="below", row=1, col=1)

    # 出货完成标记高亮（紫色）
    if "sell_flag" in data.columns:
        sf = data["sell_flag"].astype(int)
        if sf.any():
            idx = sf.index.to_list(); on=False; start_t=None
            for i, (t, v) in enumerate(sf.items()):
                if v==1 and not on: on=True; start_t=t
                if v==0 and on:
                    on=False
                    fig.add_vrect(x0=start_t, x1=idx[i-1], fillcolor="#9C27B0", opacity=0.10, line_width=0, layer="below", row=1, col=1)
            if on: fig.add_vrect(x0=start_t, x1=idx[-1], fillcolor="#9C27B0", opacity=0.10, line_width=0, layer="below", row=1, col=1)

    if show_volume:
        vol = (data["volume"] if "volume" in data.columns else (data.get("amount", pd.Series(index=data.index)) / data["close"]).replace([np.inf, -np.inf], np.nan).fillna(0.0))
        up = data["close"] >= data["open"]; colors = np.where(up, "#26a69a", "#ef5350")
        fig.add_trace(go.Bar(x=data.index, y=vol, name="成交量", marker_color=colors, opacity=0.85,
                             hovertemplate="日期=%{x|%Y-%m-%d}<br>量=%{y:.0f}<extra></extra>"),
                      row=rows, col=1)

    fig.update_layout(
        title=title, paper_bgcolor="#f7f9fb", plot_bgcolor="#f7f9fb",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=60, t=60, b=40),
    )
    for r in range(1, rows + 1): fig.update_yaxes(showgrid=True, gridcolor="#e6ecf2", row=r, col=1)
    fig.update_yaxes(title_text="价格", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="概率", row=1, col=1, secondary_y=True, range=[0, 1])
    if separate_prob_panel: fig.update_yaxes(title_text="概率(均线)", row=2, col=1, range=[0, 1])
    if show_volume: fig.update_yaxes(title_text="成交量", row=rows, col=1, rangemode="tozero")

    _hide_non_trading(fig, data.index)
    return fig

# ------------------------------
# 评估&回测（与 v1.3.0 一致，略微整理输出）
# ------------------------------
def make_eval_figure(df: pd.DataFrame, horizon: int, min_up: float, max_dd: float, thr: float, topk_pct: int, bins: int):
    data = df.copy()
    cols = {c.lower().strip(): c for c in data.columns}
    for c in ["date","open","high","low","close","acs_prob"]:
        assert c in cols, f"缺少列 {c}"
    data = data.rename(columns={cols[c]: c for c in cols})
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values("date").set_index("date")

    if "ret_up" not in data.columns or "dd_min" not in data.columns:
        ru, dd = forward_window_stats(data, horizon); data["ret_up"], data["dd_min"] = ru, dd
    if "label_weak" not in data.columns:
        lab_df = build_weak_labels(data, horizon=horizon, min_up=min_up, max_dd=max_dd)
        data["label_weak"] = lab_df["label_weak"]

    sub = data[["acs_prob","label_weak","ret_up","dd_min"]].dropna()
    if sub.empty or sub["label_weak"].nunique() < 2:
        fig = go.Figure(); fig.update_layout(title="样本不足以评估"); return fig, {"warn": True, "text": "样本不足"}

    p = sub["acs_prob"].clip(0,1); y = sub["label_weak"].astype(int)

    try:
        deciles = pd.qcut(p, q=min(10, max(3, bins)), labels=False, duplicates="drop") + 1
    except Exception:
        deciles = pd.cut(p, bins=min(10, max(3, bins)), labels=False, include_lowest=True) + 1
    df_dec = pd.DataFrame({"decile": deciles, "y": y})
    hit_by_dec = df_dec.groupby("decile")["y"].mean()
    cnt_by_dec = df_dec.groupby("decile")["y"].size()

    bin_edges = np.linspace(0, 1, max(5, bins)+1)
    bin_id = np.digitize(p, bin_edges, right=True)
    calib = []
    for b in range(1, len(bin_edges)):
        m = bin_id == b
        if m.sum() >= 10:
            calib.append((p[m].mean(), y[m].mean(), m.sum()))
    calib = pd.DataFrame(calib, columns=["p_mean","y_rate","n"]) if calib else pd.DataFrame(columns=["p_mean","y_rate","n"])

    thr_grid = np.round(np.linspace(0.40, 0.95, 12), 2)
    hr_list, n_list = [], []
    for t0 in thr_grid:
        m = p >= t0
        n = int(m.sum()); n_list.append(n)
        hr_list.append(float(y[m].mean()) if n>0 else np.nan)
    thr_df = pd.DataFrame({"thr": thr_grid, "hr": hr_list, "n": n_list})

    k = max(1, int(np.ceil(len(sub) * topk_pct / 100.0)))
    idx_topk = p.nlargest(k).index
    y_topk = y.loc[idx_topk]
    ru_topk = sub.loc[idx_topk, "ret_up"]; dd_topk = sub.loc[idx_topk, "dd_min"]
    hr_topk = float(y_topk.mean()) if len(y_topk) > 0 else np.nan

    ms = p >= float(thr)
    y_thr = y[ms]; ru_thr = sub.loc[ms, "ret_up"]; dd_thr = sub.loc[ms, "dd_min"]
    hr_thr = float(y_thr.mean()) if len(y_thr) > 0 else np.nan

    fig = make_subplots(rows=2, cols=2, specs=[[{}, {}], [{}, {"type":"table"}]],
                        subplot_titles=("概率分位命中率","可靠度（预测概率 vs 实际命中）","阈值扫描命中率","汇总表"))
    fig.add_trace(go.Bar(x=[str(i) for i in hit_by_dec.index], y=hit_by_dec.values, name="命中率"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[str(i) for i in cnt_by_dec.index], y=cnt_by_dec.values, name="样本数", mode="lines+markers", yaxis="y2"), row=1, col=1)
    if not calib.empty:
        fig.add_trace(go.Scatter(x=calib["p_mean"], y=calib["y_rate"], mode="markers+lines", name="经验命中率"), row=1, col=2)
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="完美校准", line=dict(dash="dash")), row=1, col=2)
    fig.add_trace(go.Scatter(x=thr_df["thr"], y=thr_df["hr"], mode="lines+markers", name="HR(≥阈值)"), row=2, col=1)
    fig.add_vline(x=float(thr), line_dash="dot", line_color="#FF6B6B", row=2, col=1)

    tbl = [
        ["指标", "Top-K%", "阈值(≥)"],
        ["样本数", f"{len(y_topk)}", f"{int(ms.sum())}"],
        ["命中率", f"{hr_topk:.3f}" if not np.isnan(hr_topk) else "—", f"{hr_thr:.3f}" if not np.isnan(hr_thr) else "—"],
        ["平均上行(ret_up)", f"{np.nanmean(ru_topk):.3f}" if len(ru_topk) else "—", f"{np.nanmean(ru_thr):.3f}" if len(ru_thr) else "—"],
        ["平均最差回撤(dd_min)", f"{np.nanmean(dd_topk):.3f}" if len(dd_topk) else "—", f"{np.nanmean(dd_thr):.3f}" if len(dd_thr) else "—"],
    ]
    fig.add_trace(go.Table(header=dict(values=tbl[0], fill_color="#f0f2f5"),
                           cells=dict(values=list(map(list, zip(*tbl[1:]))))), row=2, col=2)

    fig.update_layout(showlegend=False, paper_bgcolor="#f7f9fb", plot_bgcolor="#f7f9fb", margin=dict(l=50, r=40, t=60, b=40))
    return fig, {"warn": False, "n": int(len(sub)), "hr_topk": hr_topk, "hr_thr": hr_thr, "n_topk": int(len(y_topk)), "n_thr": int(ms.sum()), "auc": (roc_auc_score(y, p) if SK_OK and y.nunique()==2 else None)}

# ------------------------------
# 运算：取数 -> 特征 -> 监督概率 -> 与无监督混合 -> 保存
# ------------------------------
def compute_and_save(symbol: str, start, end, out_root: str, debug_log: list,
                     horizon: int = 20, min_up: float = 0.06, max_dd: float = 0.12, cv_gap: int = 5) -> Tuple[pd.DataFrame, str, dict]:
    if not os.path.isdir(out_root): os.makedirs(out_root, exist_ok=True)
    df = fetch_daily(symbol, start, end, debug_log)
    if df.empty: raise RuntimeError("取数失败或无数据")

    debug_log.append(f"[compute] features ...")
    feat_df, feat_cols = build_features(df)
    lab_df = build_weak_labels(df, horizon=horizon, min_up=min_up, max_dd=max_dd)

    debug_log.append(f"[compute] probability (sklearn={SK_OK}, gap={cv_gap}) ...")
    proba_sup, metrics = fit_probability(feat_df, feat_cols, lab_df["label_weak"], cv_gap=cv_gap, debug_log=debug_log)

    # 无监督振荡器（累积 vs 出货） & 动态混合
    proba_unsup, dist_z = unsup_oscillator_prob(feat_df)
    w_unsup = (0.25 + 0.25 * (1 - feat_df["regime_prob"].clip(0,1))).clip(0.15, 0.45)  # 熊时更依赖无监督
    proba_mixed = ((1 - w_unsup) * proba_sup + w_unsup * proba_unsup).clip(0,1)

    # 尾部轻度拉伸，避免“看起来总差一点到 0/1”
    proba_final = temperature_stretch(proba_mixed, T=0.9)

    # 出货完成（sell_flag）：概率在低位 + 概率自峰值回撤深 + 出货压力强 + 跌破 VWAP/事件 AVWAP
    p_sm = proba_final.rolling(3, min_periods=1).mean()
    p_peak_60 = p_sm.rolling(60, min_periods=10).max()
    p_dd = (p_peak_60 - p_sm).fillna(0)
    below_vwap = (feat_df["vwap_dist"] < 0) | (feat_df["below_avwap_evt"] < -0.01)
    sell_flag = ((p_sm < 0.20) & (p_dd > 0.40) & (dist_z > 0.50) & below_vwap).astype(int)

    res = feat_df.copy()
    res["acs_prob"] = proba_final
    res["label_weak"] = lab_df["label_weak"]
    res["ret_up"] = lab_df["ret_up"]
    res["dd_min"] = lab_df["dd_min"]
    res["sell_flag"] = sell_flag

    out = res.copy()
    out.reset_index(names="date", inplace=True)
    csv_path = os.path.join(out_root, f"{symbol}_acs_prob.csv")
    out.to_csv(csv_path, index=False, encoding="utf-8-sig")
    debug_log.append(f"[compute] saved csv -> {csv_path}")
    return res, csv_path, metrics

# ------------------------------
# Dash 应用（延续 v1.3.0 的布局与评估面板）
# ------------------------------
def build_app(title_default: str, root_dir: str):
    SIDEBAR_W = 440
    SIDEBAR_STYLE_OPEN = {"position":"fixed","top":"12px","bottom":"12px","left":"12px","width":f"{SIDEBAR_W}px","overflowY":"auto","maxHeight":"calc(100vh - 24px)","padding":"14px","background":"#ffffff","border":"1px solid #eaecef","borderRadius":"14px","boxShadow":"0 6px 22px rgba(0,0,0,0.08)","zIndex":9998}
    SIDEBAR_STYLE_CLOSED = {"display":"none"}
    MAIN_STYLE_OPEN = {"marginLeft": f"{SIDEBAR_W + 28}px", "padding":"10px", "transition":"margin-left .25s ease"}
    MAIN_STYLE_CLOSED = {"marginLeft":"12px","padding":"10px","transition":"margin-left .25s ease"}
    TOGGLE_BTN_BASE = {"position":"fixed","top":"12px","zIndex":10000,"border":"1px solid #eaecef","borderRadius":"999px","padding":"6px 12px","background":"#fff","boxShadow":"0 4px 14px rgba(0,0,0,.10)","cursor":"pointer","fontWeight":"600"}
    DROP_STYLE = {"flex":"1 1 0","minWidth":"92px","textAlign":"center"}
    ROW_FLEX = {"display":"flex","gap":"8px","alignItems":"center"}

    def _days_in_month(year: int, month: int) -> int:
        return calendar.monthrange(int(year), int(month))[1]
    def _ymd_to_iso(y,m,d) -> str:
        if not (y and m and d): return pd.Timestamp.today().strftime("%Y-%m-%d")
        y,m,d = int(y),int(m),int(d)
        d = min(d, _days_in_month(y,m))
        return f"{y:04d}-{m:02d}-{d:02d}"

    _now = pd.Timestamp.today()
    year_opts = [{"label": f"{y} 年", "value": y} for y in range(2005, int(_now.year)+1)]
    month_opts = [{"label": f"{m:02d} 月", "value": m} for m in range(1, 13)]
    start_y, start_m, start_d = 2019, 1, 1
    end_y, end_m, end_d = int(_now.year), int(_now.month), int(_now.day)
    start_day_opts = [{"label": f"{d:02d} 日", "value": d} for d in range(1, calendar.monthrange(start_y, start_m)[1] + 1)]
    end_day_opts   = [{"label": f"{d:02d} 日", "value": d} for d in range(1, calendar.monthrange(end_y, end_m)[1] + 1)]

    app = Dash(__name__)
    app.title = title_default

    app.layout = html.Div([
        html.Button("⟨ 收起面板", id="toggle-sidebar", style={**TOGGLE_BTN_BASE, "left": f"{SIDEBAR_W + 24}px"}),
        dcc.Store(id="sidebar-open", data=True),
        html.Div([
            html.H3("一键 ACS 面板", style={"marginBottom":"6px"}),
            html.Div(f"Debug {DEBUG_VERSION}", style={"fontSize":"11px","color":"#777","marginBottom":"10px"}),

            html.Label("数据根目录"),
            dcc.Input(id="root-dir", type="text", value=root_dir, style={"width":"100%","marginBottom":"8px"}),
            html.Label("标的代码（如 603533）"),
            dcc.Input(id="code-input", type="text", debounce=True, placeholder="输入后可直接‘运算’或‘搜索CSV’", style={"width":"100%","marginBottom":"8px"}),

            html.Label("日期范围"),
            html.Div([
                html.Div([
                    html.Div("开始", style={"fontSize":"12px","color":"#666","marginBottom":"4px"}),
                    html.Div([
                        dcc.Dropdown(id="start-year", options=year_opts, value=start_y, clearable=False, style=DROP_STYLE),
                        dcc.Dropdown(id="start-month", options=month_opts, value=start_m, clearable=False, style=DROP_STYLE),
                        dcc.Dropdown(id="start-day", options=start_day_opts, value=start_d, clearable=False, style=DROP_STYLE),
                    ], style={**ROW_FLEX, "marginBottom":"6px"}),
                ]),
                html.Div([
                    html.Div("结束", style={"fontSize":"12px","color":"#666","marginBottom":"4px"}),
                    html.Div([
                        dcc.Dropdown(id="end-year", options=year_opts, value=end_y, clearable=False, style=DROP_STYLE),
                        dcc.Dropdown(id="end-month", options=month_opts, value=end_m, clearable=False, style=DROP_STYLE),
                        dcc.Dropdown(id="end-day", options=end_day_opts, value=end_d, clearable=False, style=DROP_STYLE),
                    ], style=ROW_FLEX),
                ]),
            ], style={"background":"#fafcff","border":"1px solid #eef2f7","borderRadius":"10px","padding":"8px"}),

            html.Div(style={"height":"8px"}),
            html.Label("概率阈值高亮 (≥)"),
            dcc.Slider(id="thr", min=0.0, max=1.0, step=0.01, value=0.70, marks={0.0:"0.0",0.3:"0.3",0.5:"0.5",0.7:"0.7",1.0:"1.0"}),

            html.Hr(),
            html.H4("标签参数"),
            html.Div([
                html.Div("Horizon (天)", style={"fontSize":"12px","color":"#666"}),
                dcc.Slider(id="lbl-h", min=5, max=60, step=1, value=20, marks={5:"5",20:"20",40:"40",60:"60"}),
                html.Div("最小上行 min_up", style={"fontSize":"12px","color":"#666","marginTop":"6px"}),
                dcc.Slider(id="lbl-up", min=0.02, max=0.25, step=0.005, value=0.06, marks={0.02:"0.02",0.06:"0.06",0.12:"0.12",0.20:"0.20",0.25:"0.25"}),
                html.Div("最大回撤 max_dd", style={"fontSize":"12px","color":"#666","marginTop":"6px"}),
                dcc.Slider(id="lbl-dd", min=0.04, max=0.30, step=0.005, value=0.12, marks={0.04:"0.04",0.12:"0.12",0.20:"0.20",0.30:"0.30"}),
            ], style={"background":"#fafcff","border":"1px solid #eef2f7","borderRadius":"10px","padding":"8px"}),

            html.Hr(),
            html.H4("训练参数"),
            html.Div([
                html.Div("TimeSeriesSplit gap (天)", style={"fontSize":"12px","color":"#666"}),
                dcc.Slider(id="cv-gap", min=0, max=20, step=1, value=5, marks={0:"0",5:"5",10:"10",15:"15",20:"20"}),
            ], style={"background":"#fafcff","border":"1px solid #eef2f7","borderRadius":"10px","padding":"8px"}),

            html.Hr(),
            html.H4("评估参数"),
            html.Div([
                html.Div("Top-K（按概率前百分比）", style={"fontSize":"12px","color":"#666"}),
                dcc.Slider(id="topk", min=1, max=50, step=1, value=10, marks={1:"1%",10:"10%",20:"20%",30:"30%",50:"50%"}),
                html.Div("分箱数（分位/校准）", style={"fontSize":"12px","color":"#666","marginTop":"6px"}),
                dcc.Slider(id="bins", min=5, max=20, step=1, value=10, marks={5:"5",10:"10",15:"15",20:"20"}),
            ], style={"background":"#fafcff","border":"1px solid #eef2f7","borderRadius":"10px","padding":"8px"}),

            html.Hr(),
            html.H4("可视化"),
            dcc.Checklist(id="line-select",
                          options=[{"label":"吸筹概率","value":"吸筹概率"}] + [{"label":f"MA{w}","value":f"MA{w}"} for w in [5,10,20,30,60,120]],
                          value=["吸筹概率","MA20","MA60"], style={"marginBottom":"8px"}),
            dcc.Checklist(id="options",
                          options=[{"label":"单独概率面板","value":"separate"},{"label":"显示成交量","value":"showvol"}],
                          value=["showvol"], style={"marginBottom":"8px"}),
            html.Label("概率平滑窗口（天）"),
            dcc.Slider(id="smooth", min=1, max=12, step=1, value=3, marks={i:str(i) for i in [1,3,5,7,9,12]}),

            html.Hr(),
            html.Button("运算（取数+计算+存CSV）", id="btn-compute", n_clicks=0, style={"marginRight":"8px"}),
            html.Button("搜索CSV", id="btn-search", n_clicks=0, style={"marginRight":"8px"}),
            html.Button("载入所选文件", id="btn-load", n_clicks=0),

            html.Div(id="compute-msg", style={"color":"#444","marginTop":"6px"}),
            html.Div(id="search-msg", style={"marginTop":"6px","color":"#666"}),
            dcc.Dropdown(id="csv-dropdown", options=[], value=None, placeholder="先搜索再选择", style={"marginTop":"6px"}),

            html.Div(style={"height":"6px"}),
            html.Label("调试信息（最近一次操作）"),
            dcc.Textarea(id="debug-box", value="", style={"width":"100%","height":"160px","whiteSpace":"pre","fontFamily":"Menlo,Consolas,monospace","border":"1px solid #eef2f7","borderRadius":"10px","background":"#fbfdff"}),
            html.Div(id="load-msg", style={"marginTop":"8px","color":"#888"}),
            html.Hr(),
            html.Div(id="eval-metrics", style={"fontSize":"13px","color":"#333"}),
        ], id="sidebar", style=SIDEBAR_STYLE_OPEN),

        html.Div([
            dcc.Graph(id="chart", config={"scrollZoom": True}, style={"height":"56vh"}),
            dcc.Graph(id="eval-fig", config={"displayModeBar": True}, style={"height":"38vh","marginTop":"8px"}),
        ], id="main", style=MAIN_STYLE_OPEN),

        dcc.Store(id="df-store"),
        dcc.Store(id="csv-path-store"),
        dcc.Store(id="title-store", data=title_default),
    ])

    @app.callback(
        Output("sidebar", "style"), Output("main", "style"), Output("toggle-sidebar", "children"),
        Output("toggle-sidebar", "style"), Output("sidebar-open", "data"),
        Input("toggle-sidebar", "n_clicks"), State("sidebar-open", "data"), prevent_initial_call=False)
    def _toggle_sidebar(n_clicks, open_state):
        is_open = True if open_state is None else bool(open_state)
        if n_clicks: is_open = not is_open
        if is_open:
            sb_style = SIDEBAR_STYLE_OPEN.copy(); main_style = MAIN_STYLE_OPEN.copy()
            btn_style = TOGGLE_BTN_BASE.copy(); btn_style["left"] = f"{SIDEBAR_W + 24}px"; btn_text = "⟨ 收起面板"
        else:
            sb_style = SIDEBAR_STYLE_CLOSED.copy(); main_style = MAIN_STYLE_CLOSED.copy()
            btn_style = TOGGLE_BTN_BASE.copy(); btn_style["left"] = "12px"; btn_text = "☰ 打开面板"
        return sb_style, main_style, btn_text, btn_style, is_open

    @app.callback(
        Output("start-day","options"), Output("start-day","value"),
        Input("start-year","value"), Input("start-month","value"), State("start-day","value"), prevent_initial_call=False)
    def _upd_start_day(y,m,d_prev):
        n = calendar.monthrange(int(y), int(m))[1]
        opts = [{"label":f"{d:02d} 日","value":d} for d in range(1, n+1)]
        return opts, min(int(d_prev or 1), n)

    @app.callback(
        Output("end-day","options"), Output("end-day","value"),
        Input("end-year","value"), Input("end-month","value"), State("end-day","value"), prevent_initial_call=False)
    def _upd_end_day(y,m,d_prev):
        n = calendar.monthrange(int(y), int(m))[1]
        opts = [{"label":f"{d:02d} 日","value":d} for d in range(1, n+1)]
        return opts, min(int(d_prev or 1), n)

    @app.callback(
        Output("csv-dropdown","options"), Output("csv-dropdown","value"), Output("search-msg","children"),
        Input("btn-search","n_clicks"), State("code-input","value"), State("root-dir","value"), prevent_initial_call=True)
    def on_search(n_clicks, code, root_dir):
        root = root_dir or "."
        files = list_csvs(root, code)
        if not files: return [], None, f"未在 {root} 下找到匹配的 CSV。可先‘运算’生成，或留空代码后再搜。"
        options = [{"label": f"{os.path.basename(p)}  —  {p}", "value": p} for p in files]
        best = pick_best_csv(files)
        return options, best, f"找到 {len(files)} 个匹配；已预选：{os.path.basename(best)}"

    @app.callback(
        Output("df-store","data"), Output("csv-path-store","data"), Output("title-store","data"),
        Output("load-msg","children"), Output("compute-msg","children"), Output("debug-box","value"),
        Input("btn-load","n_clicks"), Input("btn-compute","n_clicks"),
        State("csv-dropdown","value"), State("code-input","value"), State("root-dir","value"),
        State("start-year","value"), State("start-month","value"), State("start-day","value"),
        State("end-year","value"), State("end-month","value"), State("end-day","value"),
        State("thr","value"), State("lbl-h","value"), State("lbl-up","value"), State("lbl-dd","value"), State("cv-gap","value"),
        prevent_initial_call=True)
    def on_load_or_compute(n_load, n_comp, csv_path, code, root_dir, sy,sm,sd, ey,em,ed, thr, lbl_h, lbl_up, lbl_dd, cv_gap):
        trig = None
        if dash_ctx is not None:
            try: trig = dash_ctx.triggered_id
            except Exception: trig = None
        if not trig: trig = (cb_ctx.triggered[0]["prop_id"].split(".")[0] if cb_ctx.triggered else None)

        start_iso = _ymd_to_iso(sy,sm,sd); end_iso = _ymd_to_iso(ey,em,ed)
        debug_lines = [f"[trigger] {trig}", f"[env] sklearn={SK_OK}  akshare={'ok' if ak else 'missing'}", f"[dates] start={start_iso}  end={end_iso}",
                       f"[label] h={lbl_h}  min_up={lbl_up}  max_dd={lbl_dd}", f"[train] cv_gap={cv_gap}"]

        root = root_dir or "."
        try:
            if trig == "btn-compute":
                if not code or not str(code).strip():
                    msg = "请输入代码再点击‘运算’。"; debug_lines.append("[error] " + msg)
                    return no_update, no_update, no_update, msg, msg, "\n".join(debug_lines)

                df_res, saved_path, metrics = compute_and_save(code.strip(), start_iso, end_iso, root, debug_lines,
                                                               horizon=int(lbl_h), min_up=float(lbl_up), max_dd=float(lbl_dd), cv_gap=int(cv_gap))
                data_json = df_res.reset_index().rename(columns={"index":"date"}).to_json(orient="split", date_format="iso")
                ttl = os.path.splitext(os.path.basename(saved_path))[0]
                auc_part = (f" | OOF AUC≈{metrics['auc']:.3f}" if metrics.get("auc") is not None else "")
                note = f"✅ 运算完成，已保存：{saved_path}｜模式：{metrics.get('mode','?')}｜校准：{metrics.get('calibration','none')}{auc_part}"
                debug_lines.append("[done] " + note)
                return data_json, saved_path, ttl, note, note, "\n".join(debug_lines)

            if trig == "btn-load":
                if not csv_path:
                    msg = "请选择一个CSV文件。"; debug_lines.append("[error] " + msg)
                    return no_update, no_update, no_update, msg, "", "\n".join(debug_lines)
                df = read_csv_smart(csv_path)
                cols = [c.lower().strip() for c in df.columns]
                for need in ["date","open","high","low","close","acs_prob"]:
                    if need not in cols:
                        msg = f"CSV 缺少列：{need}；实际列：{list(df.columns)}"; debug_lines.append("[error] " + msg)
                        return no_update, no_update, no_update, msg, "", "\n".join(debug_lines)
                data_json = df.to_json(orient="split", date_format="iso")
                ttl = os.path.splitext(os.path.basename(csv_path))[0]
                msg = f"已载入：{csv_path}"; debug_lines.append("[done] " + msg)
                return data_json, csv_path, ttl, msg, "", "\n".join(debug_lines)

            debug_lines.append("[warn] 未识别的触发源")
            return no_update, no_update, no_update, "", "", "\n".join(debug_lines)

        except Exception as e:
            err = f"操作失败：{e}"
            tb_last = traceback.format_exc().strip().splitlines()[-1]
            debug_lines.append("[except] " + err); debug_lines.append("[trace] " + tb_last)
            return no_update, no_update, no_update, err, err, "\n".join(debug_lines)

    @app.callback(
        Output("chart","figure"),
        Input("df-store","data"), Input("title-store","data"), Input("line-select","value"), Input("options","value"),
        Input("smooth","value"), Input("thr","value"),
        Input("start-year","value"), Input("start-month","value"), Input("start-day","value"),
        Input("end-year","value"),   Input("end-month","value"),   Input("end-day","value"))
    def _update_chart(df_json, title, lines, opts, smooth, thr, sy,sm,sd, ey,em,ed):
        if not df_json:
            fig = go.Figure(); fig.update_layout(title="请先‘运算’或‘载入 CSV’"); return fig
        df = pd.read_json(df_json, orient="split")
        visible = set(lines or [])
        separate = "separate" in (opts or [])
        showvol = "showvol" in (opts or [])
        start_iso = _ymd_to_iso(sy,sm,sd); end_iso = _ymd_to_iso(ey,em,ed)
        return make_figure(df, title or "吸筹概率可视化", visible, separate, showvol, smooth, thr, start_iso, end_iso)

    @app.callback(
        Output("eval-fig","figure"), Output("eval-metrics","children"),
        Input("df-store","data"), Input("thr","value"), Input("topk","value"), Input("bins","value"),
        Input("lbl-h","value"), Input("lbl-up","value"), Input("lbl-dd","value"))
    def _update_eval(df_json, thr, topk, bins, lbl_h, lbl_up, lbl_dd):
        if not df_json:
            fig = go.Figure(); fig.update_layout(title="请先‘运算’或‘载入 CSV’"); return fig, ""
        df = pd.read_json(df_json, orient="split")
        fig, stat = make_eval_figure(df, horizon=int(lbl_h), min_up=float(lbl_up), max_dd=float(lbl_dd), thr=float(thr), topk_pct=int(topk), bins=int(bins))
        if stat.get("warn"): return fig, "样本不足以评估。"
        auc_text = f"｜AUC≈{stat['auc']:.3f}" if stat.get("auc") is not None else ""
        meta = f"评估样本数={stat['n']}｜Top-{topk}%：HR={stat['hr_topk']:.3f}（{stat['n_topk']} 条）｜阈值≥{thr:.2f}：HR={stat['hr_thr']:.3f}（{stat['n_thr']} 条）{auc_text}"
        return fig, meta

    return app

# ------------------------------
# 入口
# ------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="搜索/输出根目录，默认当前目录")
    ap.add_argument("--title", default="ACS 一体化面板（取数+评估）")
    args = ap.parse_args()

    app = build_app(args.title, args.root)
    (getattr(app, "run", None) or app.run_server)(debug=False, port=8050)

if __name__ == "__main__":
    main()
