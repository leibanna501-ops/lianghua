import re
import pandas as pd
from typing import Optional
from ..registry import REGISTRY
from ..utils.legacy import load_legacy
from ..utils.compat import call_legacy

def _norm_cn_symbol(sym: str) -> str:
    """Extract 6-digit code from inputs like '000001.SZ', 'sz000001', '000001'."""
    m = re.search(r"(\d{6})", str(sym))
    return m.group(1) if m else str(sym)

def _ak_to_std_cols(df: pd.DataFrame) -> pd.DataFrame:
    # Map common AkShare Chinese columns to standard OHLCV
    mapping = {
        '开盘': 'open', '最高': 'high', '最低': 'low', '收盘': 'close',
        '成交量': 'volume', '成交额': 'amount',
        'open': 'open','high': 'high','low': 'low','close': 'close',
        'volume':'volume','amount':'amount'
    }
    cols = {c: mapping.get(c, c) for c in df.columns}
    df = df.rename(columns=cols)
    # index by date
    for cand in ['日期','date','Date']:
        if cand in df.columns:
            df = df.set_index(cand)
            break
    # keep only needed
    keep = [c for c in ['open','high','low','close','volume'] if c in df.columns]
    return df[keep]

def _fetch_via_akshare(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    try:
        import akshare as ak
    except Exception as e:
        raise RuntimeError(f"需要 akshare 才能直接取数，请安装: pip install akshare; 原因: {e}")
    out_frames = []
    cur = start
    while cur <= end:
        ce = min(cur + pd.Timedelta(days=119), end)
        s8, e8 = cur.strftime('%Y%m%d'), ce.strftime('%Y%m%d')
        code = _norm_cn_symbol(symbol)
        df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=s8, end_date=e8, adjust='qfq')
        if df is None or df.empty:
            # 再试一次不复权
            df = ak.stock_zh_a_hist(symbol=code, period='daily', start_date=s8, end_date=e8, adjust='')
        if df is None or df.empty:
            raise RuntimeError(f"AkShare 返回空: {code} {s8}-{e8}")
        out_frames.append(_ak_to_std_cols(df))
        cur = ce + pd.Timedelta(days=1)
    res = pd.concat(out_frames).sort_index()
    # to numeric
    for c in res.columns:
        res[c] = pd.to_numeric(res[c], errors='coerce')
    return res

@REGISTRY.register("fetcher", "akshare")
def fetch_daily(symbol: str, start: Optional[str]=None, end: Optional[str]=None) -> pd.DataFrame:
    """Fetch daily OHLCV with robust symbol normalization.
    - First try user's legacy fetcher
    - If it raises/returns空, fallback to direct AkShare
    """
    legacy = load_legacy()
    s = pd.to_datetime(start) if start is not None else None
    e = pd.to_datetime(end) if end is not None else None
    if e is None:
        e = pd.Timestamp.today().normalize()
    if s is None:
        s = e - pd.Timedelta(days=365*5)

    # 1) try legacy
    if legacy and hasattr(legacy, "fetch_daily"):
        try:
            df = call_legacy(legacy.fetch_daily, symbol, s, e)
            if df is not None and not df.empty:
                df.columns = [c.lower() for c in df.columns]
                return df
        except Exception as ex:
            # fallthrough to direct akshare
            pass

    # 2) fallback: direct AkShare
    df = _fetch_via_akshare(symbol, s, e)
    df.columns = [c.lower() for c in df.columns]
    return df
