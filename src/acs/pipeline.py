from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from .config import Cfg
from .utils.logging import setup_logging, timeit
from .utils.cache import get_memory
from .registry import REGISTRY

def _fingerprint(*parts: List[str]) -> str:
    h = hashlib.md5()
    for p in parts:
        h.update(str(p).encode())
    return h.hexdigest()[:10]

@timeit("run")
def run(cfg: Cfg):
    log = setup_logging()
    out_dir = Path(cfg.run.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    mem = get_memory(cfg.run.cache_dir)

    # 1) fetch
    fetcher = REGISTRY.get("fetcher", cfg.data.fetcher)
    df = fetcher(cfg.data.symbol, cfg.data.start, cfg.data.end)
    df = df.sort_index()
    if cfg.run.fp32:
        for c in df.columns:
            if df[c].dtype.kind=='f':
                df[c] = df[c].astype("float32")

    # 2) features
    feat_frames = []
    for name in cfg.feature.names:
        fn = REGISTRY.get("feature", name)
        params = cfg.feature.params.get(name, {})
        feat_frames.append(fn(df, params))
    feat = pd.concat(feat_frames, axis=1)

    # 3) label (optional, for demonstration)
    label_fn = REGISTRY.get("label", cfg.label.name)
    y = label_fn(df, cfg.label.params).reindex(df.index)

    # 4) supervised prob (placeholder: map feature via logistic-esque transform for demo)
    ps = (feat.rank(pct=True).mean(axis=1)).astype("float32")

    # 5) calibrate
    cal = REGISTRY.get("calibrator", cfg.model.calibrator)
    ps_cal = cal(ps, y, cfg.model.params).astype("float32")

    # 6) blend with unsupervised
    blender = REGISTRY.get("blender", cfg.blend.name)
    p_final = blender(ps_cal, df, cfg.blend.params).clip(0,1)

    # 7) neutral fill for any remaining NaNs (e.g., first few warm-up days, tail labels)
    ps_cal = ps_cal.fillna(0.5)
    p_final = p_final.fillna(ps_cal).fillna(0.5)

    # 8) save outputs + manifest
    symbol = cfg.data.symbol
    tag = _fingerprint(symbol, cfg.model.calibrator, json.dumps(cfg.model.params, sort_keys=True))
    out_csv = out_dir / f"{symbol}_{tag}.csv"
    res = pd.DataFrame({
        "p_sup": ps_cal,
        "p_final": p_final,
    }, index=df.index)
    res.to_csv(out_csv, encoding="utf-8")

    manifest = {
        "symbol": symbol,
        "out_csv": str(out_csv),
        "cfg": json.loads(cfg.model_dump_json()),
    }
    (out_dir / f"{symbol}_{tag}.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(json.dumps({"event":"saved","csv":str(out_csv)}))
    return out_csv
