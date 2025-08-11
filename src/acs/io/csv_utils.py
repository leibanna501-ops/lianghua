"""Utilities for discovering and reading CSV files.

Extracted from legacy `acs_panel_unified_3.py` so the project no longer
relies on that monolithic script. Provides helpers to search for CSV files
by keyword or code, pick the best candidate and read it with encoding
fallbacks.
"""

from __future__ import annotations

import glob
import os
from typing import List, Optional

import pandas as pd

PREFER_KEYWORDS = ("acs_prob", "prob", "acs")


def list_csvs(root: str, code: Optional[str] = None) -> List[str]:
    """Recursively list CSV files under *root*.

    If *code* is provided, only files containing the code are returned.
    Temporary files prefixed with ``~`` are ignored.
    """
    if code and str(code).strip():
        pattern = os.path.join(root, "**", f"*{code}*.csv")
    else:
        pattern = os.path.join(root, "**", "*acs*.csv")
    files = glob.glob(pattern, recursive=True)
    return [f for f in files if os.path.isfile(f) and not os.path.basename(f).startswith("~")]


def _score_csv_path(path: str) -> tuple[int, int, float]:
    base = os.path.basename(path).lower()
    kw_hits = sum(1 for kw in PREFER_KEYWORDS if kw in base)
    depth = -len(os.path.normpath(path).split(os.sep))
    try:
        mtime = os.path.getmtime(path)
    except Exception:
        mtime = 0.0
    return kw_hits, depth, mtime


def pick_best_csv(files: List[str]) -> Optional[str]:
    """Pick the best CSV from *files* using keyword hits, path depth and mtime."""
    if not files:
        return None
    return sorted(files, key=_score_csv_path, reverse=True)[0]


def read_csv_smart(path: str) -> pd.DataFrame:
    """Read CSV with a set of common encodings.

    Tries UTF-8 variants first, then falls back to GBK/ANSI to gracefully
    handle files exported from Windows tools.
    """
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "utf-8", "gbk", "ansi"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:  # pragma: no cover - only executed on failure
            last_err = e
    assert last_err is not None
    raise last_err
