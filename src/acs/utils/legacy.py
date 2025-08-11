from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Optional

_CACHED = None

def load_legacy() -> Optional[object]:
    """Load user's legacy acs_panel_unified_3.py from common locations."""
    global _CACHED
    if _CACHED is not None:
        return _CACHED
    candidates = [
        Path(__file__).resolve().parents[3] / "acs_panel_unified_3.py",  # project root
        Path.cwd() / "acs_panel_unified_3.py",                            # CWD
    ]
    for p in candidates:
        if p.exists():
            _CACHED = SourceFileLoader("acs_legacy", str(p)).load_module()
            return _CACHED
    return None
