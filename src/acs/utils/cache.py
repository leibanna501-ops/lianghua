from joblib import Memory
from pathlib import Path

_MEM = None

def get_memory(cache_dir: str) -> Memory:
    global _MEM
    if _MEM is None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        _MEM = Memory(cache_dir, verbose=0)
    return _MEM
