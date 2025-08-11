from typing import Callable, Dict, Any

class _Registry:
    def __init__(self):
        self._maps: Dict[str, Dict[str, Any]] = {
            "feature": {},
            "label": {},
            "calibrator": {},
            "blender": {},
            "fetcher": {},
        }
    def register(self, kind: str, name: str):
        def deco(obj: Callable):
            self._maps[kind][name] = obj
            return obj
        return deco
    def get(self, kind: str, name: str):
        return self._maps[kind][name]
    def names(self, kind: str):
        return list(self._maps[kind].keys())

REGISTRY = _Registry()
