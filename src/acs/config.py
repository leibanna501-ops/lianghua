from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import yaml

class DataCfg(BaseModel):
    symbol: str
    start: Optional[str] = None
    end: Optional[str] = None
    fetcher: str = "akshare"

class FeatureCfg(BaseModel):
    names: List[str] = Field(default_factory=lambda: ["ma_converge"])
    params: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

class LabelCfg(BaseModel):
    name: str = "weak_label_v1"
    params: Dict[str, Any] = Field(default_factory=dict)

class ModelCfg(BaseModel):
    # your probability model/calibrator choices
    calibrator: str = "legacy_platt_isotonic"
    params: Dict[str, Any] = Field(default_factory=dict)

class BlendCfg(BaseModel):
    name: str = "regime_gate_oscillator"
    params: Dict[str, Any] = Field(default_factory=dict)

class RunCfg(BaseModel):
    cache_dir: str = ".cache"
    out_dir: str = "outputs"
    seed: int = 42
    fp32: bool = True

class Cfg(BaseModel):
    data: DataCfg
    feature: FeatureCfg = FeatureCfg()
    label: LabelCfg = LabelCfg()
    model: ModelCfg = ModelCfg()
    blend: BlendCfg = BlendCfg()
    run: RunCfg = RunCfg()

def load_cfg(path: str) -> Cfg:
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return Cfg(**d)
