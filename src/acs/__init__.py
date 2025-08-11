__all__ = ["registry", "config", "pipeline", "discover"]
__version__ = "0.1.0"

def discover():
    # import submodules to trigger @register
    from .io import fetchers as _
    from .features import ma_converge as _
    from .labeling import weak_label_v1 as _
    from .model import calibration as _
    from .blend import regime_gate as _
