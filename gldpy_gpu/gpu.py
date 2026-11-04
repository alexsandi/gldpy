"""Backend selector for `gldpy_gpu.gpu`: PyTorch when importable, otherwise NumPy with a warning.

`BACKEND` says which one is active ("torch" or "numpy"). The torch code lives in
`_torch_backend.py`, the fallback in `_numpy_fallback.py`.
"""
try:
    import torch  # noqa: F401
except (ImportError, OSError):  # OSError: torch is installed but its shared libraries fail to load
    from . import _numpy_fallback as _backend

    BACKEND = "numpy"
else:
    from . import _torch_backend as _backend

    BACKEND = "torch"

GLD = _backend.GLD

__all__ = ["GLD", "BACKEND"]


def __getattr__(name: str):
    """Forward everything else (helpers such as `compute_quantile`) to the active backend."""
    return getattr(_backend, name)
