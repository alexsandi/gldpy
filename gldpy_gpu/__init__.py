"""
Python package that implements tools for using Generalized Lambda Distribution (GLD) of three different parameterization types.

For more information, see gldpy.GLD.


"""
import importlib
from types import ModuleType
from ._gld_module import GLD

__version__ = "0.3.0"

__all__ = ["GLD", "gpu", "__version__"]

def _load_gpu_submodule(package_name: str) -> ModuleType:
    """Lazily import gldpy_gpu.gpu and verify PyTorch availability on first access."""
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "gldpy_gpu.gpu requires PyTorch. Install it via `pip install torch`."
        ) from exc
    return importlib.import_module(".gpu", package=package_name)


def __getattr__(name: str):
    """PEP 562 lazy attribute loader for `gldpy_gpu.gpu`."""
    if name == "gpu":
        gpu_mod = _load_gpu_submodule(__name__)
        globals()["gpu"] = gpu_mod  # Cache in module namespace after first load
        return gpu_mod
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + ["gpu"])
