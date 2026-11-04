"""
Python package that implements tools for using Generalized Lambda Distribution (GLD) of three different parameterization types.

For more information, see gldpy_gpu.GLD. `gldpy_gpu.gpu` has the PyTorch implementation.


"""
import importlib
from importlib import metadata
from ._gld_module import GLD

try:
    __version__ = metadata.version("gldpy-gpu")  # pyproject.toml is the only place that sets it
except metadata.PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"

__all__ = ["GLD", "gpu", "__version__"]

def __getattr__(name: str):
    """PEP 562 lazy loader, so `import gldpy_gpu` does not pay for importing torch."""
    if name == "gpu":
        gpu_mod = importlib.import_module(".gpu", package=__name__)
        globals()["gpu"] = gpu_mod
        return gpu_mod
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__():
    return sorted(list(globals().keys()) + ["gpu"])
