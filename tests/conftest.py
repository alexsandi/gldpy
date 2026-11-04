import importlib.util

import numpy as np
import pytest

# Without torch, only the NumPy-side tests can run (see test_optional_torch.py).
HAS_TORCH = importlib.util.find_spec("torch") is not None
collect_ignore_glob = [] if HAS_TORCH else ["test_gpu_*.py", "test_speed.py", "test_analytic.py"]
torch = None
if HAS_TORCH:
    import torch

# Valid, representative parameters for each parameterization.
PARAMS = {
    "FMKL": [0.0, 1.0, 0.2, 0.3],
    "RS": [0.0, 0.2, 0.1, 0.15],
    "VSL": [0.0, 1.0, 0.4, 0.3],
}


def _devices():
    devs = ["cpu"]
    if HAS_TORCH and torch.cuda.is_available():
        devs.append("cuda")
    return devs


@pytest.fixture(params=_devices())
def device(request):
    return torch.device(request.param)


@pytest.fixture(params=sorted(PARAMS))
def ptype(request):
    return request.param


@pytest.fixture
def param(ptype):
    return np.array(PARAMS[ptype])


@pytest.fixture
def cpu_gld(ptype):
    from gldpy_gpu import GLD

    return GLD(ptype)


@pytest.fixture
def gpu_gld(ptype, device):
    from gldpy_gpu import gpu

    return gpu.GLD(ptype, device=device)


@pytest.fixture
def sample(cpu_gld, param):
    """1000 reproducible draws from the reference CPU implementation."""
    return cpu_gld.rand(param, size=1000, random_state=123)
