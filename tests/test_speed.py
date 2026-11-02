"""Speed checks: opt-in with `pytest -m gpu_speed`. GPU must beat CPU on large batches."""
import time

import pytest
import torch

from gldpy_gpu import GLD as CpuGLD
from gldpy_gpu import gpu

pytestmark = [
    pytest.mark.gpu_speed,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
]

P = [0.0, 1.0, 0.2, 0.3]


def _time(fn, repeat=3):
    fn()  # warm-up (CUDA context, kernels, caches)
    best = float("inf")
    for _ in range(repeat):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def test_gpu_rand_faster_than_cpu_for_10M():
    n = 10_000_000
    cpu = CpuGLD("FMKL")
    g = gpu.GLD("FMKL", device=torch.device("cuda"))
    pt = torch.tensor(P, dtype=torch.float32, device="cuda")
    assert _time(lambda: g.rand(pt, size=n)) < _time(lambda: cpu.rand(P, size=n))


def test_gpu_cdf_faster_than_cpu_for_20k():
    import numpy as np

    n = 20_000
    cpu = CpuGLD("FMKL")
    x = cpu.rand(P, size=n, random_state=0)
    g = gpu.GLD("FMKL", device=torch.device("cuda"))
    xt = torch.as_tensor(x, dtype=torch.float32, device="cuda")
    pt = torch.tensor(P, dtype=torch.float32, device="cuda")
    assert _time(lambda: g.CDF_num(xt, pt), repeat=2) < _time(lambda: cpu.CDF_num(x, np.array(P)), repeat=1)
