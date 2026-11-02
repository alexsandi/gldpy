"""Sample-statistics helpers vs. independent NumPy references."""
import itertools
import math

import numpy as np
import pytest
import torch

from gldpy_gpu import gpu


def ref_lmoments(x):
    """Hosking's direct definition via probability-weighted moments."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    i = np.arange(1, n + 1)
    b0 = x.mean()
    b1 = np.sum((i - 1) / (n - 1) * x) / n
    b2 = np.sum((i - 1) * (i - 2) / ((n - 1) * (n - 2)) * x) / n
    b3 = np.sum((i - 1) * (i - 2) * (i - 3) / ((n - 1) * (n - 2) * (n - 3)) * x) / n
    l1, l2 = b0, 2 * b1 - b0
    l3 = 6 * b2 - 6 * b1 + b0
    l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l1, l2, l3 / l2, l4 / l2


@pytest.mark.parametrize("n", [4, 7, 100, 5000])
def test_sample_lmoments(device, n):
    x = np.random.default_rng(n).gamma(2.0, size=n)
    out = gpu.compute_sample_lmoments(torch.as_tensor(x, dtype=torch.float64, device=device))
    np.testing.assert_allclose([o.item() for o in out], ref_lmoments(x), rtol=1e-8)


def test_sample_lmoments_needs_four_points(device):
    with pytest.raises(ValueError):
        gpu.compute_sample_lmoments(torch.rand(3, device=device))


def test_sample_percentiles(device):
    x = np.random.default_rng(1).normal(size=2000)
    u = 0.1
    a1, a2, a3, a4 = (
        o.item()
        for o in gpu.compute_sample_percentiles(torch.as_tensor(x, dtype=torch.float64, device=device), u)
    )
    qu, q25, q50, q75, q1mu = np.quantile(x, [u, 0.25, 0.5, 0.75, 1 - u])
    exp = (q50, q1mu - qu, (q50 - qu) / (q1mu - q50), (q75 - q25) / (q1mu - qu))
    np.testing.assert_allclose([a1, a2, a3, a4], exp, rtol=1e-8)


@pytest.mark.parametrize("u", [0.0, 0.25, -0.1, 0.5])
def test_percentiles_reject_bad_u(u):
    with pytest.raises(ValueError):
        gpu.compute_sample_percentiles(torch.rand(50), u)
