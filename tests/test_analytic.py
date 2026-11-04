"""Fits against exactly known distributions (independent of any GLD implementation).

The "sample" is the exact quantile grid Q_true((k + 0.5) / n), so no sampling noise enters and the
Kolmogorov distance measures the method and the implementation only. Thresholds are the measured
values of benchmarks/compare_analytic.py with a margin, and docs/validation.md has the full table.
"""
import math

import numpy as np
import pytest
import torch
from scipy import stats

from gldpy_gpu import GLD, gpu

F32, F64 = torch.float32, torch.float64
N = 20_000
PROBS = np.linspace(0.0005, 0.9995, 1999)

# Distributions the GLD contains exactly: only the discretization of the grid remains (~2.5e-5).
EXACT = [
    ("uniform", stats.uniform(-1, 2), "RS"),
    ("uniform", stats.uniform(-1, 2), "FMKL"),
    ("uniform", stats.uniform(-1, 2), "VSL"),
    ("logistic", stats.logistic(), "FMKL"),
    ("logistic", stats.logistic(), "VSL"),
    ("exponential", stats.expon(), "VSL"),
]
# Distributions it only approximates: the limit is the family's own approximation error.
APPROX = [
    ("normal", stats.norm(), 7e-4),
    ("student_t5", stats.t(5), 2.5e-3),
    ("gamma_2", stats.gamma(2), 1.8e-2),
    ("lognormal_0.25", stats.lognorm(0.25), 8e-3),
    ("beta_2_2", stats.beta(2, 2), 1.5e-3),
    ("beta_2_5", stats.beta(2, 5), 1.6e-2),
]


def grid(dist, n=N):
    return dist.ppf((np.arange(n) + 0.5) / n)


def ks(ptype, fit, dist, device="cpu"):
    """sup |F_fit(Q_true(p)) - p| in float64."""
    g = gpu.GLD(ptype, device=device)
    p = torch.as_tensor(np.asarray(fit, dtype=float), dtype=F64, device=device)
    x = torch.as_tensor(dist.ppf(PROBS), dtype=F64, device=device)
    return np.abs(g.CDF_num(x, p, xtol=1e-10).cpu().numpy() - PROBS).max()


@pytest.fixture(params=[F64, F32], ids=["f64", "f32"])
def dtype(request):
    return request.param


@pytest.mark.parametrize("method", ["fit_LMM", "fit_PM"])
@pytest.mark.parametrize("name, dist, ptype", EXACT, ids=[f"{n}_{t}" for n, _, t in EXACT])
def test_exact_members_are_recovered(name, dist, ptype, method, dtype, device):
    g = gpu.GLD(ptype, device=device)
    fit = getattr(g, method)(torch.as_tensor(grid(dist), dtype=dtype, device=device))
    assert ks(ptype, fit.cpu().numpy(), dist) < 1e-4


@pytest.mark.parametrize("method", ["fit_LMM", "fit_PM"])
@pytest.mark.parametrize("ptype", ["RS", "FMKL", "VSL"])
@pytest.mark.parametrize("name, dist, limit", APPROX, ids=[a[0] for a in APPROX])
def test_non_members_reach_the_family_limit(name, dist, limit, ptype, method, dtype, device):
    g = gpu.GLD(ptype, device=device)
    fit = getattr(g, method)(torch.as_tensor(grid(dist), dtype=dtype, device=device))
    assert ks(ptype, fit.cpu().numpy(), dist) < limit


@pytest.mark.slow
@pytest.mark.parametrize("name, dist, ptype", EXACT, ids=[f"{n}_{t}" for n, _, t in EXACT])
def test_ML_recovers_exact_members(name, dist, ptype, dtype, device):
    g = gpu.GLD(ptype, device=device)
    fit = g.fit_ML(torch.as_tensor(grid(dist, 5000), dtype=dtype, device=device))
    assert ks(ptype, fit.cpu().numpy(), dist) < 2e-3


def test_exponential_on_rs_reaches_its_limiting_shape(dtype, device):
    """Not a member of RS: it is the limit lambda_3 -> inf, lambda_4 -> 0 or the like."""
    g = gpu.GLD("RS", device=device)
    fit = g.fit_LMM(torch.as_tensor(grid(stats.expon()), dtype=dtype, device=device))
    assert ks("RS", fit.cpu().numpy(), stats.expon()) < 2e-3


def test_float32_and_float64_agree_on_the_fit_quality(ptype, device):
    dist = stats.norm()
    g = gpu.GLD(ptype, device=device)
    results = [ks(ptype, g.fit_LMM(torch.as_tensor(grid(dist), dtype=dt, device=device)).cpu().numpy(), dist) for dt in (F32, F64)]
    assert abs(results[0] - results[1]) < 1e-5


@pytest.mark.parametrize("name, dist, ptype", EXACT, ids=[f"{n}_{t}" for n, _, t in EXACT])
def test_numpy_class_recovers_exact_members(name, dist, ptype):
    kw = dict(disp_optimizer=False, disp_fit=False, test_gof=False)
    start = None if ptype == "VSL" else [0.5, 0.5]
    fit = GLD(ptype).fit_LMM(grid(dist), start, xtol=1e-7, **kw)
    assert ks(ptype, fit, dist) < 1e-4


# ---- Moments of the fit against the distribution's own


@pytest.mark.parametrize("ptype", ["FMKL", "VSL"])
@pytest.mark.parametrize(
    "dist, var_tol",
    [(stats.norm(), 2e-3), (stats.logistic(), 1e-3), (stats.beta(2, 2), 2e-3), (stats.gamma(2), 0.03)],
    ids=["normal", "logistic", "beta_2_2", "gamma_2"],
)
def test_fitted_mean_and_variance_match_the_analytic_values(dist, var_tol, ptype, device):
    fit = gpu.GLD(ptype, device=device).fit_LMM(torch.as_tensor(grid(dist), dtype=F64, device=device)).cpu().numpy()
    g = GLD(ptype)
    assert g.mean(fit) == pytest.approx(dist.mean(), abs=2e-3)
    assert g.var(fit) == pytest.approx(dist.var(), rel=var_tol)


# ---- Published values


def test_ramberg_schmeiser_normal_approximation():
    """Ramberg & Schmeiser (1974): N(0, 1) ~ RS(0, 0.1975, 0.1349, 0.1349), obtained by matching moments."""
    kw = dict(disp_optimizer=False, disp_fit=False, test_gof=False)
    fit = GLD("RS").fit_MM(grid(stats.norm(), 500_000), np.array([0.1, 0.1]), xtol=1e-7, **kw)
    np.testing.assert_allclose(fit, [0.0, 0.1975, 0.1349, 0.1349], atol=1e-3)


def test_logistic_is_fmkl_at_zero_exactly(dtype, device):
    g = gpu.GLD("FMKL", device=device)
    p = torch.as_tensor([0.0, 1.0, 0.0, 0.0], dtype=dtype, device=device)
    y = np.linspace(1e-6, 1 - 1e-6, 201)
    out = g.Q(torch.as_tensor(y, dtype=dtype, device=device), p).cpu().double().numpy()
    y32 = torch.as_tensor(y, dtype=dtype).double().numpy()
    np.testing.assert_allclose(out, np.log(y32) - np.log1p(-y32), atol=2e-5 if dtype == F32 else 1e-9)
    assert math.isclose(GLD("FMKL").var([0, 1, 0, 0]), math.pi**2 / 3, rel_tol=1e-9)
