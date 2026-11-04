"""Tail probabilities and float32/float64 behavior of the torch backend.

Oracle: GLD special cases with exact closed forms. The CPU class cannot serve here, since its
fminbound-based CDF_num is already wrong by a factor of 5 at tail probability 1e-6.
"""
import math

import numpy as np
import pytest
import torch

from gldpy_gpu import gpu

F64, F32 = torch.float64, torch.float32

# name: (ptype, param, sf(x), cdf(x), log_pdf(x), x grid reaching deep tails)
EXACT = {
    "logistic_FMKL": ("FMKL", [0.0, 1.0, 0.0, 0.0]),
    "logistic_VSL": ("VSL", [0.0, 2.0, 0.5, 0.0]),
}
TOL = {F64: 1e-5, F32: 2e-3}


def logistic_sf(x):
    return np.exp(-np.logaddexp(0.0, x))


def logistic_cdf(x):
    return np.exp(-np.logaddexp(0.0, -x))


def T(x, dtype, device):
    return torch.as_tensor(x, dtype=dtype, device=device)


@pytest.fixture(params=[F64, F32], ids=["f64", "f32"])
def dtype(request):
    return request.param


def deep(dtype, values_f64, values_f32):
    return np.array(values_f64 if dtype == F64 else values_f32, dtype=float)


@pytest.mark.parametrize("case", sorted(EXACT))
def test_logistic_both_tails_keep_relative_accuracy(case, dtype, device):
    ptype, param = EXACT[case]
    g = gpu.GLD(ptype, device=device)
    x = deep(dtype, [5.0, 20.0, 50.0, 200.0, 600.0], [5.0, 15.0, 30.0, 60.0])
    p = T(param, dtype, device)
    xt = T(x, dtype, device)
    xd = xt.cpu().double().numpy()
    np.testing.assert_allclose(g.SF_num(xt, p).cpu().double().numpy(), logistic_sf(xd), rtol=TOL[dtype])
    np.testing.assert_allclose(g.CDF_num(-xt, p).cpu().double().numpy(), logistic_cdf(-xd), rtol=TOL[dtype])


def test_exponential_upper_tail(dtype, device):
    """VSL(0, 1, 1, 0) is Exp(1): Q = -log(1 - y), SF = exp(-x)."""
    g = gpu.GLD("VSL", device=device)
    x = T(deep(dtype, [1.0, 30.0, 300.0, 650.0], [1.0, 20.0, 60.0]), dtype, device)
    p = T([0.0, 1.0, 1.0, 0.0], dtype, device)
    xd = x.cpu().double().numpy()
    np.testing.assert_allclose(g.SF_num(x, p).cpu().double().numpy(), np.exp(-xd), rtol=TOL[dtype])


def test_pareto_upper_tail(dtype, device):
    """VSL(0, 1, 1, -1) is Pareto: Q = 1/(1 - y) - 1, SF = 1/(1 + x)."""
    g = gpu.GLD("VSL", device=device)
    x = T(deep(dtype, [1.0, 1e4, 1e10, 1e100], [1.0, 1e4, 1e10, 1e30]), dtype, device)
    p = T([0.0, 1.0, 1.0, -1.0], dtype, device)
    np.testing.assert_allclose(
        g.SF_num(x, p).cpu().double().numpy(), 1.0 / (1.0 + x.cpu().double().numpy()), rtol=TOL[dtype]
    )


def test_ISF_exact_on_exponential(dtype, device):
    g = gpu.GLD("VSL", device=device)
    s = deep(dtype, [1e-2, 1e-10, 1e-100, 1e-300], [1e-2, 1e-10, 1e-30])
    out = g.ISF(T(s, dtype, device), T([0.0, 1.0, 1.0, 0.0], dtype, device)).cpu().double().numpy()
    np.testing.assert_allclose(out, -np.log(s), rtol=1e-5 if dtype == F32 else 1e-12)


@pytest.mark.parametrize(
    "ptype, param",
    [
        ("FMKL", [0.0, 1.0, -0.2, -0.1]),
        ("FMKL", [1.0, 2.0, 0.2, 0.3]),
        ("RS", [0.0, -0.2, -0.1, -0.15]),
        ("RS", [0.0, 0.2, 0.1, 0.15]),
        ("VSL", [0.0, 1.0, 0.4, -0.2]),
        ("VSL", [0.0, 1.0, 0.4, 0.3]),
    ],
)
def test_tail_round_trip_every_parameterization(ptype, param, dtype, device):
    g = gpu.GLD(ptype, device=device)
    p = T(param, dtype, device)
    s = T(deep(dtype, [1e-3, 1e-6, 1e-9, 1e-12], [1e-3, 1e-5]), dtype, device)
    rtol = 1e-3 if dtype == F64 else 0.05  # bounded tails are ill-conditioned: x ~ sup - s**lam
    torch.testing.assert_close(g.SF_num(g.ISF(s, p), p), s, rtol=rtol, atol=0.0)
    torch.testing.assert_close(g.CDF_num(g.Q(s, p), p), s, rtol=rtol, atol=0.0)


def test_one_far_point_does_not_change_the_others(device):
    g = gpu.GLD("FMKL", device=device)
    p = T([0.0, 1.0, -0.2, -0.1], F64, device)
    near = T([0.1, -3.0, 2.0], F64, device)
    alone = g.CDF_num(near, p)
    together = g.CDF_num(torch.cat([near, T([1e30, -1e30], F64, device)]), p)
    torch.testing.assert_close(together[:3], alone, rtol=1e-9, atol=0.0)


def test_nan_stays_local(device):
    g = gpu.GLD("FMKL", device=device)
    p = T([0.0, 1.0, 0.2, 0.3], F64, device)
    out = g.CDF_num(T([0.5, float("nan"), 0.1], F64, device), p)
    assert torch.isnan(out[1]) and torch.isfinite(out[[0, 2]]).all()


def test_log_PDF_exact_in_far_tail(dtype, device):
    """Logistic density = y(1-y). Clamping y used to report a constant density out here."""
    g = gpu.GLD("FMKL", device=device)
    x = deep(dtype, [10.0, 100.0, 500.0], [10.0, 40.0, 70.0])
    xt = T(x, dtype, device)
    xd = xt.cpu().double().numpy()
    truth = -xd - 2.0 * np.logaddexp(0.0, -xd)
    p = T([0.0, 1.0, 0.0, 0.0], dtype, device)
    np.testing.assert_allclose(g.log_PDF_num(xt, p).cpu().double().numpy(), truth, rtol=TOL[dtype])
    np.testing.assert_allclose(g.PDF_num(xt[:2], p).cpu().double().numpy(), np.exp(truth[:2]), rtol=0.01)


def test_log_PDF_minus_inf_outside_support(device):
    g = gpu.GLD("FMKL", device=device)
    p = T([0.0, 1.0, 0.5, 0.5], F64, device)
    lo, hi = g.supp(p)
    out = g.log_PDF_num(torch.stack([lo - 1.0, hi + 1.0]), p)
    assert torch.all(out == -math.inf)


def test_PDF_at_support_edge_matches_cpu(device):
    """FMKL(2, 2) is uniform on [-0.5, 0.5] with non-zero density at the edge."""
    from gldpy_gpu import GLD

    g = gpu.GLD("FMKL", device=device)
    p = [0.0, 1.0, 2.0, 2.0]
    lo, hi = GLD("FMKL").supp(p)
    edge = np.array([float(lo), float(hi)])
    out = g.PDF_num(T(edge, F64, device), T(p, F64, device)).cpu().numpy()
    np.testing.assert_allclose(out, GLD("FMKL").PDF_num(edge, p), rtol=1e-3)


@pytest.mark.parametrize("lam", [1e-3, 1e-5, 1e-6, 1e-7, 1e-9, 1e-12])
def test_near_zero_lambda_has_no_cancellation(lam, dtype, device):
    g = gpu.GLD("FMKL", device=device)
    y = np.array([0.01, 0.1, 0.9, 0.99])
    truth = (np.expm1(lam * np.log(y)) - np.expm1(lam * np.log1p(-y))) / lam
    out = g.Q(T(y, dtype, device), T([0.0, 1.0, lam, lam], dtype, device)).cpu().double().numpy()
    np.testing.assert_allclose(out, truth, rtol=1e-5 if dtype == F32 else 1e-13)


@pytest.mark.parametrize(
    "ptype, param",
    [
        ("FMKL", [0.0, 1.0, 0.0, 0.5]),
        ("FMKL", [0.0, 1.0, -0.5, 0.0]),
        ("RS", [0.0, 1.0, 0.0, 1.0]),
        ("RS", [0.0, 1.0, 1.0, 0.0]),
        ("RS", [0.0, -1.0, -0.5, -0.5]),
        ("VSL", [0.0, 1.0, 0.0, 0.5]),
        ("VSL", [0.0, 1.0, 1.0, 0.0]),
    ],
)
def test_Q_and_density_at_exact_endpoints_match_cpu(ptype, param, device):
    from gldpy_gpu import GLD

    cpu_gld = GLD(ptype)
    g = gpu.GLD(ptype, device=device)
    ends = np.array([0.0, 1.0])
    with np.errstate(all="ignore"):
        cq = cpu_gld.Q(ends, param)
    out = g.Q(T(ends, F64, device), T(param, F64, device)).cpu().numpy()
    np.testing.assert_array_equal(np.isnan(out), np.isnan(cq))
    np.testing.assert_allclose(out[~np.isnan(cq)], cq[~np.isnan(cq)], rtol=1e-12)
    assert not torch.isnan(g.PDF_Q(T(ends, F64, device), T(param, F64, device))).any()


# ---- dtype handling


def test_numpy_float64_input_is_not_downcast(device):
    g = gpu.GLD("FMKL", device=device)
    out = g.Q(np.array([1 - 1e-10]), [0.0, 1.0, -0.2, -0.1])
    assert out.dtype == F64
    assert out.item() == pytest.approx(89.99999917, rel=1e-6)


def test_mixed_tensor_dtypes_promote_instead_of_rounding(device):
    g = gpu.GLD("FMKL", device=device)
    y = T([1 - 1e-10], F64, device)
    p = T([0.0, 1.0, -0.25, -0.125], F32, device)  # exactly representable
    out = g.Q(y, p)
    assert out.dtype == F64 and torch.isfinite(out).all()


def test_list_inputs_default_to_float32(device):
    assert gpu.GLD("FMKL", device=device).Q([0.5], [0.0, 1.0, 0.2, 0.3]).dtype == F32


@pytest.mark.parametrize("dt", [F32, F64])
def test_constructor_dtype_is_applied(dt, device):
    g = gpu.GLD("FMKL", device=device, dtype=dt)
    assert g.dtype == dt
    assert g.Q(np.array([0.5]), np.array([0.0, 1.0, 0.2, 0.3])).dtype == dt
    assert g.rand([0.0, 1.0, 0.2, 0.3], size=4).dtype == dt


def test_supp_with_batched_params(device):
    g = gpu.GLD("FMKL", device=device)
    p = T([[0.0, 1.0, 0.2, 0.3], [1.0, 2.0, 0.1, 0.1]], F64, device)
    lo, hi = g.supp(p)
    assert lo.shape == hi.shape == (2,)
    torch.testing.assert_close(lo, T([-5.0, -4.0], F64, device))
    torch.testing.assert_close(hi, T([1.0 / 0.3, 6.0], F64, device))


def test_rs_validity_in_float32_with_large_lambda(device):
    """(l4-1)**(l4-1) overflows float32 at l4 ~ 30; validity must not turn False because of that."""
    from gldpy_gpu import GLD

    p = [0.0, -1.0, -0.9, 60.0]
    g = gpu.GLD("RS", device=device)
    assert g.check_param(T(p, F32, device)) == bool(GLD("RS").check_param(p))
    assert g.check_param(T(p, F32, device)) is True


def test_check_param_float32_matches_cpu(ptype, cpu_gld, gpu_gld, device):
    rng = np.random.default_rng(1)
    for p in rng.uniform(-3, 3, size=(300, 4)):
        assert gpu_gld.check_param(T(p, F32, device)) == bool(cpu_gld.check_param(p)), p


# ---- rand


def test_rand_float32_stays_finite_and_reaches_past_the_float32_grid(device):
    """float32 uniforms have step 2**-24: hitting 0 gave -inf, and the upper tail stopped at Q(1 - 2**-24)."""
    g = gpu.GLD("FMKL", device=device)
    p = T([0.0, 1.0, -0.2, -0.1], F32, device)
    cap = g.ISF(T([2.0**-24], F64, device), p).item()
    gen = torch.Generator(device=device).manual_seed(3)
    top, n_bad = -math.inf, 0
    for _ in range(5):
        x = g.rand(p, size=8_000_000, generator=gen)
        assert x.dtype == F32
        n_bad += (~torch.isfinite(x)).sum().item()
        top = max(top, x.max().item())
    assert n_bad == 0
    assert top > cap


def test_rand_upper_tail_frequency_is_right(dtype, device):
    g = gpu.GLD("FMKL", device=device)
    p = T([0.0, 1.0, -0.2, -0.1], dtype, device)
    gen = torch.Generator(device=device).manual_seed(4)
    n, s = 5_000_000, 1e-4
    x = g.rand(p, size=n, generator=gen)
    hits = (x > g.ISF(T([s], F64, device), p).to(dtype)).sum().item()
    assert abs(hits - n * s) < 5 * math.sqrt(n * s)


def test_GoF_large_sample_float32_grid_is_exact(device):
    """float32 cannot hold (k + 0.5) / n for n > 2**24; the plotting positions must stay exact."""
    g = gpu.GLD("FMKL", device=device)
    p = T([0.0, 1.0, -0.2, -0.1], F32, device)
    n = 200_000
    y64 = (torch.arange(n, dtype=F64, device=device) + 0.5) / n
    exact = g.Q(y64, p.double()).to(F32)
    assert g.GoF_Q_metric(exact, p).item() < 1e-7


@pytest.mark.parametrize("lam", [1e-2, 1e-3, 1e-5, 1e-7])
def test_rs_near_zero_lambda_has_no_cancellation(lam, dtype, device):
    """y**a - s**b is a difference of two numbers close to 1; written with expm1 it keeps its digits."""
    y = np.array([0.01, 0.1, 0.9, 0.99])
    truth = (np.expm1(lam * np.log(y)) - np.expm1(lam * np.log1p(-y))) / lam
    g = gpu.GLD("RS", device=device)
    out = g.Q(T(y, dtype, device), T([0.0, lam, lam, lam], dtype, device)).cpu().double().numpy()
    np.testing.assert_allclose(out, truth, rtol=1e-5 if dtype == F32 else 1e-13)
