"""The fits must solve the same equations as the CPU oracle, at either precision."""
import numpy as np
import pytest
import torch

from gldpy_gpu import GLD, gpu

F64, F32 = torch.float64, torch.float32
FIT_KW = dict(disp_optimizer=False, disp_fit=False, test_gof=False)

# Includes the cases where the old grid + Adam refine stopped ~1e-2 short, or picked a far root.
CASES = [
    ("FMKL", [0.0, 1.0, 0.2, 0.3]),
    ("FMKL", [0.0, 1.0, -0.15, -0.1]),
    ("FMKL", [1.0, 2.0, 0.5, 0.05]),
    ("RS", [0.0, 0.2, 0.1, 0.15]),
    ("RS", [0.0, -0.2, -0.1, -0.15]),
    ("RS", [0.0, 1.0, 1.0, 2.0]),
    ("VSL", [0.0, 1.0, 0.4, 0.3]),
    ("VSL", [0.0, 1.0, 0.4, -0.15]),
]
IDS = [f"{t}{p[2:]}" for t, p in CASES]


@pytest.fixture(params=[F64, F32], ids=["f64", "f32"])
def dtype(request):
    return request.param


def draw(ptype, param, n=5000, seed=11):
    return GLD(ptype).rand(np.array(param), size=n, random_state=seed)


def tensor(x, dtype, device):
    return torch.as_tensor(x, dtype=dtype, device=device)


def guess(ptype, param):
    return param[3] if ptype == "VSL" else param[2:]


@pytest.mark.parametrize("ptype, param", CASES, ids=IDS)
def test_fit_LMM_from_same_start_matches_cpu_solution(ptype, param, dtype, device):
    data = draw(ptype, param)
    start = None if ptype == "VSL" else param[2:]
    oracle = GLD(ptype).fit_LMM(data, start, xtol=1e-7, **FIT_KW)
    fit = gpu.GLD(ptype, device=device).fit_LMM(tensor(data, dtype, device), initial_guess=start)
    assert fit.dtype == dtype
    np.testing.assert_allclose(fit.cpu().double().numpy(), oracle, atol=2e-3 if dtype == F32 else 1e-4, rtol=0.0)


@pytest.mark.parametrize("ptype, param", CASES, ids=IDS)
def test_fit_LMM_without_guess_is_as_close_to_the_data_as_cpu(ptype, param, dtype, device):
    """RS(1, 2) has several exact roots; the no-guess rule must find one at least as good."""
    data = draw(ptype, param)
    cpu_gld = GLD(ptype)
    oracle = cpu_gld.fit_LMM(data, None if ptype == "VSL" else param[2:], xtol=1e-7, **FIT_KW)
    fit = gpu.GLD(ptype, device=device).fit_LMM(tensor(data, dtype, device)).cpu().double().numpy()
    assert cpu_gld.GoF_Q_metric(data, fit) <= cpu_gld.GoF_Q_metric(data, oracle) * 1.02


PM_PARITY = [c for c in CASES if c[0] != "RS"]


@pytest.mark.parametrize("ptype, param", PM_PARITY, ids=[f"{t}{p[2:]}" for t, p in PM_PARITY])
def test_fit_PM_matches_cpu_solution(ptype, param, dtype, device):
    """No guess is given: of several exact roots, the fit must still pick the data's own."""
    data = draw(ptype, param)
    oracle = GLD(ptype).fit_PM(data, [param[3]] if ptype == "VSL" else param[2:], xtol=1e-7, **FIT_KW)
    fit = gpu.GLD(ptype, device=device).fit_PM(tensor(data, dtype, device))
    np.testing.assert_allclose(fit.cpu().double().numpy(), oracle, atol=5e-3 if dtype == F32 else 1e-3, rtol=0.0)


def percentile_stats(q):
    """(median, spread, t3-like, t4-like) from the values at [u, .25, .5, .75, 1 - u]."""
    a, b, c, d, e = q
    return np.array([c, e - a, (c - a) / (e - c), (d - b) / (e - a)])


@pytest.mark.parametrize("param", [p for t, p in CASES if t == "RS"], ids=lambda p: str(p[2:]))
def test_fit_PM_rs_solves_the_percentile_equations(param, dtype, device):
    """Not CPU parity: upstream's RS objective lacks an abs() on its a4 term, so it leaves a residual
    (see ROADMAP). Oracle: the fitted quantile function must reproduce the sample percentiles."""
    data = draw("RS", param)
    probs = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    fit = gpu.GLD("RS", device=device).fit_PM(tensor(data, dtype, device)).cpu().double().numpy()
    np.testing.assert_allclose(
        percentile_stats(GLD("RS").Q(probs, fit)), percentile_stats(np.quantile(data, probs)),
        atol=1e-3 if dtype == F32 else 1e-8,
    )
    assert np.all(np.abs(fit[2:] - np.array(param[2:])) < 0.5)  # the data's root, not a far one


def test_initial_guess_selects_the_root_like_the_cpu(device):
    """FMKL(1, 1) and FMKL(2, 2) are both uniform: the guess decides which one comes back."""
    data = draw("FMKL", [0.0, 1.0, 1.0, 1.0])
    d = tensor(data, F64, device)
    g = gpu.GLD("FMKL", device=device)
    for start in ([1.2, 1.2], [2.2, 2.2]):
        fit = g.fit_LMM(d, initial_guess=start).cpu().numpy()
        oracle = GLD("FMKL").fit_LMM(data, start, xtol=1e-7, **FIT_KW)
        np.testing.assert_allclose(fit, oracle, atol=1e-3)
        pm = g.fit_PM(d, initial_guess=start).cpu().numpy()
        np.testing.assert_allclose(pm, GLD("FMKL").fit_PM(data, start, xtol=1e-7, **FIT_KW), atol=1e-3)


def test_fit_PM_vsl_guess_selects_root(device):
    data = draw("VSL", [0.0, 1.0, 0.5, 1.0])
    d = tensor(data, F64, device)
    fit = gpu.GLD("VSL", device=device).fit_PM(d, initial_guess=2.0).cpu().numpy()
    assert fit[3] == pytest.approx(GLD("VSL").fit_PM(data, [2.0], xtol=1e-7, **FIT_KW)[3], abs=1e-3)


@pytest.mark.parametrize("method", ["fit_LMM", "fit_PM"])
def test_fits_are_location_scale_equivariant(ptype, method, dtype, device):
    from conftest import PARAMS

    data = draw(ptype, PARAMS[ptype])
    g = gpu.GLD(ptype, device=device)
    base = getattr(g, method)(tensor(data, F64, device)).cpu().numpy()
    k = 1e4
    scaled = getattr(g, method)(tensor(data * k + 3 * k, F64, device)).cpu().numpy()
    np.testing.assert_allclose(scaled[2:], base[2:], atol=1e-6)


def test_fit_PM_above_torch_quantile_size_limit():
    """torch.quantile raises above 2**24 elements."""
    g = gpu.GLD("FMKL", device="cpu")
    n = 2**24 + 10
    x = g.rand([0.0, 1.0, 0.2, 0.3], size=n, generator=torch.Generator().manual_seed(1)).to(F32)
    stats = gpu.compute_sample_percentiles(x)
    assert all(torch.isfinite(v) for v in stats)


@pytest.mark.parametrize("method", ["fit_LMM", "fit_PM", "fit_ML"])
@pytest.mark.parametrize("bad", ["nan", "inf", "constant"])
def test_bad_data_is_rejected(method, bad, device):
    d = torch.linspace(0, 1, 100, dtype=F64, device=device)
    if bad == "constant":
        d = torch.ones_like(d)
    else:
        d[3] = float(bad)
    with pytest.raises(ValueError):
        getattr(gpu.GLD("VSL", device=device), method)(d)


def test_correct_supp_matches_cpu(ptype, device):
    from conftest import PARAMS

    param = np.array(PARAMS[ptype], float)
    data = GLD(ptype).rand(param * [1, 1, 1, 1], size=500, random_state=2) * 1.3
    if not np.isfinite(GLD(ptype).supp(param)).all():
        pytest.skip("infinite support")
    g = gpu.GLD(ptype, device=device)
    fit = g.correct_supp(tensor(data, F64, device), tensor(param, F64, device)).cpu().numpy()
    oracle = GLD(ptype).correct_supp(data, param.copy())
    # The oracle stops at fmin's xtol, so compare what both are trying to achieve.
    for p in (fit, oracle):
        assert GLD(ptype).Q(1e-4, p) == pytest.approx(min(data.min(), GLD(ptype).Q(1e-4, param)), abs=1e-2)
        assert GLD(ptype).Q(1 - 1e-4, p) == pytest.approx(max(data.max(), GLD(ptype).Q(1 - 1e-4, param)), abs=1e-2)
    # Closed form: exact at both ends.
    assert GLD(ptype).Q(1 - 1e-4, fit) == pytest.approx(max(data.max(), GLD(ptype).Q(1 - 1e-4, param)), rel=1e-9)


def true_nll(ptype, data, param):
    """Oracle likelihood at high precision via the logit solver in float64, on the CPU."""
    g = gpu.GLD(ptype, device="cpu")
    lp = g.log_PDF_num(torch.as_tensor(data, dtype=F64), torch.as_tensor(param, dtype=F64), xtol=1e-11, max_iter=200)
    return -lp.mean().item()


# Mean NLL reached by the CPU `fit_ML(method="LMM", maxfun=300)` on draw(ptype, param, n=300, seed=5).
# Literals, because that oracle takes ~20 s per case.
ML_ORACLE = [
    ("FMKL", [0.0, 1.0, 0.2, 0.3], 1.64965),
    ("FMKL", [0.0, 1.0, -0.15, -0.1], 2.21525),
    ("VSL", [0.0, 1.0, 0.4, 0.3], 0.87871),
    ("RS", [0.0, 0.2, 0.1, 0.15], 1.35504),
]


@pytest.mark.parametrize("ptype, param, oracle_nll", ML_ORACLE, ids=["FMKL_bounded", "FMKL_heavy", "VSL", "RS"])
def test_fit_ML_reaches_cpu_likelihood(ptype, param, oracle_nll, dtype, device):
    data = draw(ptype, param, n=300, seed=5)
    fit = gpu.GLD(ptype, device=device).fit_ML(tensor(data, dtype, device)).cpu().double().numpy()
    assert true_nll(ptype, data, fit) <= oracle_nll + 5e-4


@pytest.mark.parametrize("k", [1e-5, 1.0, 1e5])
def test_fit_ML_is_scale_invariant(k, device):
    """A fixed Adam step in raw parameter units stalls or explodes depending on the data's unit."""
    param = [0.0, 1.0, 0.2, 0.3]
    data = draw("FMKL", param, n=500, seed=6)
    g = gpu.GLD("FMKL", device=device)
    ref = g.fit_ML(tensor(data, F64, device), steps=30).cpu().numpy()
    out = g.fit_ML(tensor(data * k, F64, device), steps=30).cpu().numpy()
    np.testing.assert_allclose(out[2:], ref[2:], atol=1e-5)
    assert out[0] == pytest.approx(ref[0] * k, abs=1e-5 * k)
    assert out[1] == pytest.approx(ref[1] / k, rel=1e-4)


def test_fit_ML_support_contains_the_data(device):
    """Likelihood is 0 for a point outside the support; the old clamp gave it a finite density."""
    param = [0.0, 1.0, 1.5, 1.2]
    data = draw("FMKL", param, n=800, seed=7)
    g = gpu.GLD("FMKL", device=device)
    fit = g.fit_ML(tensor(data, F64, device))
    lo, hi = g.supp(fit)
    assert lo.item() <= data.min() and hi.item() >= data.max()


def test_fit_ML_survives_float32_with_large_offset(device):
    data = draw("FMKL", [0.0, 1.0, 0.2, 0.3], n=2000, seed=8) + 5000.0
    g = gpu.GLD("FMKL", device=device)
    fit = g.fit_ML(tensor(data, F32, device))
    assert g.check_param(fit)
    assert fit[0].item() == pytest.approx(5000.0, abs=0.5)


def test_fit_ML_keeps_moving_when_the_support_touches_the_data(device):
    """RS approaches the exponential through bounded shapes, with the largest point at the support's
    end. Rejecting every step that crosses it froze the shape parameters at NLL 1.19 (truth: 1.0)."""
    from scipy import stats

    n = 5000
    data = stats.expon().ppf((np.arange(n) + 0.5) / n)
    g = gpu.GLD("RS", device=device)
    d = tensor(data, F64, device)
    fit = g.fit_ML(d, steps=200)
    assert g.check_param(fit)
    assert -g.log_PDF_num(d, fit, xtol=1e-10).mean().item() < 1.03


@pytest.mark.parametrize("method, tol", [("fit_LMM", 1e-4), ("fit_PM", 1e-3)])
def test_rs_logistic_root_near_zero_found_in_float32(method, tol, device):
    """The logistic is RS's lambda -> 0 limit, so the fit must approach it in float32 as well.
    A finite-difference Jacobian with h ~ eps**(1/3) ~ 5e-3 cannot resolve a root at lambda ~ 1e-4."""
    from scipy import stats

    n = 20000
    data = stats.logistic().ppf((np.arange(n) + 0.5) / n)
    fit = getattr(gpu.GLD("RS", device=device), method)(tensor(data, F32, device)).cpu().double().numpy()
    assert np.abs(fit[2:]).max() < 0.05
    probs = np.linspace(0.01, 0.99, 99)
    np.testing.assert_allclose(GLD("RS").Q(probs, fit), stats.logistic().ppf(probs), atol=tol * 100)


def test_fit_ML_never_ends_worse_than_its_start_in_float32(device):
    from scipy import stats

    n = 5000
    data = stats.logistic().ppf((np.arange(n) + 0.5) / n)
    g = gpu.GLD("RS", device=device)
    fit = g.fit_ML(tensor(data, F32, device), steps=200).cpu().double().numpy()
    d64 = tensor(data, F64, device)
    nll = -g.log_PDF_num(d64, tensor(fit, F64, device), xtol=1e-10).mean().item()
    assert nll < 2.02  # the logistic's entropy is 2


# Mean NLL reached by the original gldpy's fit_ML (n=800, maxfun=400), evaluated on the full grid.
# Literals, because the oracle takes 50-130 s per case.
@pytest.mark.parametrize(
    "dist, entropy_gap",
    [("lognorm_0.25", 0.0053), ("beta_2_5", 0.0010), ("gamma_2", 0.0022)],
)
def test_fit_ML_on_non_members_is_as_good_as_the_original(dist, entropy_gap, device):
    from scipy import stats

    d = {"lognorm_0.25": stats.lognorm(0.25), "beta_2_5": stats.beta(2, 5), "gamma_2": stats.gamma(2)}[dist]
    n = 5000
    data = d.ppf((np.arange(n) + 0.5) / n)
    g = gpu.GLD("FMKL", device=device)
    t = tensor(data, F64, device)
    fit = g.fit_ML(t, steps=300)
    nll = -g.log_PDF_num(t, fit, xtol=1e-10).mean().item()
    assert nll - float(d.entropy()) <= entropy_gap * 1.5
