"""Every bug documented in docs/upstream-bugs.md, as a pair of tests.

`test_Unn_original_*` pins the misbehavior in the frozen PyPI package gldpy 0.2, and
`test_Unn_fixed_*` shows `gldpy_gpu.GLD` does not have it. The last section checks that the fixes
changed nothing else.
"""
import math
import warnings

import numpy as np
import pytest

from gldpy_gpu import GLD

Original = pytest.importorskip("gldpy").GLD

FIT_KW = dict(disp_optimizer=False, disp_fit=False, test_gof=False)
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def percentile_stats(q):
    a, b, c, d, e = q
    return np.array([c, e - a, (c - a) / (e - c), (d - b) / (e - a)])


PROBS = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
RS_DATA = Original("RS").rand(np.array([0.0, -0.2, -0.1, -0.15]), size=5000, random_state=11)


def rs_pm_residual(cls):
    fit = cls("RS").fit_PM(RS_DATA, [-0.1, -0.15], xtol=1e-7, **FIT_KW)
    return np.abs(percentile_stats(Original("RS").Q(PROBS, fit)) - percentile_stats(np.quantile(RS_DATA, PROBS))).max()


def test_U01_original_rs_fit_pm_leaves_a_residual():
    assert rs_pm_residual(Original) > 1e-3


def test_U01_fixed_rs_fit_pm_solves_its_equations():
    assert rs_pm_residual(GLD) < 1e-6


# U02: logistic, F(x) = 1 / (1 + exp(-x))
TAILS = np.array([1e-6, 1e-8, 1e-12])


def test_U02_original_cdf_is_wrong_in_the_tails():
    g = Original("FMKL")
    p = [0.0, 1.0, 0.0, 0.0]
    x = np.log(TAILS)  # x = logit(s) ~ log(s)
    truth = 1.0 / (1.0 + np.exp(-x))
    assert np.abs(g.CDF_num(x, p) / truth - 1.0).min() > 1.0


def test_U02_fixed_cdf_and_sf_keep_relative_accuracy():
    g = GLD("FMKL")
    p = [0.0, 1.0, 0.0, 0.0]
    x = np.log(TAILS)
    np.testing.assert_allclose(g.CDF_num(x, p), 1.0 / (1.0 + np.exp(-x)), rtol=1e-7)
    np.testing.assert_allclose(g.SF_num(-x, p), 1.0 / (1.0 + np.exp(-x)), rtol=1e-7)


# U03: mean, variance, skewness, kurtosis with closed forms
MOMENTS = [
    ("FMKL", [0.0, 1.0, 0.0, 0.0], (0.0, math.pi**2 / 3, 0.0, 4.2)),  # logistic
    ("VSL", [0.0, 2.0, 0.5, 0.0], (0.0, math.pi**2 / 3, 0.0, 4.2)),  # logistic
    ("VSL", [0.0, 1.0, 1.0, 0.0], (1.0, 1.0, 2.0, 9.0)),  # exponential
]


def moments(g, p):
    return np.array([g.mean(p), g.var(p), g.skewness(p), g.kurtosis(p)])


@pytest.mark.parametrize("ptype, p, truth", MOMENTS, ids=["logistic_FMKL", "logistic_VSL", "exponential"])
def test_U03_original_moments_are_nan_at_lambda_zero(ptype, p, truth):
    assert np.isnan(moments(Original(ptype), p)).all()


@pytest.mark.parametrize("ptype, p, truth", MOMENTS, ids=["logistic_FMKL", "logistic_VSL", "exponential"])
def test_U03_fixed_moments_match_closed_forms(ptype, p, truth):
    np.testing.assert_allclose(moments(GLD(ptype), p), truth, atol=1e-7)


def test_U03_original_kurtosis_explodes_near_lambda_zero():
    assert Original("FMKL").kurtosis([0.0, 1.0, 1e-5, 1e-5]) > 1000.0


@pytest.mark.parametrize("lam", [1e-9, 1e-5, 1e-3, 0.019, 0.021, 0.05])
def test_U03_fixed_moments_are_smooth_through_the_small_lambda_switch(lam):
    """FMKL(lam, lam) moves continuously away from the logistic's kurtosis 4.2."""
    g = GLD("FMKL")
    p = [0.0, 1.0, lam, lam]
    y = (np.arange(400_000) + 0.5) / 400_000
    q = g.Q(y, p)
    assert g.var(p) == pytest.approx(q.var(), rel=1e-3)
    assert g.kurtosis(p) == pytest.approx(((q - q.mean()) ** 4).mean() / q.var() ** 2, rel=2e-3)


# U04: fits started at lambda = 0
LOGISTIC = Original("FMKL").rand([0.0, 1.0, 0.0, 0.0], size=3000, random_state=2)


@pytest.mark.parametrize("method", ["fit_LMM", "fit_PM", "fit_MM"])
def test_U04_original_fit_from_lambda_zero_returns_nan(method):
    assert np.isnan(getattr(Original("FMKL"), method)(LOGISTIC, [0.0, 0.0], **FIT_KW)).any()


@pytest.mark.parametrize("method", ["fit_LMM", "fit_PM", "fit_MM"])
def test_U04_fixed_fit_from_lambda_zero_finds_the_logistic(method):
    fit = getattr(GLD("FMKL"), method)(LOGISTIC, [0.0, 0.0], **FIT_KW)
    assert np.abs(fit[2:]).max() < 0.1 and abs(fit[1] - 1.0) < 0.1


# U05: random_state
def test_U05_original_ignores_seed_zero():
    a = Original("FMKL").rand([0.0, 1.0, 0.2, 0.3], size=3, random_state=0)
    b = Original("FMKL").rand([0.0, 1.0, 0.2, 0.3], size=3, random_state=0)
    assert not np.array_equal(a, b)


def test_U05_original_reseeds_the_global_generator():
    np.random.seed(5)
    expected = np.random.random()
    np.random.seed(5)
    Original("FMKL").rand([0.0, 1.0, 0.2, 0.3], size=3, random_state=9)
    assert np.random.random() != expected


def test_U05_fixed_seed_zero_works_and_global_generator_is_untouched():
    a = GLD("FMKL").rand([0.0, 1.0, 0.2, 0.3], size=3, random_state=0)
    b = GLD("FMKL").rand([0.0, 1.0, 0.2, 0.3], size=3, random_state=0)
    assert np.array_equal(a, b)
    np.random.seed(5)
    expected = np.random.random()
    np.random.seed(5)
    GLD("FMKL").rand([0.0, 1.0, 0.2, 0.3], size=3, random_state=9)
    assert np.random.random() == expected


# U06: correct_supp
def test_U06_original_correct_supp_mutates_and_truncates():
    p = np.array([0.0, 1.0, 0.5, 0.5])
    Original("FMKL").correct_supp(np.array([-9.0, 9.0]), p)
    assert not np.array_equal(p, [0.0, 1.0, 0.5, 0.5])
    out = Original("RS").correct_supp(np.array([-9.0, 9.0]), np.array([0, 1, 1, 2]))
    assert out.dtype.kind == "i"


def test_U06_fixed_correct_supp_returns_new_float_parameters():
    p = np.array([0.0, 1.0, 0.5, 0.5])
    out = GLD("FMKL").correct_supp(np.array([-9.0, 9.0]), p)
    assert np.array_equal(p, [0.0, 1.0, 0.5, 0.5])
    assert GLD("FMKL").Q(1e-4, out) == pytest.approx(-9.0)
    assert GLD("FMKL").Q(1 - 1e-4, out) == pytest.approx(9.0)
    from_int = GLD("RS").correct_supp(np.array([-9.0, 9.0]), np.array([0, 1, 1, 2]))
    assert from_int.dtype.kind == "f" and GLD("RS").Q(1 - 1e-4, from_int) == pytest.approx(9.0)


# U07: GoF_tests
BOUNDED = [0.0, 1.0, 0.5, 0.5]
WITH_OUTLIER = np.r_[Original("FMKL").rand(BOUNDED, size=200, random_state=1), 100.0]


def test_U07_original_gof_crashes_on_data_outside_the_support():
    with pytest.raises(ValueError):
        Original("FMKL").GoF_tests(BOUNDED, WITH_OUTLIER)


def test_U07_fixed_gof_counts_outside_points_in_the_end_bins():
    ks, chi2 = GLD("FMKL").GoF_tests(BOUNDED, WITH_OUTLIER)
    assert np.isfinite(ks.pvalue) and np.isfinite(chi2.pvalue)


# U08: array shapes
def test_U08_original_pdf_num_crashes_on_2d_input():
    with pytest.raises(IndexError):
        Original("FMKL").PDF_num(np.array([[0.1, 0.2], [0.3, 0.4]]), [0.0, 1.0, 0.2, 0.3])


def test_U08_fixed_pdf_and_cdf_keep_the_input_shape():
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    assert GLD("FMKL").PDF_num(x, [0.0, 1.0, 0.2, 0.3]).shape == (2, 2)
    assert GLD("FMKL").CDF_num(x, [0.0, 1.0, 0.2, 0.3]).shape == (2, 2)
    assert GLD("FMKL").CDF_num(0.1, [0.0, 1.0, 0.2, 0.3]).shape == (1,)


# U09: exact endpoints
def test_U09_original_endpoints_give_nan():
    ends = np.array([0.0, 1.0])
    assert np.isnan(Original("VSL").Q(ends, [0.0, 1.0, 1.0, -1.0])[0])
    assert np.isnan(Original("RS").PDF_Q(ends, [0.0, 1.0, 0.0, 1.0])[0])


def test_U09_fixed_endpoints():
    ends = np.array([0.0, 1.0])
    # VSL(0, 1, 1, -1) is Pareto on [0, inf); RS(0, 1, 0, 1) is Q = y, uniform on [0, 1], density 1.
    np.testing.assert_array_equal(GLD("VSL").Q(ends, [0.0, 1.0, 1.0, -1.0]), [0.0, np.inf])
    np.testing.assert_allclose(GLD("RS").PDF_Q(ends, [0.0, 1.0, 0.0, 1.0]), [1.0, 1.0])


# U10: grid search, RS with negative lambda2
RS_HEAVY = Original("RS").rand(np.array([0.0, -0.2, -0.1, -0.15]), size=300, random_state=3)


def grid_result(cls):
    g = cls("RS")

    def objective(q):
        return np.inf if not g.check_param(q) else g.GoF_Q_metric(RS_HEAVY, q)

    return g, g.grid_search(RS_HEAVY, fun_min=objective, n_grid=5, grid_min=-0.5, grid_max=0.5)


def test_U10_original_rs_grid_search_returns_invalid_parameters():
    g, out = grid_result(Original)
    assert not g.check_param(out)


def test_U10_fixed_rs_grid_search_returns_valid_parameters():
    g, out = grid_result(GLD)
    assert g.check_param(out)
    assert g.GoF_Q_metric(RS_HEAVY, out) < 1.0


# U11: maximum product of spacings
def mps_excluded_points(cls, seed):
    g = cls("FMKL")
    d = Original("FMKL").rand(BOUNDED, size=40, random_state=seed)
    lo, hi = g.supp(g.fit_MPS(d, [0.5, 0.5], method="LMM", maxfun=400, **FIT_KW))
    return int((d < lo).sum() + (d > hi).sum())


def test_U11_original_mps_excludes_observed_points_from_the_support():
    assert mps_excluded_points(Original, 1) >= 1


@pytest.mark.parametrize("seed", [1, 2, 5])
def test_U11_fixed_mps_keeps_every_point_in_the_support(seed):
    assert mps_excluded_points(GLD, seed) == 0


# U12: tie jitter
TIED = np.round(Original("FMKL").rand(BOUNDED, size=30, random_state=4), 1)


def test_U12_original_mps_with_ties_is_not_reproducible():
    a = Original("FMKL").fit_MPS(TIED, [0.5, 0.5], method="LMM", maxfun=30, **FIT_KW)
    b = Original("FMKL").fit_MPS(TIED, [0.5, 0.5], method="LMM", maxfun=30, **FIT_KW)
    assert not np.array_equal(a, b)


def test_U12_fixed_mps_with_ties_is_reproducible():
    a = GLD("FMKL").fit_MPS(TIED, [0.5, 0.5], method="LMM", maxfun=30, **FIT_KW)
    b = GLD("FMKL").fit_MPS(TIED, [0.5, 0.5], method="LMM", maxfun=30, **FIT_KW)
    assert np.array_equal(a, b)


# U13: documentation
@pytest.mark.parametrize("sign, cls", [("-", Original), ("+", GLD)], ids=["original_says_minus", "fixed_says_plus"])
def test_U13_class_docstring_density_sign(sign, cls):
    doc = cls.__doc__
    assert f"\\lambda_3 y^{{\\lambda_3-1}} {sign} \\lambda_4" in doc
    assert f"y^{{\\lambda_3-1}} {sign} (1-y)^{{\\lambda_4-1}}" in doc


# ---- The fixes changed nothing else


CASES = [
    ("FMKL", [0.0, 1.0, 0.2, 0.3]),
    ("FMKL", [1.0, 2.0, -0.15, 0.1]),
    ("RS", [0.0, 0.2, 0.1, 0.15]),
    ("RS", [0.0, -0.2, -0.1, -0.15]),
    ("VSL", [0.0, 1.0, 0.4, 0.3]),
    ("VSL", [1.0, 2.0, 0.6, -0.1]),
]
IDS = [f"{t}{p[1:]}" for t, p in CASES]
Y = np.linspace(0.001, 0.999, 201)


@pytest.mark.parametrize("ptype, p", CASES, ids=IDS)
def test_parity_core(ptype, p):
    a, b = GLD(ptype), Original(ptype)
    np.testing.assert_allclose(a.Q(Y, p), b.Q(Y, p), rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(a.PDF_Q(Y, p), b.PDF_Q(Y, p), rtol=1e-12)
    assert np.all(a.supp(p) == np.array(b.supp(p)))
    x = b.Q(np.linspace(0.05, 0.95, 19), p)
    np.testing.assert_allclose(a.CDF_num(x, p), b.CDF_num(x, p), atol=1e-4)
    np.testing.assert_allclose(a.PDF_num(x, p), b.PDF_num(x, p), rtol=1e-3)


@pytest.mark.parametrize("ptype, p", CASES, ids=IDS)
def test_parity_moments(ptype, p):
    np.testing.assert_allclose(moments(GLD(ptype), p), moments(Original(ptype), p), rtol=1e-10, atol=1e-10)
    assert GLD(ptype).median(p) == pytest.approx(Original(ptype).median(p))


@pytest.mark.parametrize("ptype, p", CASES, ids=IDS)
def test_parity_rand_stream_with_a_seed(ptype, p):
    np.testing.assert_allclose(
        GLD(ptype).rand(p, size=50, random_state=7), Original(ptype).rand(p, size=50, random_state=7), rtol=1e-10, atol=1e-10
    )


@pytest.mark.parametrize("ptype, p", CASES, ids=IDS)
@pytest.mark.parametrize("method", ["fit_LMM", "fit_PM"])
def test_parity_fits_same_start(method, ptype, p):
    if method == "fit_PM" and ptype == "RS":
        pytest.skip("U01: the original's result is the bug")
    data = Original(ptype).rand(p, size=3000, random_state=11)
    if ptype == "VSL":
        start = None if method == "fit_LMM" else [p[3]]
    else:
        start = p[2:]
    a = getattr(GLD(ptype), method)(data, start, xtol=1e-7, **FIT_KW)
    b = getattr(Original(ptype), method)(data, start, xtol=1e-7, **FIT_KW)
    np.testing.assert_allclose(a, b, atol=2e-5)


def test_original_and_oracle_agree_on_fit_MM():
    data = Original("FMKL").rand([0.0, 1.0, 0.2, 0.3], size=3000, random_state=11)
    a = GLD("FMKL").fit_MM(data, [0.2, 0.3], xtol=1e-6, **FIT_KW)
    b = Original("FMKL").fit_MM(data, [0.2, 0.3], xtol=1e-6, **FIT_KW)
    np.testing.assert_allclose(a, b, atol=2e-4)


@pytest.mark.slow
@pytest.mark.parametrize("ptype, p", [c for c in CASES if c[1][3] > 0.0 and c[1][2] > 0.0], ids=lambda v: str(v))
def test_parity_fit_ML_likelihood(ptype, p):
    data = Original(ptype).rand(p, size=120, random_state=5)
    start = None if ptype == "VSL" else p[2:]
    kw = dict(method="LMM", maxfun=150, **FIT_KW)
    a = GLD(ptype).fit_ML(data, initial_guess=start, **kw)
    b = Original(ptype).fit_ML(data, initial_guess=start, **kw)
    nll = lambda q: -np.log(GLD(ptype).PDF_num(data, q)).mean()  # noqa: E731
    assert nll(a) <= nll(b) + 2e-3


# U14: cancellation in Q as lambda -> 0
SMALL = 1e-12
Y_SMALL = np.array([0.01, 0.1, 0.9, 0.99])
TRUTH_SMALL = (np.expm1(SMALL * np.log(Y_SMALL)) - np.expm1(SMALL * np.log1p(-Y_SMALL))) / SMALL


def test_U14_original_q_loses_digits_near_lambda_zero():
    errors = np.abs(Original("FMKL").Q(Y_SMALL, [0.0, 1.0, SMALL, SMALL]) / TRUTH_SMALL - 1.0)
    assert errors.max() > 1e-7


@pytest.mark.parametrize("ptype, p", [("FMKL", [0.0, 1.0, SMALL, SMALL]), ("RS", [0.0, SMALL, SMALL, SMALL])], ids=["FMKL", "RS"])
def test_U14_fixed_q_keeps_its_digits(ptype, p):
    np.testing.assert_allclose(GLD(ptype).Q(Y_SMALL, p), TRUTH_SMALL, rtol=1e-13)
