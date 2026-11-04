"""Optimizations must not change results: every fast path against the general path it replaces."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from gldpy_gpu import _torch_backend as B

F32, F64 = torch.float32, torch.float64


@pytest.fixture(params=[F64, F32], ids=["f64", "f32"])
def dtype(request):
    return request.param


CASES = [
    ("FMKL", [0.0, 1.0, 0.2, 0.3]),
    ("FMKL", [1.0, 2.0, -0.2, -0.1]),
    ("RS", [0.0, 0.2, 0.1, 0.15]),
    ("RS", [0.0, -0.2, -0.1, -0.15]),
    ("RS", [0.0, -1.0, -0.5, 2.0]),
    ("VSL", [0.0, 1.0, 0.4, 0.3]),
    ("VSL", [0.0, 1.0, 0.4, -0.2]),
]


def logs(dtype, device, n=2001):
    t = torch.linspace(-25.0, 25.0, n, dtype=dtype, device=device)
    return t, F.logsigmoid(t), F.logsigmoid(-t)


@pytest.mark.parametrize("ptype, param", CASES, ids=[f"{t}{p[1:]}" for t, p in CASES])
def test_interior_log_density_equals_the_general_form(ptype, param, dtype, device):
    p = torch.as_tensor(param, dtype=dtype, device=device)
    if not B.validate_gld_params(p, ptype):
        pytest.skip("not a valid parameter set")
    _, log_y, log_s = logs(dtype, device)
    fast = B._log_density_from_logs(log_y, log_s, p, ptype, interior=True)
    slow = B._log_density_from_logs(log_y, log_s, p, ptype)
    torch.testing.assert_close(fast, slow, rtol=1e-4 if dtype == F32 else 1e-12, atol=1e-5 if dtype == F32 else 1e-12)


@pytest.mark.parametrize("lam", [0.3, -0.2, 1e-6, -1e-8])
def test_box_cox_value_does_not_depend_on_requires_grad(lam, dtype, device):
    _, log_y, _ = logs(dtype, device)
    plain = B._box_cox(log_y, torch.tensor(lam, dtype=dtype, device=device))
    tracked = B._box_cox(log_y, torch.tensor(lam, dtype=dtype, device=device, requires_grad=True))
    torch.testing.assert_close(plain, tracked.detach(), rtol=1e-5 if dtype == F32 else 1e-12, atol=1e-5 if dtype == F32 else 1e-12)


def test_box_cox_gradient_in_lambda_is_stable_near_zero(device):
    """d/dlam (u**lam - 1)/lam -> log(u)**2 / 2. The plain formula cancels, the series branch must not."""
    u = torch.tensor([0.5], dtype=F32, device=device)
    for lam in (0.0, 1e-7, 1e-5):
        l = torch.tensor(lam, dtype=F32, device=device, requires_grad=True)
        B._box_cox(torch.log(u), l).backward()
        assert l.grad.item() == pytest.approx(np.log(0.5) ** 2 / 2, rel=2e-2)


def test_lambda_vector_mixing_zero_and_nonzero(device):
    log_u = torch.log(torch.tensor([0.25, 0.25, 0.25], dtype=F64, device=device))
    lam = torch.tensor([0.0, 0.5, -0.5], dtype=F64, device=device)
    expected = [np.log(0.25), (0.25**0.5 - 1) / 0.5, (0.25**-0.5 - 1) / -0.5]
    torch.testing.assert_close(B._box_cox(log_u, lam), torch.tensor(expected, dtype=F64, device=device))
    expected_pm = [0.0, 0.25**0.5 - 1, 0.25**-0.5 - 1]
    torch.testing.assert_close(B._pow_minus_one(log_u, lam), torch.tensor(expected_pm, dtype=F64, device=device))


@pytest.mark.parametrize("ptype, param", CASES, ids=[f"{t}{p[1:]}" for t, p in CASES])
def test_solver_result_does_not_depend_on_the_start(ptype, param, dtype, device):
    """Table start and warm start only change the iteration count."""
    p = torch.as_tensor(param, dtype=dtype, device=device)
    if not B.validate_gld_params(p, ptype):
        pytest.skip("not a valid parameter set")
    t_true, log_y, log_s = logs(dtype, device, n=4001)
    x = B._quantile_from_logs(log_y, log_s, p, ptype)
    tol = 1e-3 if dtype == F32 else 1e-8
    finite = torch.isfinite(x)
    table = B.solve_logit_cdf(x[finite], p, ptype, xtol=1e-7)
    cold = B.solve_logit_cdf(x[finite], p, ptype, xtol=1e-7, t0=torch.zeros_like(x[finite]))
    # Compare the probabilities: a bounded tail does not determine t once x has saturated.
    torch.testing.assert_close(torch.sigmoid(table), torch.sigmoid(cold), rtol=tol * 10, atol=tol)


def test_solver_with_batched_parameters_matches_one_by_one(device):
    params = torch.tensor([[0.0, 1.0, 0.2, 0.3], [1.0, 2.0, -0.2, -0.1], [0.0, 1.0, 0.0, 0.0]], dtype=F64, device=device)
    x = torch.tensor([[0.5, -1.0, 2.0]], dtype=F64, device=device).T.expand(3, 3).T
    together = B.solve_logit_cdf(x, params, "FMKL")
    for i in range(3):
        alone = B.solve_logit_cdf(x[:, i], params[i], "FMKL")
        torch.testing.assert_close(together[:, i], alone, rtol=1e-6, atol=1e-9)


def test_solver_keeps_the_input_shape_and_special_values(device):
    p = torch.tensor([0.0, 1.0, 0.5, 0.5], dtype=F64, device=device)
    x = torch.tensor([[0.0, float("nan")], [100.0, -100.0]], dtype=F64, device=device)
    t = B.solve_logit_cdf(x, p, "FMKL")
    assert t.shape == x.shape
    assert t[0, 0].abs() < 1e-5 and torch.isnan(t[0, 1])
    assert t[1, 0] == float("inf") and t[1, 1] == -float("inf")


@pytest.mark.parametrize("ptype, param", CASES, ids=[f"{t}{p[1:]}" for t, p in CASES])
def test_solver_iteration_budget(ptype, param, dtype, device, monkeypatch):
    """Table start and the tolerance floor keep the iteration count small in both precisions."""
    p = torch.as_tensor(param, dtype=dtype, device=device)
    if not B.validate_gld_params(p, ptype):
        pytest.skip("not a valid parameter set")
    y = (torch.arange(100_000, dtype=dtype, device=device) + 0.5) / 100_000
    x = B.compute_quantile(y, p, ptype)
    calls = []
    original = B._log_density_from_logs
    monkeypatch.setattr(B, "_log_density_from_logs", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    B.solve_logit_cdf(x, p, ptype)
    assert len(calls) <= 30


@pytest.mark.parametrize("ptype, param", CASES, ids=[f"{t}{p[1:]}" for t, p in CASES])
def test_rounding_noise_does_not_throw_away_a_converged_answer(ptype, param, dtype, device, monkeypatch):
    """A tiny Newton step failing "must halve the previous one" because of rounding noise used to bisect a
    bracket that was still wide, which cost a full climb back (24 iterations in float32)."""
    p = torch.as_tensor(param, dtype=dtype, device=device)
    if not B.validate_gld_params(p, ptype):
        pytest.skip("not a valid parameter set")
    g = torch.Generator(device=device).manual_seed(1)
    log_y, log_s = B._sample_log_uniform((200_000,), dtype, torch.device(device), g)
    x = B._quantile_from_logs(log_y, log_s, p, ptype)
    calls = []
    original = B._log_density_from_logs
    monkeypatch.setattr(B, "_log_density_from_logs", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    t = B.solve_logit_cdf(x, p, ptype)
    assert len(calls) <= 14
    # And the answer stays right: the round trip through Q.
    back = B._quantile_from_logs(F.logsigmoid(t), F.logsigmoid(-t), p, ptype)
    ok = torch.isfinite(back) & torch.isfinite(x)
    assert ((back[ok] - x[ok]).abs() <= 1e-3 * (x[ok].abs() + 1.0)).all()
