"""GPU (torch) implementation must agree with the NumPy reference."""
import numpy as np
import pytest
import torch

from gldpy_gpu import gpu

TOL = dict(rtol=1e-5, atol=1e-6)


def t(x, device):
    return torch.as_tensor(x, dtype=torch.float64, device=device)


def test_invalid_param_type():
    with pytest.raises(ValueError):
        gpu.GLD("nope")


def test_check_param_matches_cpu(ptype, cpu_gld, gpu_gld):
    rng = np.random.default_rng(0)
    cands = rng.uniform(-3, 3, size=(300, 4))
    for p in cands:
        assert gpu_gld.check_param(t(p, gpu_gld.device)) == bool(cpu_gld.check_param(p)), p


def test_check_param_rejects_nan(gpu_gld, device):
    assert not gpu_gld.check_param(t([0, float("nan"), 0.2, 0.3], device))


def test_Q_matches_cpu(cpu_gld, gpu_gld, param, device):
    y = np.linspace(0.01, 0.99, 99)
    out = gpu_gld.Q(t(y, device), t(param, device)).cpu().numpy()
    np.testing.assert_allclose(out, cpu_gld.Q(y, param), **TOL)


def test_Q_rejects_out_of_range(gpu_gld, param, device):
    with pytest.raises(ValueError):
        gpu_gld.Q(t([1.5], device), t(param, device))


def test_Q_rejects_invalid_params(gpu_gld, device):
    with pytest.raises(ValueError):
        gpu_gld.Q(t([0.5], device), t([0, -1.0, 0.2, 0.3], device))


def test_PDF_Q_matches_cpu(cpu_gld, gpu_gld, param, device):
    y = np.linspace(0.01, 0.99, 99)
    out = gpu_gld.PDF_Q(t(y, device), t(param, device)).cpu().numpy()
    np.testing.assert_allclose(out, cpu_gld.PDF_Q(y, param), **TOL)


def test_supp_matches_cpu(cpu_gld, gpu_gld, param, device):
    lo, hi = gpu_gld.supp(t(param, device))
    clo, chi = cpu_gld.supp(param)
    np.testing.assert_allclose([lo.item(), hi.item()], [clo, chi], **TOL)


def test_CDF_num_roundtrip(gpu_gld, param, device):
    p = t(param, device)
    y = torch.linspace(0.02, 0.98, 50, dtype=torch.float64, device=device)
    x = gpu_gld.Q(y, p)
    back = gpu_gld.CDF_num(x, p, xtol=1e-10, max_iter=60)
    torch.testing.assert_close(back, y, rtol=1e-6, atol=1e-6)


def test_CDF_num_matches_cpu(cpu_gld, gpu_gld, param, device):
    x = cpu_gld.Q(np.linspace(0.05, 0.95, 40), param)
    out = gpu_gld.CDF_num(t(x, device), t(param, device), xtol=1e-9, max_iter=60).cpu().numpy()
    np.testing.assert_allclose(out, cpu_gld.CDF_num(x, param), atol=1e-4)


def test_PDF_num_matches_cpu(cpu_gld, gpu_gld, param, device):
    x = cpu_gld.Q(np.linspace(0.05, 0.95, 40), param)
    out = gpu_gld.PDF_num(t(x, device), t(param, device), xtol=1e-9, max_iter=60).cpu().numpy()
    np.testing.assert_allclose(out, cpu_gld.PDF_num(x, param), rtol=1e-3, atol=1e-4)


def test_pdf_zero_outside_support(gpu_gld, device):
    # FMKL with positive shape params has finite support.
    if gpu_gld.param_type != "FMKL":
        pytest.skip("finite-support case only built for FMKL")
    p = t([0.0, 1.0, 0.5, 0.5], device)
    lo, hi = gpu_gld.supp(p)
    x = torch.stack([lo - 1.0, hi + 1.0])
    assert torch.all(gpu_gld.PDF_num(x, p) == 0)


def test_rand_shape_dtype_and_reproducibility(gpu_gld, param, device):
    p = t(param, device)
    g1 = torch.Generator(device=device).manual_seed(7)
    g2 = torch.Generator(device=device).manual_seed(7)
    a = gpu_gld.rand(p, size=(5, 3), generator=g1)
    b = gpu_gld.rand(p, size=(5, 3), generator=g2)
    assert a.shape == (5, 3) and a.dtype == torch.float64 and a.device.type == device.type
    torch.testing.assert_close(a, b)


def test_rand_int_size(gpu_gld, param, device):
    assert gpu_gld.rand(t(param, device), size=10).shape == (10,)


def test_rand_distribution_matches_quantiles(gpu_gld, param, device):
    p = t(param, device)
    g = torch.Generator(device=device).manual_seed(0)
    x = gpu_gld.rand(p, size=200_000, generator=g)
    for q in (0.1, 0.5, 0.9):
        emp = torch.quantile(x[:100_000].cpu(), q).item()
        theo = gpu_gld.Q(t([q], device), p).item()
        assert emp == pytest.approx(theo, abs=0.05 * (1 + abs(theo)))


def test_GoF_matches_cpu(cpu_gld, gpu_gld, param, sample, device):
    out = gpu_gld.GoF_Q_metric(t(sample, device), t(param, device)).item()
    assert out == pytest.approx(cpu_gld.GoF_Q_metric(sample, param), rel=1e-5)
