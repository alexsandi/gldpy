"""Fitting recovers a distribution that explains the data as well as the truth does."""
import numpy as np
import pytest
import torch

from gldpy_gpu import gpu


def _fit_quality(gld, data, fitted, truth, device):
    d = torch.as_tensor(data, dtype=torch.float64, device=device)
    f = torch.as_tensor(fitted.detach().cpu().numpy(), dtype=torch.float64, device=device)
    tr = torch.as_tensor(truth, dtype=torch.float64, device=device)
    return gld.GoF_Q_metric(d, f).item(), gld.GoF_Q_metric(d, tr).item()


def _data(cpu_gld, param, n=5000):
    return cpu_gld.rand(param, size=n, random_state=11)


def test_fit_LMM_returns_valid_params(ptype, cpu_gld, gpu_gld, param, device):
    data = _data(cpu_gld, param)
    fit = gpu_gld.fit_LMM(torch.as_tensor(data, dtype=torch.float64, device=device))
    assert fit.shape == (4,)
    assert gpu_gld.check_param(fit)


def test_fit_LMM_explains_data(ptype, cpu_gld, gpu_gld, param, device):
    data = _data(cpu_gld, param)
    fit = gpu_gld.fit_LMM(torch.as_tensor(data, dtype=torch.float64, device=device))
    got, ref = _fit_quality(gpu_gld, data, fit, param, device)
    assert got < max(10 * ref, 0.05)


def test_fit_PM_explains_data(ptype, cpu_gld, gpu_gld, param, device):
    data = _data(cpu_gld, param)
    fit = gpu_gld.fit_PM(torch.as_tensor(data, dtype=torch.float64, device=device))
    assert gpu_gld.check_param(fit)
    got, ref = _fit_quality(gpu_gld, data, fit, param, device)
    assert got < max(20 * ref, 0.1)


def test_fit_ML_not_worse_than_init(ptype, cpu_gld, gpu_gld, param, device):
    data = torch.as_tensor(_data(cpu_gld, param, 2000), dtype=torch.float64, device=device)

    def nll(p):
        y = gpu_gld.CDF_num(data, p, xtol=1e-10, max_iter=60).clamp(1e-6, 1 - 1e-6)
        return -torch.log(gpu.compute_pdf_quantile(y, p, ptype).clamp_min(1e-12)).mean().item()

    init = gpu_gld.fit_LMM(data)
    fit = gpu_gld.fit_ML(data, steps=30)
    assert nll(fit) <= nll(init) + 1e-6


def test_fit_ML_bad_init_method(gpu_gld, device):
    with pytest.raises(ValueError):
        gpu_gld.fit_ML(torch.rand(100, device=device), init_method="nope")


def test_fit_LMM_float32_input(ptype, cpu_gld, gpu_gld, param, device):
    data = torch.as_tensor(_data(cpu_gld, param), dtype=torch.float32, device=device)
    fit = gpu_gld.fit_LMM(data)
    assert torch.isfinite(fit).all()
