import pytest
import torch

from gldpy_gpu import gpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_outputs_live_on_cuda():
    g = gpu.GLD("FMKL", device=torch.device("cuda"))
    out = g.Q([0.1, 0.5, 0.9], [0.0, 1.0, 0.2, 0.3])
    assert out.device.type == "cuda"


def test_cpu_tensor_moved_to_requested_device():
    g = gpu.GLD("FMKL", device=torch.device("cpu"))
    out = g.Q(torch.tensor([0.5]), torch.tensor([0.0, 1.0, 0.2, 0.3]))
    assert out.device.type == "cpu"


def test_default_device_is_cuda_when_available():
    g = gpu.GLD("FMKL")
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert g.device.type == expected


def _fake_cuda(monkeypatch, capability, archs):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: capability)
    monkeypatch.setattr(torch.cuda, "get_arch_list", lambda: archs)


def test_cuda_usable_when_cubin_matches_major(monkeypatch):
    _fake_cuda(monkeypatch, (6, 1), ["sm_50", "sm_60", "sm_70"])
    assert gpu._cuda_usable()


def test_cuda_unusable_when_arch_dropped(monkeypatch):
    # Pascal card with a torch build that only ships Turing and newer.
    _fake_cuda(monkeypatch, (6, 1), ["sm_75", "sm_80", "sm_90"])
    assert not gpu._cuda_usable()


def test_ptx_forward_compatibility(monkeypatch):
    _fake_cuda(monkeypatch, (12, 0), ["sm_80", "compute_90"])
    assert gpu._cuda_usable()


def test_unusable_cuda_falls_back_to_cpu_with_warning(monkeypatch):
    _fake_cuda(monkeypatch, (6, 1), ["sm_75", "sm_80"])
    with pytest.warns(UserWarning, match="cu126"):
        g = gpu.GLD("FMKL")
    assert g.device.type == "cpu"
