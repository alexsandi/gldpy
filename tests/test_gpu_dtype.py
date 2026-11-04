"""Precision is the caller's choice: a method computes in the dtype it was given, and only in that."""
import numpy as np
import pytest
import torch
from torch.overrides import TorchFunctionMode

from gldpy_gpu import GLD, gpu

F32, F64 = torch.float32, torch.float64
PARAM = {"FMKL": [0.0, 1.0, 0.2, 0.3], "RS": [0.0, 0.2, 0.1, 0.15], "VSL": [0.0, 1.0, 0.4, 0.3]}


def walk(out):
    if isinstance(out, torch.Tensor):
        yield out
    elif isinstance(out, (tuple, list)):
        for o in out:
            yield from walk(o)


class ForbidDtype(TorchFunctionMode):
    """Fails on any tensor of `dtype` created while active."""

    def __init__(self, dtype, allow_scalars=False):
        super().__init__()
        self.dtype = dtype
        self.allow_scalars = allow_scalars  # torch.optim.Adam keeps its step counter in the default dtype

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        for t in walk(out):
            if not (self.allow_scalars and t.dim() == 0):
                assert t.dtype != self.dtype, f"{func} produced {self.dtype}"
        return out


CALLS = {
    "Q": lambda g, y, x, p, **k: g.Q(y, p, **k),
    "ISF": lambda g, y, x, p, **k: g.ISF(y, p, **k),
    "PDF_Q": lambda g, y, x, p, **k: g.PDF_Q(y, p, **k),
    "CDF_num": lambda g, y, x, p, **k: g.CDF_num(x, p, **k),
    "SF_num": lambda g, y, x, p, **k: g.SF_num(x, p, **k),
    "PDF_num": lambda g, y, x, p, **k: g.PDF_num(x, p, **k),
    "log_PDF_num": lambda g, y, x, p, **k: g.log_PDF_num(x, p, **k),
    "rand": lambda g, y, x, p, **k: g.rand(p, size=1000, **k),
    "GoF_Q_metric": lambda g, y, x, p, **k: g.GoF_Q_metric(x, p, **k),
    "correct_supp": lambda g, y, x, p, **k: g.correct_supp(x, p, **k),
    "supp": lambda g, y, x, p, **k: g.supp(p, **k)[0],
    "fit_LMM": lambda g, y, x, p, **k: g.fit_LMM(x, **k),
    "fit_PM": lambda g, y, x, p, **k: g.fit_PM(x, **k),
    "fit_ML": lambda g, y, x, p, **k: g.fit_ML(x, steps=3, **k),
}


def inputs(ptype, dtype, device):
    y = torch.linspace(0.05, 0.95, 400, dtype=dtype, device=device)
    p = torch.as_tensor(PARAM[ptype], dtype=dtype, device=device)
    x = gpu.compute_quantile(y, p, ptype)
    return y, x, p


@pytest.mark.parametrize("name", sorted(CALLS))
def test_float32_call_never_touches_float64(name, ptype, device):
    g = gpu.GLD(ptype, device=device)
    y, x, p = inputs(ptype, F32, device)
    with ForbidDtype(F64):
        out = CALLS[name](g, y, x, p)
    assert out.dtype == F32


@pytest.mark.parametrize("name", sorted(CALLS))
def test_float64_call_never_touches_float32(name, ptype, device):
    g = gpu.GLD(ptype, device=device)
    y, x, p = inputs(ptype, F64, device)
    with ForbidDtype(F32, allow_scalars=name == "fit_ML"):
        out = CALLS[name](g, y, x, p)
    assert out.dtype == F64


@pytest.mark.parametrize("name", sorted(CALLS))
@pytest.mark.parametrize("given, asked", [(F32, F64), (F64, F32)], ids=["f32_to_f64", "f64_to_f32"])
def test_per_call_dtype_overrides_the_inputs(name, given, asked, ptype, device):
    g = gpu.GLD(ptype, device=device)
    y, x, p = inputs(ptype, given, device)
    assert CALLS[name](g, y, x, p, dtype=asked).dtype == asked


def test_precedence_call_then_constructor_then_inputs_then_default(device):
    p = [0.0, 1.0, 0.2, 0.3]
    assert gpu.GLD("FMKL", device=device).Q([0.5], p).dtype == F32
    assert gpu.GLD("FMKL", device=device).Q(np.array([0.5]), p).dtype == F64
    assert gpu.GLD("FMKL", device=device, dtype=F32).Q(np.array([0.5]), p).dtype == F32
    assert gpu.GLD("FMKL", device=device, dtype=F32).Q([0.5], p, dtype=F64).dtype == F64


@pytest.mark.parametrize("name, dt", [("float32", F32), ("float64", F64)])
def test_dtype_accepts_a_name(name, dt, device):
    assert gpu.GLD("FMKL", device=device).Q([0.5], [0.0, 1.0, 0.2, 0.3], dtype=name).dtype == dt
    assert gpu.GLD("FMKL", device=device, dtype=name).Q([0.5], [0.0, 1.0, 0.2, 0.3]).dtype == dt


def test_unknown_dtype_is_rejected(device):
    with pytest.raises(ValueError):
        gpu.GLD("FMKL", device=device, dtype="float16")
    with pytest.raises(ValueError):
        gpu.GLD("FMKL", device=device).Q([0.5], [0.0, 1.0, 0.2, 0.3], dtype=torch.int32)


def test_promoting_at_the_call_recovers_float64_accuracy(ptype, device):
    """float32 stores the data; asking for float64 runs the solve there."""
    from conftest import PARAMS

    data = GLD(ptype).rand(np.array(PARAMS[ptype]), size=5000, random_state=11)
    g = gpu.GLD(ptype, device=device)
    reference = g.fit_LMM(torch.as_tensor(data.astype(np.float32).astype(float), device=device))
    promoted = g.fit_LMM(torch.as_tensor(data, dtype=F32, device=device), dtype=F64)
    assert promoted.dtype == F64
    torch.testing.assert_close(promoted, reference, rtol=1e-10, atol=1e-10)


def test_fallback_accepts_dtype_everywhere_and_stays_float64():
    from gldpy_gpu._numpy_fallback import GLD as Fallback

    with pytest.warns(UserWarning):
        g = Fallback("VSL")
    p = PARAM["VSL"]
    x = g.rand(p, size=200, dtype="float32")
    assert x.dtype == np.float64
    y = np.array([0.1, 0.9])
    for out in (
        g.Q(y, p, dtype="float32"),
        g.ISF(y, p, dtype="float32"),
        g.PDF_Q(y, p, dtype="float32"),
        g.CDF_num(x[:3], p, dtype="float32"),
        g.SF_num(x[:3], p, dtype="float32"),
        g.PDF_num(x[:3], p, dtype="float32"),
        g.log_PDF_num(x[:3], p, dtype="float32"),
        g.correct_supp(x, p, dtype="float32"),
        g.fit_LMM(x, dtype="float32"),
        g.fit_PM(x, dtype="float32"),
    ):
        assert out.dtype == np.float64
    assert np.isfinite(g.GoF_Q_metric(x, p, dtype="float32"))
    assert g.supp(p, dtype="float32")[0] < g.supp(p, dtype="float32")[1]
