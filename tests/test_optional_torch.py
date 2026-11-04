"""torch is optional: without it `gpu.GLD` warns and runs on NumPy, with the same call shape."""
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent


# A meta-path blocker, as sys.modules["torch"] = None would break SciPy's array-API shim.
BLOCK_TORCH = """
import sys
class _Block:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ModuleNotFoundError("No module named 'torch'", name="torch")
sys.meta_path.insert(0, _Block())
"""


def run_without_torch(code: str) -> str:
    """Run `code` in a fresh interpreter where `import torch` fails, return stdout."""
    script = BLOCK_TORCH + textwrap.dedent(code)
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    assert res.returncode == 0, res.stderr
    return res.stdout.strip()


def test_torch_is_an_extra_not_a_requirement():
    tomllib = pytest.importorskip("tomllib")  # Python 3.11+
    meta = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert not any(d.startswith("torch") for d in meta["dependencies"])
    assert any(d.startswith("torch") for d in meta["optional-dependencies"]["gpu"])


def test_cpu_gld_works_without_torch():
    out = run_without_torch("""
        import gldpy_gpu
        print(gldpy_gpu.GLD("FMKL").Q(0.5, [0, 1, .2, .3]))
    """)
    assert float(out) == pytest.approx(0.0, abs=0.2)


def test_star_import_works_without_torch():
    run_without_torch("from gldpy_gpu import *; assert GLD and gpu")


@pytest.mark.parametrize("stmt", ["from gldpy_gpu import gpu", "import gldpy_gpu.gpu as gpu"])
def test_gpu_module_falls_back_to_numpy_and_says_so(stmt):
    out = run_without_torch(f"""
        import warnings
        {stmt}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            g = gpu.GLD("FMKL")
        assert any("gldpy-gpu[gpu]" in str(x.message) for x in w), [str(x.message) for x in w]
        print(gpu.BACKEND, type(g.Q([0.5], [0, 1, .2, .3])).__name__)
    """)
    assert out == "numpy ndarray"


def test_backend_reports_torch_when_present():
    pytest.importorskip("torch")
    from gldpy_gpu import gpu

    assert gpu.BACKEND == "torch"


def _fallback():
    from gldpy_gpu._numpy_fallback import GLD

    return GLD


@pytest.fixture(params=["FMKL", "RS", "VSL"])
def fb(request):
    with pytest.warns(UserWarning, match=r"gldpy-gpu\[gpu\]"):
        return _fallback()(request.param)


TRUTH = {"FMKL": [0.0, 1.0, 0.2, 0.3], "RS": [0.0, 0.2, 0.1, 0.15], "VSL": [0.0, 1.0, 0.4, 0.3]}


def test_fallback_accepts_torch_call_shape(fb):
    p = TRUTH[fb.param_type]
    x = fb.rand(p, size=(2000,), generator=np.random.default_rng(1))
    assert x.shape == (2000,)
    assert fb.CDF_num(x[:5], p, xtol=1e-8, max_iter=40).shape == (5,)
    assert fb.PDF_num(x[:5], p, xtol=1e-8, max_iter=40).shape == (5,)
    for fit in (fb.fit_LMM(x, n_grid=64, refine_steps=10), fb.fit_PM(x, u=0.1, n_grid=64)):
        assert fit.shape == (4,) and fb.check_param(fit)


def test_fallback_fits_need_no_initial_guess_and_recover_truth(fb):
    p = TRUTH[fb.param_type]
    x = fb.rand(p, size=5000, generator=np.random.default_rng(2))
    fit = fb.fit_LMM(x)
    assert fb.GoF_Q_metric(x, fit) < max(3 * fb.GoF_Q_metric(x, p), 1e-2)


def test_fallback_fit_ML_uses_same_call_shape():
    with pytest.warns(UserWarning):
        g = _fallback()("VSL")
    p = TRUTH["VSL"]
    x = g.rand(p, size=100, generator=np.random.default_rng(3))
    assert g.check_param(g.fit_ML(x, init_method="LMM", steps=20))


def test_fallback_warns_when_gpu_explicitly_requested():
    with pytest.warns(UserWarning, match="cuda"):
        _fallback()("FMKL", device="cuda")


def test_fallback_upper_tail_is_not_saturated():
    with pytest.warns(UserWarning):
        g = _fallback()("FMKL")
    p = [0.0, 1.0, -0.2, -0.1]
    s = np.array([1e-4, 1e-10, 1e-14])
    x = g.ISF(s, p)
    np.testing.assert_allclose(g.SF_num(x, p), s, rtol=1e-6)
