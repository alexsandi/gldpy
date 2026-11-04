"""NumPy stand-in for the torch `gpu.GLD`, used when PyTorch is not installed.

Accepts the same calls as the torch class and returns ndarrays. Fits run the SciPy optimizers
of the corrected NumPy class (`_gld_module`), and are slower.
"""
import functools
import warnings
from typing import Optional

import numpy as np

from ._gld_module import GLD as _UpstreamGLD

_MISSING_TORCH = (
    "PyTorch is not installed: gldpy_gpu.gpu.GLD is running on the CPU with NumPy/SciPy. "
    "Install GPU support with: pip install gldpy-gpu[gpu]"
)
_FIT_KW = dict(disp_optimizer=False, disp_fit=False, test_gof=False)
# Shape-parameter starts for when the caller gives no guess (upstream requires one).
_SHAPE_STARTS = (-0.15, 0.1, 0.5, 1.5)


class GLD(_UpstreamGLD):
    """Upstream NumPy GLD with the call signatures of the torch `gpu.GLD`."""

    def __init__(self, param_type: str = "FMKL", device=None, dtype=None):
        super().__init__(param_type)
        if device is not None and str(device) != "cpu":
            warnings.warn(f"device={device!r} is not available (cuda needs torch). " + _MISSING_TORCH, UserWarning, stacklevel=2)
        else:
            warnings.warn(_MISSING_TORCH, UserWarning, stacklevel=2)
        self.device = "cpu"
        self.dtype = np.float64  # the NumPy class computes in float64 only, so `dtype` is ignored

    def CDF_num(self, x, param, xtol: float = 1e-8, max_iter: Optional[int] = None):
        return super().CDF_num(x, param, xtol)

    def SF_num(self, x, param, xtol: float = 1e-8, max_iter: Optional[int] = None):
        return super().SF_num(x, param, xtol)

    def log_PDF_num(self, x, param, xtol: float = 1e-8, max_iter: Optional[int] = None):
        return super().log_PDF_num(x, param, xtol)

    def PDF_num(self, x, param, xtol: float = 1e-8, max_iter: Optional[int] = None):
        return super().PDF_num(x, param, xtol)

    def rand(self, param, size=(1,), generator: Optional[np.random.Generator] = None, random_state=None):
        """`generator` is a NumPy Generator here; `random_state` is upstream's integer seed."""
        if generator is None:
            generator = np.random.default_rng(random_state)
        return self.Q(generator.random(size), param)

    def _best_of_starts(self, data, fit_one):
        """Fit from each shape start, keep the valid result closest to the data (upstream needs a guess)."""
        best, best_gof = None, np.inf
        for l3 in _SHAPE_STARTS:
            for l4 in _SHAPE_STARTS:
                with warnings.catch_warnings(), np.errstate(all="ignore"):
                    warnings.simplefilter("ignore")
                    param = fit_one([l3, l4])
                    valid = bool(self.check_param(param))
                    gof = self.GoF_Q_metric(data, param) if valid else np.inf
                if gof < best_gof:
                    best, best_gof = param, gof
        if best is None:
            raise ValueError("No valid parameters found. Pass an initial_guess.")
        return best

    def fit_LMM(self, data, initial_guess=None, n_grid=None, refine_steps=None, **kw):
        """L-moment fit. `n_grid` and `refine_steps` only apply to the torch backend."""
        data = np.asarray(data, dtype=float)
        kw = {**_FIT_KW, **kw}
        if initial_guess is None and self.param_type != "VSL":
            return self._best_of_starts(data, lambda g: super(GLD, self).fit_LMM(data, g, **kw))
        return super().fit_LMM(data, initial_guess, **kw)

    def fit_PM(self, data, initial_guess=None, u=0.1, n_grid=None, refine_steps=None, **kw):
        """Percentile fit. `n_grid` and `refine_steps` only apply to the torch backend."""
        data = np.asarray(data, dtype=float)
        kw = {**_FIT_KW, **kw}
        if initial_guess is None:
            if self.param_type != "VSL":
                return self._best_of_starts(data, lambda g: super(GLD, self).fit_PM(data, g, u=u, **kw))
            initial_guess = self.fit_LMM(data)[3]
        return super().fit_PM(data, initial_guess, u=u, **kw)

    def fit_ML(self, data, init_method="LMM", steps=None, lr=None, xtol=1e-4, initial_guess=None, **kw):
        """Maximum likelihood with upstream's Nelder-Mead. `steps` caps function evaluations; `lr` is unused."""
        if init_method not in ("LMM", "PM"):
            raise ValueError("init_method must be 'LMM' or 'PM'.")
        data = np.asarray(data, dtype=float)
        if initial_guess is None:
            start = self.fit_LMM(data) if init_method == "LMM" else self.fit_PM(data)
            initial_guess = start[3] if self.param_type == "VSL" else start[2:]
        kw = {**_FIT_KW, **kw}
        if steps is not None:
            kw.setdefault("maxfun", steps)
        return super().fit_ML(data, initial_guess=initial_guess, method=init_method, xtol=xtol, **kw)


def _accept_dtype(method):
    """Accept the torch backend's `dtype` argument. This backend always computes in float64."""

    @functools.wraps(method)
    def wrapper(self, *args, dtype=None, **kwargs):
        return method(self, *args, **kwargs)

    return wrapper


for _name in (
    "Q", "ISF", "PDF_Q", "supp", "CDF_num", "SF_num", "log_PDF_num", "PDF_num", "rand",
    "GoF_Q_metric", "correct_supp", "fit_LMM", "fit_PM", "fit_ML", "check_param",
):
    setattr(GLD, _name, _accept_dtype(getattr(GLD, _name)))
