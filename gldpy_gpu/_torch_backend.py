import math
import warnings
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F


def _cuda_usable() -> bool:
    """True if CUDA is present AND the installed torch build ships kernels for this GPU.

    A cubin `sm_XY` runs on devices with the same major version and minor >= Y;
    PTX `compute_XY` can be JIT-compiled for any device with capability >= XY.
    """
    if not torch.cuda.is_available():
        return False
    cap = tuple(torch.cuda.get_device_capability())
    for arch in torch.cuda.get_arch_list():
        kind, _, num = arch.partition("_")
        if not num.isdigit():
            continue
        a = (int(num) // 10, int(num) % 10)
        if kind == "sm" and a[0] == cap[0] and a[1] <= cap[1]:
            return True
        if kind == "compute" and a <= cap:
            return True
    return False


def _default_device() -> torch.device:
    if _cuda_usable():
        return torch.device("cuda")
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        warnings.warn(
            f"CUDA device with compute capability {cap[0]}.{cap[1]} found, but this PyTorch build "
            f"({torch.__version__}) has no kernels for it (supports {torch.cuda.get_arch_list()}). "
            "Falling back to CPU. For Pascal GPUs (GTX 10xx) install a CUDA 12.6 build: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu126",
            UserWarning,
            stacklevel=3,
        )
    return torch.device("cpu")


def _with_support_covering(param: torch.Tensor, z_min: torch.Tensor, z_max: torch.Tensor, param_type: str) -> torch.Tensor:
    """Smallest location/scale change that puts [z_min, z_max] strictly inside the support.

    Q is affine in location and scale, so this is exact, and the shape parameters do not change.
    An infinite end is never violated, and a single finite end only needs a shift.
    """
    zero = torch.zeros_like(param[0])
    inf = torch.full_like(zero, float("inf"))
    lo = _quantile_from_logs(-inf, zero, param, param_type)
    hi = _quantile_from_logs(zero, -inf, param, param_type)
    margin = 1e-6 * (z_max - z_min)
    new_lo = torch.minimum(lo, z_min - margin)
    new_hi = torch.maximum(hi, z_max + margin)
    if torch.isfinite(lo) and torch.isfinite(hi):
        k = (new_hi - new_lo) / (hi - lo)
        scale = param[1] * k if param_type == "VSL" else param[1] / k
        return torch.stack([new_lo - (lo - param[0]) * k, scale, param[2], param[3]])
    shift = new_lo - lo if torch.isfinite(lo) else (new_hi - hi if torch.isfinite(hi) else zero)
    return torch.stack([param[0] + shift, param[1], param[2], param[3]])


def _as_dtype(dtype: Union[None, str, torch.dtype]) -> Optional[torch.dtype]:
    """torch.float32/float64 or their names ("float32", "float64"), which also work without torch."""
    if dtype is None:
        return None
    known = {"float32": torch.float32, "float64": torch.float64, torch.float32: torch.float32, torch.float64: torch.float64}
    if dtype not in known:
        raise ValueError(f"dtype must be float32 or float64, got {dtype!r}.")
    return known[dtype]


# =====================================================================
# PHASE 1: PARAMETER VALIDATION HELPERS
# =====================================================================

# [STABLE HELPER]
def validate_gld_params(param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Vectorized validity check for GLD parameters. Supports shape (..., 4)."""
    if param.shape[-1] != 4:
        raise ValueError("GLD requires 4 parameters along the last dimension.")
    finite_mask = torch.isfinite(param).all(dim=-1)
    p1, p2, p3, p4 = param.unbind(dim=-1)

    if param_type == "FMKL":
        return finite_mask & (p2 > 0)
    if param_type == "VSL":
        return finite_mask & (p2 > 0) & (p3 >= 0.0) & (p3 <= 1.0)
    if param_type == "RS":
        r1 = (p2 < 0) & (p3 <= -1) & (p4 >= 1)
        r2 = (p2 < 0) & (p3 >= 1) & (p4 <= -1)
        r3 = (p2 > 0) & (p3 >= 0) & (p4 >= 0) & ((p3 != 0) | (p4 != 0))
        r4 = (p2 < 0) & (p3 <= 0) & (p4 <= 0) & ((p3 != 0) | (p4 != 0))

        # Compared in log space: the powers themselves overflow float32 for lambda ~ 30.
        def lhs_le_rhs(neg, pos):
            neg_c = torch.clamp(neg, min=-1.0, max=0.0)
            pos_c = torch.clamp(pos, min=1.0)
            xlogy = torch.special.xlogy
            log_lhs = (
                xlogy(1.0 - neg_c, 1.0 - neg_c)
                + xlogy(pos_c - 1.0, pos_c - 1.0)
                - xlogy(pos_c - neg_c, pos_c - neg_c)
            )
            return log_lhs <= torch.log(-neg_c / pos_c)

        r5 = (p2 < 0) & (p3 <= 0) & (p3 >= -1) & (p4 >= 1) & lhs_le_rhs(p3, p4)
        r6 = (p2 < 0) & (p3 >= 1) & (p4 >= -1) & (p4 <= 0) & lhs_le_rhs(p4, p3)

        return finite_mask & (r1 | r2 | r3 | r4 | r5 | r6)
    raise ValueError(f"Unknown parameterization '{param_type}'.")


# =====================================================================
# PHASE 2: CORE QUANTILE, DENSITY & ROOT-FINDING TRANSFORMATIONS
# =====================================================================

_START_TABLE_SIZE = 512
_SMALL_STEP_FACTOR = 1000.0

# Everything below is written in terms of log(y) and log(1 - y). Tail probabilities keep their
# relative accuracy that way, because neither `1 - y` nor a power of a tiny number is ever formed.

def _no_zero(values: torch.Tensor) -> bool:
    return bool((values != 0).all())


def _pow_from_log(log_u: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """u**lam from log(u), with 0**0 = 1 as in NumPy."""
    if _no_zero(lam):
        return torch.exp(lam * log_u)
    return torch.where(lam == 0, torch.ones_like(lam * 1.0), torch.exp(lam * log_u))


def _pow_minus_one(log_u: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """u**lam - 1 from log(u). RS uses y**a - s**b = (y**a - 1) - (s**b - 1), which avoids subtracting two numbers near 1."""
    if _no_zero(lam):
        return torch.expm1(lam * log_u)
    return torch.where(lam == 0, torch.zeros_like(lam * 1.0), torch.expm1(lam * log_u))


def _box_cox(log_u: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """(u**lam - 1) / lam from log(u). No cancellation as lam -> 0, and autograd-safe in lam."""
    tracks_lam = torch.is_grad_enabled() and lam.requires_grad
    if not tracks_lam and _no_zero(lam):
        return torch.expm1(lam * log_u) / lam  # expm1 keeps every digit; only d/dlam needs the series
    z = lam * log_u
    small = (lam == 0) | (z.abs() < 1e-3)
    z_series = lam * torch.where(torch.isinf(log_u), torch.zeros_like(log_u), log_u)
    series = log_u * (1.0 + z_series / 2.0 + z_series * z_series / 6.0)
    safe_lam = torch.where(lam == 0, torch.ones_like(lam), lam)
    return torch.where(small, series, torch.expm1(torch.where(lam == 0, torch.zeros_like(z), z)) / safe_lam)


def _box_cox_power(u: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    """(u**lam - 1) / lam."""
    return _box_cox(torch.log(u), lam)


def _quantile_from_logs(log_y: torch.Tensor, log_s: torch.Tensor, param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Q from log(y) and log(s), where s = 1 - y."""
    p1, p2, p3, p4 = param.unbind(dim=-1)
    if param_type == "RS":
        return p1 + (_pow_minus_one(log_y, p3) - _pow_minus_one(log_s, p4)) / p2
    if param_type == "FMKL":
        return p1 + (_box_cox(log_y, p3) - _box_cox(log_s, p4)) / p2
    if param_type == "VSL":
        lower, upper = (1.0 - p3) * _box_cox(log_y, p4), p3 * _box_cox(log_s, p4)
        if not (_no_zero(p3) and _no_zero(1.0 - p3)):
            # delta = 0 or 1 removes a term entirely, also where its box-cox value is infinite.
            lower = torch.where(p3 == 1, torch.zeros_like(lower), lower)
            upper = torch.where(p3 == 0, torch.zeros_like(upper), upper)
        return p1 + p2 * (lower - upper)
    raise ValueError(f"Unknown parameterization '{param_type}'.")


def _density_terms(param: torch.Tensor, param_type: str) -> Tuple[torch.Tensor, ...]:
    """(c3, c4, e3, e4, log_scale) such that f(y) = exp(log_scale) / |c3 y**e3 + c4 (1 - y)**e4|."""
    p1, p2, p3, p4 = param.unbind(dim=-1)
    if param_type == "RS":
        return p3, p4, p3 - 1.0, p4 - 1.0, torch.log(p2.abs())
    if param_type == "FMKL":
        return torch.ones_like(p3), torch.ones_like(p4), p3 - 1.0, p4 - 1.0, torch.log(p2)
    if param_type == "VSL":
        return 1.0 - p3, p3, p4 - 1.0, p4 - 1.0, -torch.log(p2)
    raise ValueError(f"Unknown parameterization '{param_type}'.")


def _log_density_from_logs(
    log_y: torch.Tensor, log_s: torch.Tensor, param: torch.Tensor, param_type: str, interior: bool = False
) -> torch.Tensor:
    """log f(y) = -log Q'(y). Q' = c3 y**e3 + c4 s**e4 is summed after factoring out its larger term.

    `interior=True` promises finite log_y and log_s, which skips the handling of the endpoints.
    """
    c3, c4, e3, e4, log_scale = _density_terms(param, param_type)
    if interior and _no_zero(c3) and _no_zero(c4):
        term3, term4 = e3 * log_y, e4 * log_s
        log_c3, log_c4 = torch.log(c3.abs()), torch.log(c4.abs())
        m = torch.maximum(log_c3 + term3, log_c4 + term4).detach()
        return log_scale - m - torch.log((c3 * torch.exp(term3 - m) + c4 * torch.exp(term4 - m)).abs())

    # 0 * inf terms are 0: a term with zero exponent is constant even at y = 0.
    term3 = torch.where(e3 == 0, torch.zeros_like(e3 * 1.0), e3 * log_y)
    term4 = torch.where(e4 == 0, torch.zeros_like(e4 * 1.0), e4 * log_s)
    # `m` is only a scaling constant, so it is detached. log|c| has an infinite gradient at c = 0
    # (VSL delta on its bounds), which would turn into NaN if it were differentiated through max().
    neg_inf = torch.full_like(log_scale, -float("inf"))
    m = torch.maximum(
        torch.where(c3 == 0, neg_inf, torch.log(c3.abs()) + term3),
        torch.where(c4 == 0, neg_inf, torch.log(c4.abs()) + term4),
    ).detach()
    ok = torch.isfinite(m)  # at y = 0 or 1 one term is infinite and decides alone
    m0 = torch.where(ok, m, torch.zeros_like(m))
    limit = 0.5 * torch.log(torch.as_tensor(torch.finfo(m.dtype).max, dtype=m.dtype, device=m.device))
    exp3 = torch.exp(torch.clamp(torch.where(ok, term3 - m0, torch.zeros_like(m0)), max=limit))
    exp4 = torch.exp(torch.clamp(torch.where(ok, term4 - m0, torch.zeros_like(m0)), max=limit))
    inner = torch.log((c3 * exp3 + c4 * exp4).abs())
    return log_scale - m - torch.where(ok, inner, torch.zeros_like(inner))


def compute_quantile(y: torch.Tensor, param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Evaluate quantile function Q(y) with full broadcasting over y and param."""
    return _quantile_from_logs(torch.log(y), torch.log1p(-y), param, param_type)


def compute_pdf_quantile(y: torch.Tensor, param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Evaluate density quantile function f_Q(y) = 1 / Q'(y). Underflows to 0 far out; see `log_PDF_num`."""
    c3, c4, e3, e4, log_scale = _density_terms(param, param_type)
    zero = torch.zeros_like(log_scale)
    term3 = torch.where(c3 == 0, zero, c3 * _pow_from_log(torch.log(y), e3))
    term4 = torch.where(c4 == 0, zero, c4 * _pow_from_log(torch.log1p(-y), e4))
    return torch.exp(log_scale) / (term3 + term4).abs()


def _table_start(x: torch.Tensor, param: torch.Tensor, param_type: str, bound: torch.Tensor) -> torch.Tensor:
    """Starting t for each x, by linear interpolation in a small table of Q(t).

    Q is monotone, so a binary search finds the cell. The start is then close enough that about
    three Newton steps finish, instead of the 6 to 13 it takes from t = 0.
    """
    limit = min(40.0, bound.item())
    grid = torch.linspace(-limit, limit, _START_TABLE_SIZE, dtype=x.dtype, device=x.device)
    table = _quantile_from_logs(F.logsigmoid(grid), F.logsigmoid(-grid), param, param_type)
    above = torch.searchsorted(table, x).clamp(1, _START_TABLE_SIZE - 1)
    below = above - 1
    weight = torch.clamp((x - table[below]) / (table[above] - table[below]), 0.0, 1.0)
    start = grid[below] + weight * (grid[above] - grid[below])
    return torch.where(torch.isfinite(start), start, torch.zeros_like(start))


@torch.no_grad()
def solve_logit_cdf(
    x: torch.Tensor,
    param: torch.Tensor,
    param_type: str,
    xtol: float = 1e-6,
    max_iter: int = 64,
    t0: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Solve Q(sigmoid(t)) = x element-wise with a safeguarded Newton iteration.

    Working on t = logit(y) makes `xtol` a relative tolerance on the smaller of y and 1 - y, so
    both tails converge alike and F, 1 - F and their logs all come out of t without cancellation.
    `xtol` has a floor of 32 eps. Each element stops on its own, and finished elements leave the arrays, so the cost of an
    iteration follows the number of elements still unresolved.

    Returns -inf/+inf at or beyond the support, NaN for NaN input, and saturates at t = +-log(tiny),
    i.e. tail probabilities below ~1e-38 (float32) or ~1e-308 (float64) report as that floor.
    `t0` is an optional starting point (a warm start); by default a table supplies one.
    Not differentiable: `fit_ML` shows how to get gradients from the implicit function theorem.
    """
    finfo = torch.finfo(x.dtype)
    bound = -torch.log(torch.as_tensor(finfo.tiny, dtype=x.dtype, device=x.device))
    tol = max(float(xtol), 32.0 * finfo.eps)  # below this, steps are rounding noise and never settle
    zero = torch.zeros(1, dtype=x.dtype, device=x.device)
    inf = torch.full_like(zero, float("inf"))
    q_min = _quantile_from_logs(-inf, zero, param, param_type)
    q_max = _quantile_from_logs(zero, -inf, param, param_type)
    x, q_min, q_max = torch.broadcast_tensors(x, q_min, q_max)

    below, above, missing = x <= q_min, x >= q_max, torch.isnan(x)
    result = torch.where(below, -inf, torch.where(above, inf, x))  # x itself is only kept where it is NaN
    open_elements = ~(below | above | missing)
    positions = torch.nonzero(open_elements, as_tuple=True)
    x_a = x[positions]
    if not x_a.numel():
        return result

    # A single parameter set stays one vector, and a batch is indexed along with the elements.
    single = param.dim() == 1
    p_a = param if single else param.expand(*x.shape, 4)[positions]
    if t0 is not None:
        start = torch.broadcast_to(t0, x.shape)[positions]
        t = torch.where(torch.isfinite(start), start, torch.zeros_like(start)).clamp(-bound, bound)
    elif single:
        t = _table_start(x_a, param, param_type, bound)
    else:
        t = torch.zeros_like(x_a)
    low, high = torch.full_like(x_a, -bound), torch.full_like(x_a, bound)
    prev_step = torch.full_like(x_a, 2.0 * bound)
    done = torch.zeros_like(x_a, dtype=torch.bool)

    def retire(mask: torch.Tensor) -> None:
        result[tuple(index[mask] for index in positions)] = t[mask]

    for _ in range(max_iter):
        log_y, log_s = F.logsigmoid(t), F.logsigmoid(-t)
        q = _quantile_from_logs(log_y, log_s, p_a, param_type)
        err = q - x_a
        low = torch.where(err < 0.0, t, low)
        high = torch.where(err >= 0.0, t, high)

        # dt/dx = f(y) / (y (1 - y))
        log_density = _log_density_from_logs(log_y, log_s, p_a, param_type, interior=True)
        newton = -err * torch.exp(log_density - log_y - log_s)
        candidate = t + newton
        use_newton = (
            torch.isfinite(candidate)
            & (candidate > low)
            & (candidate < high)
            # A step this small cannot diverge, and noise in it would otherwise trigger a bisection
            # of a bracket that is still wide, losing an answer that had already converged.
            & ((newton.abs() <= 0.5 * prev_step.abs()) | (newton.abs() <= _SMALL_STEP_FACTOR * tol))
        )
        t_next = torch.where(use_newton, candidate, 0.5 * (low + high))
        step = t_next - t
        # A root met to rounding error keeps t: `low` or `high` now equals t, and a further bisection would leave it.
        at_root = err.abs() <= 2.0 * finfo.eps * (x_a.abs() + q.abs())
        t = torch.where(at_root, t, t_next)
        prev_step = step
        done = at_root | (step.abs() <= tol) | (high - low <= tol)

        finished = int(done.sum())
        if finished == done.numel():
            break
        if 4 * finished >= done.numel():  # compact only when it pays for the indexing
            retire(done)
            keep = ~done
            positions = tuple(index[keep] for index in positions)
            x_a, t, low, high, prev_step = x_a[keep], t[keep], low[keep], high[keep], prev_step[keep]
            if not single:
                p_a = p_a[keep]

    retire(torch.ones_like(t, dtype=torch.bool))
    return result


def invert_quantile_newton_bisect(
    x: torch.Tensor,
    param: torch.Tensor,
    param_type: str,
    xtol: float = 1e-6,
    max_iter: int = 64,
) -> torch.Tensor:
    """Vectorized F(x), the inverse of Q (see `solve_logit_cdf`)."""
    return torch.sigmoid(solve_logit_cdf(x, param, param_type, xtol=xtol, max_iter=max_iter))


def _plotting_logs(k: torch.Tensor, n: int, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    """log and log1p(-) of (k + 0.5) / n for integer k.

    y and 1 - y come from k and n - k separately, so the upper positions keep their relative
    accuracy in any precision, instead of being the difference of two numbers close to 1.
    """
    log_n = math.log(n)
    return torch.log(k.to(dtype) + 0.5) - log_n, torch.log((n - k).to(dtype) - 0.5) - log_n


def _sample_log_uniform(size: Tuple[int, ...], dtype: torch.dtype, device: torch.device, generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """log(y) and log(1 - y) for y uniform on (0, 1), with relative accuracy in both tails.

    A plain uniform has a fixed step (2**-24 in float32), so it can be exactly 0, which gives
    Q = -inf, and it stops the upper tail at Q(1 - 2**-24). Here the distance to the nearer end is
    refined instead: a value below `cut` is uniform on (0, cut) given that it is below `cut`, so
    it is multiplied by a fresh uniform, as many times as it takes. Each round touches a
    fraction `cut` of the previous one, so this costs almost nothing.
    """
    u = torch.rand(size, generator=generator, dtype=dtype, device=device)
    upper = u >= 0.5
    tail = torch.where(upper, 1.0 - u, u)
    cut = 2.0**-12
    log_cut = math.log(cut)
    log_tail = torch.log(torch.clamp(tail, min=cut))
    log_other = torch.log1p(-tail)

    deep = torch.nonzero(tail < cut, as_tuple=True)
    pending = deep
    while pending[0].numel():
        v = torch.rand(pending[0].numel(), generator=generator, dtype=dtype, device=device)
        again = v < cut
        log_tail[pending] += torch.where(again, torch.full_like(v, log_cut), torch.log(torch.clamp(v, min=cut)))
        pending = tuple(index[again] for index in pending)
    if deep[0].numel():
        log_other[deep] = torch.log1p(-torch.exp(log_tail[deep]))

    return torch.where(upper, log_other, log_tail), torch.where(upper, log_tail, log_other)


# =====================================================================
# PHASE 3: VECTORIZED SAMPLE STATISTICS HELPERS
# =====================================================================

def compute_sample_lmoments(data: torch.Tensor, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, ...]:
    """Sample (l1, l2, t3 = l3/l2, t4 = l4/l2) in O(N log N), computed in `dtype` (default: the data's).

    The unbiased weights are built as products of ratios, each in [0, 1], so nothing overflows, and
    applied to centered data, which cancels the mean because the weights of l2, l3 and l4 sum to 0.
    """
    if dtype is None:
        dtype = data.dtype if data.is_floating_point() else torch.float32
    x, _ = torch.sort(data.reshape(-1).to(dtype))
    n = x.numel()
    if n < 4:
        raise ValueError("At least 4 data points are required to compute 4 L-moments.")

    k = torch.arange(n, dtype=dtype, device=x.device)
    m = (n - 1) - k
    # k(k-1)/(n(n-1)), k m/(n(n-1)) and m(m-1)/(n(n-1))
    kk = (k / n) * ((k - 1.0) / (n - 1))
    km = (k / n) * (m / (n - 1))
    mm = (m / n) * ((m - 1.0) / (n - 1))

    l1 = torch.mean(x)
    c = x - l1
    l2 = torch.sum((k - m) / n / (n - 1) * c)
    l3 = torch.sum((kk - 4.0 * km + mm) / (n - 2) * c)
    w4 = kk * ((k - 2.0 - 9.0 * m) / (n - 2)) + km * (9.0 * (m - 1.0) / (n - 2)) - mm * ((m - 2.0) / (n - 2))
    l4 = torch.sum(w4 / (n - 3) * c)
    return l1, l2, l3 / l2, l4 / l2


def compute_sample_percentiles(data: torch.Tensor, u: float = 0.1, dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, ...]:
    """Sample percentile statistics, interpolated like numpy.quantile, in `dtype` (default: the data's).

    Written out because torch.quantile refuses inputs above 2**24 elements.
    """
    if not (0.0 < u < 0.25):
        raise ValueError("u must be in the open interval (0, 0.25).")
    if dtype is None:
        dtype = data.dtype if data.is_floating_point() else torch.float32
    x, _ = torch.sort(data.reshape(-1).to(dtype))
    last = x.numel() - 1

    def at(q: float) -> torch.Tensor:
        position = q * last  # a Python float: exact as an index in every precision
        below = int(position)
        above = min(below + 1, last)
        return x[below] + (x[above] - x[below]) * (position - below)

    qu, q25, q50, q75, q1mu = at(u), at(0.25), at(0.5), at(0.75), at(1.0 - u)
    a2 = q1mu - qu
    return q50, a2, (q50 - qu) / (q1mu - q50), (q75 - q25) / a2


def _require_fit_data(data: torch.Tensor) -> torch.Tensor:
    d = data.reshape(-1)
    if d.numel() < 4:
        raise ValueError("At least 4 data points are required.")
    if not torch.isfinite(d).all():
        raise ValueError("Data contains NaN or infinite values.")
    if d.min() == d.max():
        raise ValueError("All data points are equal: a GLD cannot be fitted.")
    return d


# =====================================================================
# PHASE 4: PARAMETER SOLVERS (in the dtype of their inputs)
# =====================================================================

_GOF_MAX_POINTS = 100_000
_LM_MAX_IDLE_STEPS = 4  # with no accepted step this long, the damping has grown by 4**4 and nothing moves


def _root_loss(dtype: torch.dtype) -> float:
    """Squared residual below which a shape solution counts as an exact root."""
    return max(1e-10, (2000.0 * torch.finfo(dtype).eps) ** 2)


def _quantile_sample(data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(sorted data, log y, log(1 - y)), thinned and moved to the CPU, in the data's precision.

    The shape solves are small batches of scalar formulas: on a GPU, launch overhead makes them
    slower than on the CPU. Only the sort of the full sample benefits from the device.
    """
    x, _ = torch.sort(data.reshape(-1))
    n = x.numel()
    m = min(n, _GOF_MAX_POINTS)
    idx = (torch.arange(m, device=x.device) * (n - 1)) // max(m - 1, 1)
    log_y, log_s = _plotting_logs(idx, n, x.dtype)
    return x[idx].cpu(), log_y.cpu(), log_s.cpu()


def _gof_per_candidate(candidates: torch.Tensor, sample: Tuple[torch.Tensor, ...], param_type: str) -> torch.Tensor:
    """Q-Q discrepancy of each candidate (K, 4) on the sample, invalid ones -> inf."""
    x, log_y, log_s = sample
    q = _quantile_from_logs(log_y.unsqueeze(0), log_s.unsqueeze(0), candidates.unsqueeze(1), param_type)
    mse = torch.mean((x.unsqueeze(0) - q) ** 2, dim=-1)
    valid = validate_gld_params(candidates, param_type) & torch.isfinite(mse)
    return torch.where(valid, mse, torch.full_like(mse, float("inf")))


def solve_vsl_lmm_closed_form(
    sample: Tuple[torch.Tensor, ...],
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    a4: torch.Tensor,
    initial_guess: Optional[float] = None,
) -> torch.Tensor:
    """Exact analytical L-moment estimator for the VSL parameterization."""
    disc = a4 * a4 + 98.0 * a4 + 1.0
    min_a4 = (-98.0 + (98.0**2 - 4.0) ** 0.5) / 2.0 + 1e-10
    a4_safe = torch.where(disc < 0.0, torch.full_like(a4, min_a4), a4)
    sqrt_disc = torch.sqrt(torch.clamp(a4_safe * a4_safe + 98.0 * a4_safe + 1.0, min=0.0))

    p4_pair = torch.stack(
        [
            (3.0 + 7.0 * a4_safe + sqrt_disc) / (2.0 * (1.0 - a4_safe)),
            (3.0 + 7.0 * a4_safe - sqrt_disc) / (2.0 * (1.0 - a4_safe)),
        ]
    )
    p3_pair = torch.where(
        p4_pair == 1.0,
        torch.full_like(p4_pair, 0.5),
        0.5 * (1.0 - a3 * (p4_pair + 3.0) / (p4_pair - 1.0)),
    )
    p3_pair = torch.clamp(p3_pair, 0.0, 1.0)
    p2_pair = a2 * (p4_pair + 1.0) * (p4_pair + 2.0)
    p1_pair = a1 + p2_pair * (1.0 - 2.0 * p3_pair) / (p4_pair + 1.0)

    candidates = torch.stack([p1_pair, p2_pair, p3_pair, p4_pair], dim=-1)  # (2, 4)
    if initial_guess is not None:
        return candidates[torch.argmin(torch.abs(p4_pair - float(initial_guess)))]
    return candidates[torch.argmin(_gof_per_candidate(candidates, sample, "VSL"))]


def _levenberg_marquardt(residual_fn, is_valid, l3: torch.Tensor, l4: torch.Tensor, steps: int) -> Tuple[torch.Tensor, ...]:
    """Batched Levenberg-Marquardt on a 2-equation, 2-unknown system.

    The Jacobian comes from autograd, because a difference step (eps**(1/3) ~ 5e-3 in float32) can
    be larger than the root itself: logistic data have their RS root at lambda ~ 1e-4. Each batch
    element depends only on its own unknowns, so the gradient of a sum gives per-element derivatives.
    """
    stop = (64.0 * torch.finfo(l3.dtype).eps) ** 2
    l3, l4 = l3.detach(), l4.detach()
    with torch.no_grad():
        r1, r2 = residual_fn(l3, l4)
    loss = r1 * r1 + r2 * r2
    loss = torch.where(torch.isfinite(loss), loss, torch.full_like(loss, float("inf")))
    mu = torch.full_like(l3, 1e-3)
    idle = 0
    for _ in range(steps):
        if loss.max() < stop or idle >= _LM_MAX_IDLE_STEPS:
            break
        x3, x4 = l3.clone().requires_grad_(True), l4.clone().requires_grad_(True)
        g1_r, g2_r = residual_fn(x3, x4)
        j11, j12 = torch.autograd.grad(g1_r.sum(), (x3, x4), retain_graph=True)
        j21, j22 = torch.autograd.grad(g2_r.sum(), (x3, x4))

        a11 = j11 * j11 + j21 * j21 + mu
        a12 = j11 * j12 + j21 * j22
        a22 = j12 * j12 + j22 * j22 + mu
        g1 = j11 * r1 + j21 * r2
        g2 = j12 * r1 + j22 * r2
        det = a11 * a22 - a12 * a12
        d3 = torch.clamp((a12 * g2 - a22 * g1) / det, -1.0, 1.0)
        d4 = torch.clamp((a12 * g1 - a11 * g2) / det, -1.0, 1.0)

        new3, new4 = l3 + d3, l4 + d4
        with torch.no_grad():
            nr1, nr2 = residual_fn(new3, new4)
        new_loss = nr1 * nr1 + nr2 * nr2
        accept = torch.isfinite(new_loss) & (new_loss < loss) & is_valid(new3, new4)
        idle = 0 if accept.any() else idle + 1
        l3, l4 = torch.where(accept, new3, l3), torch.where(accept, new4, l4)
        r1, r2 = torch.where(accept, nr1, r1), torch.where(accept, nr2, r2)
        loss = torch.where(accept, new_loss, loss)
        mu = torch.clamp(torch.where(accept, mu * 0.3, mu * 4.0), 1e-14, 1e8)
    return l3, l4, loss


def solve_shape_multistart(
    residual_fn,
    param_fn,
    sample: Tuple[torch.Tensor, ...],
    param_type: str,
    grid_min: float,
    grid_max: float,
    n_grid: int = 128,
    refine_steps: int = 40,
    max_starts: int = 16,
    initial_guess=None,
) -> torch.Tensor:
    """Solve the two shape equations of a fit and assemble the (4,) parameter vector.

    With `initial_guess`, this refines from there, as the CPU class does. Otherwise it refines
    every local minimum of a parallel grid. The equations often have several exact roots
    (e.g. a far root at lambda ~ 3.5 for data drawn with lambda ~ 0.2), so among the roots it
    keeps the one with the smallest Q-Q discrepancy on the data.
    """
    device, dtype = sample[0].device, sample[0].dtype

    def is_valid(l3, l4):
        p = param_fn(l3, l4)
        return validate_gld_params(p, param_type) & torch.isfinite(p).all(dim=-1)

    if initial_guess is not None:
        guess = torch.as_tensor(initial_guess, dtype=dtype, device=device).reshape(-1)
        if guess.numel() != 2:
            raise ValueError("Specify initial guess for two parameters.")
        l3, l4, loss = _levenberg_marquardt(residual_fn, is_valid, guess[:1], guess[1:], max(refine_steps, 100))
        if loss.item() <= _root_loss(dtype) and is_valid(l3, l4).all():
            return param_fn(l3, l4)[0]
        warnings.warn("Refining from initial_guess failed; falling back to a grid search.", UserWarning, stacklevel=3)

    g = torch.linspace(grid_min, grid_max, n_grid, device=device, dtype=dtype)
    l3_grid, l4_grid = torch.meshgrid(g, g, indexing="ij")
    r1, r2 = residual_fn(l3_grid, l4_grid)
    loss = r1 * r1 + r2 * r2
    usable = torch.isfinite(loss) & is_valid(l3_grid, l4_grid)
    loss = torch.where(usable, loss, torch.full_like(loss, float("inf")))
    if not usable.any():
        raise ValueError("No valid GLD parameters found on the search grid.")

    neighbor_min = -F.max_pool2d(-loss[None, None], 3, stride=1, padding=1)[0, 0]
    starts = torch.nonzero((loss == neighbor_min) & usable)
    starts = starts[torch.argsort(loss[starts[:, 0], starts[:, 1]])[:max_starts]]

    l3, l4, final = _levenberg_marquardt(residual_fn, is_valid, l3_grid[starts[:, 0], starts[:, 1]], l4_grid[starts[:, 0], starts[:, 1]], refine_steps)
    roots = final <= max(_root_loss(dtype), 4.0 * final.min().item())
    candidates = param_fn(l3[roots], l4[roots])
    return candidates[torch.argmin(_gof_per_candidate(candidates, sample, param_type))]


def _lmm_equations(param_type: str, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, a4: torch.Tensor):
    """Residuals and parameter assembly of the L-moment equations for RS and FMKL.

    Uses simplified closed forms of the upstream sums, for example
    (-1/(1+l) + 2/(2+l)) / l = 1/((1+l)(2+l)). FMKL's 1/l singularity cancels out that way.
    """
    rs = param_type == "RS"

    def g2(l):
        v = 1.0 / ((1.0 + l) * (2.0 + l))
        return l * v if rs else v

    def g3(l):
        v = (l - 1.0) / ((1.0 + l) * (2.0 + l) * (3.0 + l))
        return l * v if rs else v

    def g4(l):
        v = (l - 1.0) * (l - 2.0) / ((1.0 + l) * (2.0 + l) * (3.0 + l) * (4.0 + l))
        return l * v if rs else v

    def residual_fn(l3, l4):
        # L-moments exist only for lambda > -1; the formulas alone would not notice.
        exists = (l3 > -1.0) & (l4 > -1.0)
        scale = g2(l3) + g2(l4)
        r1 = (g3(l3) - g3(l4)) / scale - a3
        r2 = (g4(l3) + g4(l4)) / scale - a4
        nan = torch.full_like(r1, float("nan"))
        return torch.where(exists, r1, nan), torch.where(exists, r2, nan)

    def param_fn(l3, l4):
        l2 = (g2(l3) + g2(l4)) / a2
        shift = 1.0 / (1.0 + l4) - 1.0 / (1.0 + l3)
        l1 = a1 + (shift if rs else -shift) / l2
        return torch.stack([l1, l2, l3, l4], dim=-1)

    return residual_fn, param_fn


def _pm_equations(param_type: str, u: float, a1: torch.Tensor, a2: torch.Tensor, a3: torch.Tensor, a4: torch.Tensor):
    """Residuals and parameter assembly of the percentile equations for RS and FMKL."""
    term = _pow_minus_one if param_type == "RS" else _box_cox
    dtype, device = a1.dtype, a1.device
    log_of = lambda v: torch.log(torch.as_tensor(v, dtype=dtype, device=device))  # noqa: E731
    lu, lhalf, lq1, lq3, l1mu = log_of(u), log_of(0.5), log_of(0.25), log_of(0.75), log_of(1.0 - u)

    def spread(hi, lo, l3, l4):
        """[term(hi, l3) - term(lo, l3)] + [term(1 - lo, l4) - term(1 - hi, l4)], i.e. Q(hi) - Q(lo) up to scale."""
        return term(hi, l3) - term(lo, l3) + term(_mirror[lo], l4) - term(_mirror[hi], l4)

    _mirror = {lu: l1mu, l1mu: lu, lhalf: lhalf, lq1: lq3, lq3: lq1}

    def residual_fn(l3, l4):
        r1 = spread(lhalf, lu, l3, l4) / spread(l1mu, lhalf, l3, l4) - a3
        r2 = spread(lq3, lq1, l3, l4) / spread(l1mu, lu, l3, l4) - a4
        return r1, r2

    def param_fn(l3, l4):
        l2 = spread(l1mu, lu, l3, l4) / a2
        l1 = a1 - (term(lhalf, l3) - term(lhalf, l4)) / l2
        return torch.stack([l1, l2, l3, l4], dim=-1)

    return residual_fn, param_fn


def _scalar_guess(initial_guess) -> float:
    return float(np.asarray(initial_guess, dtype=float).reshape(-1)[0])


def _solve_vsl_pm(sample: Tuple[torch.Tensor, ...], u: float, stats: Tuple[torch.Tensor, ...], n_grid: int, initial_guess=None) -> torch.Tensor:
    """Percentile estimator for VSL: every root of the one-dimensional lambda equation, then closed forms."""
    a1, a2, a3, a4 = stats
    dtype, device = a1.dtype, a1.device
    log_of = lambda v: torch.log(torch.as_tensor(v, dtype=dtype, device=device))  # noqa: E731
    lu, lhalf, lq1, lq3, l1mu = log_of(u), log_of(0.5), log_of(0.25), log_of(0.75), log_of(1.0 - u)

    def excess(l):
        return (_box_cox(lq3, l) - _box_cox(lq1, l)) / (_box_cox(l1mu, l) - _box_cox(lu, l)) - a4

    g = torch.linspace(-2.0, 4.0, n_grid * 8, device=device, dtype=dtype)
    values = excess(g)
    crossing = torch.nonzero((values[:-1] * values[1:] <= 0.0) & torch.isfinite(values[:-1] * values[1:])).reshape(-1)
    if crossing.numel():
        low, high = g[crossing], g[crossing + 1]
        for _ in range(80):
            mid = 0.5 * (low + high)
            same_side = excess(mid) * excess(low) > 0.0
            low, high = torch.where(same_side, mid, low), torch.where(same_side, high, mid)
        lam = 0.5 * (low + high)
    else:
        lam = g[torch.argmin(torch.where(torch.isfinite(values), values.abs(), torch.full_like(values, float("inf"))))].reshape(1)

    b_up, b_half, b_lo = _box_cox(l1mu, lam), _box_cox(lhalf, lam), _box_cox(lu, lam)
    delta = (a3 * (b_up - b_half) - b_half + b_lo) / ((a3 + 1.0) * (b_up - 2.0 * b_half + b_lo))
    delta = torch.where(torch.isfinite(delta), torch.clamp(delta, 0.0, 1.0), torch.full_like(delta, 0.5))
    beta = a2 / (b_up - b_lo)
    alpha = a1 - beta * b_half * (1.0 - 2.0 * delta)
    candidates = torch.stack([alpha, beta, delta, lam], dim=-1)
    if initial_guess is not None:
        return candidates[torch.argmin((lam - _scalar_guess(initial_guess)).abs())]
    return candidates[torch.argmin(_gof_per_candidate(candidates, sample, "VSL"))]


# =====================================================================
# MODIFIED GLD CLASS (PYTORCH GPU IMPLEMENTATION)
# =====================================================================

class GLD:
    """PyTorch Generalized Lambda Distribution ('RS', 'FMKL', 'VSL') for CPU and CUDA.

    Every method computes in a single float dtype, chosen in this order: the method's `dtype`
    argument, the constructor's `dtype`, the widest float dtype of the tensor or NumPy inputs, and
    finally float32 for plain Python numbers and lists. `dtype` is torch.float32, torch.float64,
    or their names. Results have that dtype, and nothing is computed in another precision.
    """

    def __init__(
        self,
        param_type: str = "FMKL",
        device: Optional[torch.device] = None,
        dtype: Union[None, str, torch.dtype] = None,
    ):
        if param_type not in ("RS", "FMKL", "VSL"):
            raise ValueError(f"Unknown parameterization '{param_type}'. Use 'RS', 'FMKL', or 'VSL'.")
        self.param_type = param_type
        if device is None:
            device = _default_device()
        self.device = torch.device(device)
        self.dtype = _as_dtype(dtype)

    def _resolve_dtype(self, xs, dtype=None) -> torch.dtype:
        chosen = _as_dtype(dtype)
        if chosen is None:
            chosen = self.dtype
        if chosen is not None:
            return chosen
        found = None
        for x in xs:
            if isinstance(x, torch.Tensor) and x.is_floating_point():
                dt = x.dtype
            elif getattr(getattr(x, "dtype", None), "kind", "") == "f":
                dt = torch.float32 if x.dtype.itemsize <= 4 else torch.float64  # NumPy array or scalar
            else:
                continue
            found = dt if found is None else torch.promote_types(found, dt)
        return found if found is not None else torch.float32

    def _tensors(self, *xs, dtype=None) -> Tuple[torch.Tensor, ...]:
        """Convert inputs to tensors of one common dtype on this device."""
        chosen = self._resolve_dtype(xs, dtype)
        return tuple(torch.as_tensor(x, dtype=chosen, device=self.device) for x in xs)

    def _checked(self, p: torch.Tensor) -> torch.Tensor:
        if not bool(validate_gld_params(p, self.param_type).all().item()):
            raise ValueError("Parameters are not valid.")
        return p

    def check_param(self, param, dtype=None) -> bool:
        p = self._tensors(param, dtype=dtype)[0]
        return bool(validate_gld_params(p, self.param_type).all().item())

    def _probability_and_param(self, y, param, name: str = "y", dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        y_t, p = self._tensors(y, param, dtype=dtype)
        if torch.any((y_t < 0.0) | (y_t > 1.0)):
            raise ValueError(f"{name} must be in range [0, 1].")
        return y_t, self._checked(p)

    def Q(self, y, param, dtype=None) -> torch.Tensor:
        """Quantile function at lower-tail probability `y`."""
        y_t, p = self._probability_and_param(y, param, dtype=dtype)
        return compute_quantile(y_t, p, self.param_type)

    def ISF(self, s, param, dtype=None) -> torch.Tensor:
        """Q(1 - s): the quantile at upper-tail probability `s`, exact where `1 - s` would round to 1."""
        s_t, p = self._probability_and_param(s, param, name="s", dtype=dtype)
        return _quantile_from_logs(torch.log1p(-s_t), torch.log(s_t), p, self.param_type)

    def PDF_Q(self, y, param, dtype=None) -> torch.Tensor:
        y_t, p = self._probability_and_param(y, param, dtype=dtype)
        return compute_pdf_quantile(y_t, p, self.param_type)

    def supp(self, param, dtype=None) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self._checked(self._tensors(param, dtype=dtype)[0])
        zero = torch.zeros(1, dtype=p.dtype, device=p.device)
        inf = torch.full_like(zero, float("inf"))
        lo = _quantile_from_logs(-inf, zero, p, self.param_type)
        hi = _quantile_from_logs(zero, -inf, p, self.param_type)
        return (lo.squeeze(0), hi.squeeze(0)) if p.dim() == 1 else (lo, hi)

    def _logit(self, x, param, xtol: float, max_iter: int, dtype=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_t, p = self._tensors(x, param, dtype=dtype)
        self._checked(p)
        return solve_logit_cdf(x_t, p, self.param_type, xtol=xtol, max_iter=max_iter), x_t, p

    def CDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        """P(X <= x). `xtol` is the relative accuracy of the smaller tail probability."""
        return torch.sigmoid(self._logit(x, param, xtol, max_iter, dtype)[0])

    def SF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        """P(X > x), computed directly. Use it for upper tails: `1 - CDF_num` loses everything below eps."""
        return torch.sigmoid(-self._logit(x, param, xtol, max_iter, dtype)[0])

    def log_PDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        """log density at `x`, -inf outside the support. Valid where PDF_num itself underflows."""
        t, x_t, p = self._logit(x, param, xtol, max_iter, dtype)
        out = _log_density_from_logs(F.logsigmoid(t), F.logsigmoid(-t), p, self.param_type)
        lo, hi = self.supp(p)
        outside = (x_t < lo) | (x_t > hi)
        return torch.where(outside, torch.full_like(out, -float("inf")), out)

    def PDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 64, dtype=None) -> torch.Tensor:
        return torch.exp(self.log_PDF_num(x, param, xtol=xtol, max_iter=max_iter, dtype=dtype))

    def rand(
        self,
        param,
        size: Tuple[int, ...] = (1,),
        generator: Optional[torch.Generator] = None,
        dtype=None,
    ) -> torch.Tensor:
        """Inverse-transform sampling, with both tails resolved in any precision (`_sample_log_uniform`)."""
        p = self._checked(self._tensors(param, dtype=dtype)[0])
        if isinstance(size, int):
            size = (size,)
        log_y, log_s = _sample_log_uniform(tuple(size), p.dtype, p.device, generator)
        return _quantile_from_logs(log_y, log_s, p, self.param_type)

    def GoF_Q_metric(self, data, param, dtype=None) -> torch.Tensor:
        d, p = self._tensors(data, param, dtype=dtype)
        self._checked(p)
        x_sorted, _ = torch.sort(d.reshape(-1))
        n = x_sorted.numel()
        log_y, log_s = _plotting_logs(torch.arange(n, device=p.device), n, p.dtype)
        return torch.mean((x_sorted - _quantile_from_logs(log_y, log_s, p, self.param_type)) ** 2)

    def fit_LMM(self, data, initial_guess=None, n_grid: int = 128, refine_steps: int = 40, dtype=None) -> torch.Tensor:
        """L-moment estimation: closed form for VSL, multistart solve for RS/FMKL.

        Without `initial_guess`, the root closest to the data wins (see `solve_shape_multistart`).
        """
        d = _require_fit_data(self._tensors(data, dtype=dtype)[0])
        stats = tuple(v.cpu() for v in compute_sample_lmoments(d))
        sample = _quantile_sample(d)
        if self.param_type == "VSL":
            guess = None if initial_guess is None else _scalar_guess(initial_guess)
            return solve_vsl_lmm_closed_form(sample, *stats, initial_guess=guess).to(d.device)
        residual_fn, param_fn = _lmm_equations(self.param_type, *stats)
        return solve_shape_multistart(
            residual_fn, param_fn, sample, self.param_type, grid_min=-0.95, grid_max=4.0,
            n_grid=n_grid, refine_steps=refine_steps, initial_guess=initial_guess,
        ).to(d.device)

    def fit_PM(self, data, initial_guess=None, u: float = 0.1, n_grid: int = 128, refine_steps: int = 40, dtype=None) -> torch.Tensor:
        """Percentile-method estimation. Without `initial_guess`, the root closest to the data wins."""
        d = _require_fit_data(self._tensors(data, dtype=dtype)[0])
        stats = tuple(v.cpu() for v in compute_sample_percentiles(d, u))
        sample = _quantile_sample(d)
        if self.param_type == "VSL":
            return _solve_vsl_pm(sample, u, stats, n_grid, initial_guess=initial_guess).to(d.device)
        residual_fn, param_fn = _pm_equations(self.param_type, u, *stats)
        return solve_shape_multistart(
            residual_fn, param_fn, sample, self.param_type, grid_min=-2.0, grid_max=4.0,
            n_grid=n_grid, refine_steps=refine_steps, initial_guess=initial_guess,
        ).to(d.device)

    def correct_supp(self, data, param, eps: float = 1e-4, dtype=None) -> torch.Tensor:
        """Shift and rescale `param` so Q(eps) and Q(1 - eps) cover the data's range.

        Q is affine in location and scale, so the two end conditions have an exact solution.
        Returns new parameters and leaves `param` untouched.
        """
        d, p = self._tensors(data, param, dtype=dtype)
        self._checked(p)
        d = d.reshape(-1)
        log_low = torch.as_tensor([math.log(eps), math.log1p(-eps)], dtype=p.dtype, device=p.device)
        log_high = log_low.flip(0)
        q_lo, q_hi = _quantile_from_logs(log_low, log_high, p, self.param_type).unbind(0)
        lo = torch.minimum(d.min(), q_lo)
        hi = torch.maximum(d.max(), q_hi)
        k = (hi - lo) / (q_hi - q_lo)
        new_scale = p[1] * k if self.param_type == "VSL" else p[1] / k
        return torch.stack([lo - (q_lo - p[0]) * k, new_scale, p[2], p[3]])

    def fit_ML(self, data, init_method: str = "LMM", steps: int = 300, lr: float = 0.02, xtol: float = 1e-6, initial_guess=None, dtype=None) -> torch.Tensor:
        """Maximum likelihood: Adam with implicit differentiation through the CDF solve.

        Runs on standardized data, so `lr` means the same in any unit and a large offset does not
        eat float32 precision. A point outside the support has zero likelihood, so after each step
        location and scale move just enough to cover the data (`_with_support_covering`), which
        leaves the shape parameters free to keep moving. A step that makes the parameters invalid
        is undone and `lr` halved.
        """
        if init_method not in ("LMM", "PM"):
            raise ValueError("init_method must be 'LMM' or 'PM'.")
        d = _require_fit_data(self._tensors(data, dtype=dtype)[0])
        ptype = self.param_type
        init = self.fit_LMM(d, initial_guess) if init_method == "LMM" else self.fit_PM(d, initial_guess)

        center, spread = d.median(), d.std()
        z = (d - center) / spread

        def to_standard(p):
            scale = p[1] / spread if ptype == "VSL" else p[1] * spread
            return torch.stack([(p[0] - center) / spread, scale, p[2], p[3]])

        def from_standard(p):
            scale = p[1] * spread if ptype == "VSL" else p[1] / spread
            return torch.stack([p[0] * spread + center, scale, p[2], p[3]])

        z_min, z_max = z.min(), z.max()
        start = _with_support_covering(to_standard(init), z_min, z_max, ptype)

        param = start.clone().requires_grad_(True)
        best_param, best_nll = start.clone(), float("inf")
        optimizer = torch.optim.Adam([param], lr=lr)
        t_prev = None

        for _ in range(steps):
            optimizer.zero_grad()
            nll = None
            if validate_gld_params(param.detach(), ptype):
                with torch.no_grad():
                    t = solve_logit_cdf(z, param.detach(), ptype, xtol=xtol, t0=t_prev)
                if torch.isfinite(t).all():
                    t_prev = t
                    log_y, log_s = F.logsigmoid(t), F.logsigmoid(-t)
                    # Implicit function theorem: dt/dtheta = -(dQ/dtheta) * dt/dx
                    dt_dx = torch.exp(_log_density_from_logs(log_y, log_s, param.detach(), ptype, interior=True) - log_y - log_s)
                    t_diff = t - (_quantile_from_logs(log_y, log_s, param, ptype) - z) * dt_dx
                    nll = -_log_density_from_logs(F.logsigmoid(t_diff), F.logsigmoid(-t_diff), param, ptype, interior=True).mean()

            if nll is None or not torch.isfinite(nll):
                lr *= 0.5
                if lr < 1e-5:
                    break
                param = best_param.clone().requires_grad_(True)
                optimizer = torch.optim.Adam([param], lr=lr)
                continue

            if nll.item() < best_nll:
                best_nll, best_param = nll.item(), param.detach().clone()
            nll.backward()
            optimizer.step()
            with torch.no_grad():
                if ptype == "VSL":
                    param[2].clamp_(0.0, 1.0)
                if validate_gld_params(param, ptype):
                    param.copy_(_with_support_covering(param, z_min, z_max, ptype))

        return from_standard(best_param)
