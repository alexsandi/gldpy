import warnings

import torch
from typing import Optional, Tuple


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

        # Safe clamping to avoid complex powers outside region 5 and 6
        p3_r5 = torch.clamp(p3, min=-1.0, max=0.0)
        p4_r5 = torch.clamp(p4, min=1.0 + 1e-12)
        lhs_r5 = (
            torch.pow(1.0 - p3_r5, 1.0 - p3_r5)
            * torch.pow(p4_r5 - 1.0, p4_r5 - 1.0)
            / torch.pow(p4_r5 - p3_r5, p4_r5 - p3_r5)
        )
        r5 = (p2 < 0) & (p3 <= 0) & (p3 >= -1) & (p4 >= 1) & (lhs_r5 <= -p3_r5 / p4_r5)

        p3_r6 = torch.clamp(p3, min=1.0 + 1e-12)
        p4_r6 = torch.clamp(p4, min=-1.0, max=0.0)
        lhs_r6 = (
            torch.pow(1.0 - p4_r6, 1.0 - p4_r6)
            * torch.pow(p3_r6 - 1.0, p3_r6 - 1.0)
            / torch.pow(p3_r6 - p4_r6, p3_r6 - p4_r6)
        )
        r6 = (p2 < 0) & (p3 >= 1) & (p4 >= -1) & (p4 <= 0) & (lhs_r6 <= -p4_r6 / p3_r6)

        return finite_mask & (r1 | r2 | r3 | r4 | r5 | r6)
    raise ValueError(f"Unknown parameterization '{param_type}'.")


# =====================================================================
# PHASE 2: CORE QUANTILE, DENSITY & ROOT-FINDING TRANSFORMATIONS
# =====================================================================

# [STABLE HELPER]
def _box_cox_power(u: torch.Tensor, lmbda: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Branchless, autograd-safe evaluation of (u^lambda - 1) / lambda."""
    is_nonzero = torch.abs(lmbda) > eps
    safe_lmbda = torch.where(is_nonzero, lmbda, torch.ones_like(lmbda))
    return torch.where(is_nonzero, (torch.pow(u, safe_lmbda) - 1.0) / safe_lmbda, torch.log(u))


# [STABLE HELPER]
def compute_quantile(y: torch.Tensor, param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Evaluate quantile function Q(y) with full broadcasting over y and param."""
    p1, p2, p3, p4 = param.unbind(dim=-1)
    if param_type == "RS":
        return p1 + (torch.pow(y, p3) - torch.pow(1.0 - y, p4)) / p2
    if param_type == "FMKL":
        f1 = _box_cox_power(y, p3)
        f2 = _box_cox_power(1.0 - y, p4)
        return p1 + (f1 - f2) / p2
    if param_type == "VSL":
        f1 = _box_cox_power(y, p4)
        f2 = _box_cox_power(1.0 - y, p4)
        return p1 + p2 * ((1.0 - p3) * f1 - p3 * f2)
    raise ValueError(f"Unknown parameterization '{param_type}'.")


# [STABLE HELPER]
def compute_pdf_quantile(y: torch.Tensor, param: torch.Tensor, param_type: str) -> torch.Tensor:
    """Evaluate density quantile function f_Q(y) = 1 / Q'(y)."""
    p1, p2, p3, p4 = param.unbind(dim=-1)
    if param_type == "RS":
        denom = p3 * torch.pow(y, p3 - 1.0) + p4 * torch.pow(1.0 - y, p4 - 1.0)
        return p2 / denom
    if param_type == "FMKL":
        denom = torch.pow(y, p3 - 1.0) + torch.pow(1.0 - y, p4 - 1.0)
        return p2 / denom
    if param_type == "VSL":
        denom = (1.0 - p3) * torch.pow(y, p4 - 1.0) + p3 * torch.pow(1.0 - y, p4 - 1.0)
        return 1.0 / (p2 * denom)
    raise ValueError(f"Unknown parameterization '{param_type}'.")


# [STABLE HELPER]
def invert_quantile_newton_bisect(
    x: torch.Tensor,
    param: torch.Tensor,
    param_type: str,
    xtol: float = 1e-6,
    max_iter: int = 22,
) -> torch.Tensor:
    """Vectorized hybrid Newton-Raphson + Bisection solver for Q(y) = x on GPU."""
    eps = torch.finfo(x.dtype).eps
    low = torch.full_like(x, eps)
    high = torch.full_like(x, 1.0 - eps)
    y = torch.full_like(x, 0.5)

    for _ in range(max_iter):
        q_val = compute_quantile(y, param, param_type)
        err = q_val - x
        if torch.max(torch.abs(err)) <= xtol:
            break

        low = torch.where(err < 0.0, y, low)
        high = torch.where(err >= 0.0, y, high)

        # Since Q'(y) = 1 / f_Q(y), Newton step is err * f_Q(y)
        f_q = compute_pdf_quantile(y, param, param_type)
        y_newton = y - err * f_q
        y_bisect = 0.5 * (low + high)

        in_bounds = (y_newton > low) & (y_newton < high) & torch.isfinite(y_newton)
        y = torch.where(in_bounds, y_newton, y_bisect)

    # Clamp out-of-support values to 0.0 and 1.0
    z0 = torch.zeros(1, dtype=x.dtype, device=x.device)
    z1 = torch.ones(1, dtype=x.dtype, device=x.device)
    q_min = compute_quantile(z0, param, param_type)
    q_max = compute_quantile(z1, param, param_type)
    y = torch.where(x <= q_min, torch.zeros_like(y), y)
    y = torch.where(x >= q_max, torch.ones_like(y), y)
    return y


# =====================================================================
# PHASE 3: VECTORIZED SAMPLE STATISTICS HELPERS
# =====================================================================

# [STABLE HELPER]
def compute_sample_lmoments(data: torch.Tensor) -> Tuple[torch.Tensor, ...]:
    """Compute sample L-moments (l1, l2, t3=l3/l2, t4=l4/l2) in O(N log N) without loops."""
    x, _ = torch.sort(data.reshape(-1))
    n = x.numel()
    if n < 4:
        raise ValueError("At least 4 data points are required to compute 4 L-moments.")

    # Use float64 for combinatorial index weights to prevent overflow at large N
    x64 = x.to(torch.float64)
    k = torch.arange(n, dtype=torch.float64, device=x.device)
    m = (n - 1) - k

    l1 = torch.mean(x64)

    w2 = k - m
    l2 = torch.sum(w2 * x64) / (n * (n - 1))

    c_k2 = k * (k - 1.0) * 0.5
    c_m2 = m * (m - 1.0) * 0.5
    w3 = c_k2 - 2.0 * k * m + c_m2
    l3 = torch.sum(w3 * x64) / (n * (n - 1) * (n - 2) / 2.0)

    c_k3 = k * (k - 1.0) * (k - 2.0) / 6.0
    c_m3 = m * (m - 1.0) * (m - 2.0) / 6.0
    w4 = c_k3 - 3.0 * c_k2 * m + 3.0 * k * c_m2 - c_m3
    l4 = torch.sum(w4 * x64) / (n * (n - 1) * (n - 2) * (n - 3) / 6.0)

    out_dtype = data.dtype if data.is_floating_point() else torch.float32
    return (
        l1.to(out_dtype),
        l2.to(out_dtype),
        (l3 / l2).to(out_dtype),
        (l4 / l2).to(out_dtype),
    )


# [STABLE HELPER]
def compute_sample_percentiles(data: torch.Tensor, u: float = 0.1) -> Tuple[torch.Tensor, ...]:
    """Compute the 4 percentile statistics on GPU in a single torch.quantile call."""
    if not (0.0 < u < 0.25):
        raise ValueError("u must be in the open interval (0, 0.25).")
    x = data.reshape(-1)
    q_probs = torch.tensor([u, 0.25, 0.5, 0.75, 1.0 - u], dtype=x.dtype, device=x.device)
    qu, q25, q50, q75, q1mu = torch.quantile(x, q_probs).unbind(0)

    a1 = q50
    a2 = q1mu - qu
    a3 = (q50 - qu) / (q1mu - q50)
    a4 = (q75 - q25) / a2
    return a1, a2, a3, a4


# =====================================================================
# PHASE 4: GPU-ACCELERATED PARAMETER SOLVERS
# =====================================================================

# [STABLE HELPER]
def solve_vsl_lmm_closed_form(
    data: torch.Tensor,
    a1: torch.Tensor,
    a2: torch.Tensor,
    a3: torch.Tensor,
    a4: torch.Tensor,
    initial_guess: Optional[float] = None,
) -> torch.Tensor:
    """Exact analytical L-moment estimator for VSL parameterization on GPU."""
    disc = a4 * a4 + 98.0 * a4 + 1.0
    min_a4 = (-98.0 + (98.0**2 - 4.0) ** 0.5) / 2.0 + 1e-7
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
        idx = torch.argmin(torch.abs(p4_pair - float(initial_guess)))
        return candidates[idx]

    # Select candidate with minimal empirical Q-Q discrepancy in parallel
    x_sorted, _ = torch.sort(data.reshape(-1))
    n = x_sorted.numel()
    y_emp = (torch.arange(n, dtype=x_sorted.dtype, device=x_sorted.device) + 0.5) / n
    q_cand = compute_quantile(y_emp.unsqueeze(0), candidates.unsqueeze(1), "VSL")
    mse = torch.mean((x_sorted.unsqueeze(0) - q_cand) ** 2, dim=-1)
    valid = validate_gld_params(candidates, "VSL")
    mse = torch.where(valid, mse, torch.full_like(mse, float("inf")))
    return candidates[torch.argmin(mse)]


# [STABLE HELPER]
def solve_shape_grid_refine(
    loss_fn,
    grid_min: float = -0.95,
    grid_max: float = 4.0,
    n_grid: int = 128,
    refine_steps: int = 40,
    lr: float = 0.05,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Parallel 2D GPU grid search followed by Adam refinement for (l3, l4)."""
    g = torch.linspace(grid_min, grid_max, n_grid, device=device, dtype=dtype)
    l3_grid, l4_grid = torch.meshgrid(g, g, indexing="ij")
    with torch.no_grad():
        losses = loss_fn(l3_grid, l4_grid)
        losses = torch.where(torch.isfinite(losses), losses, torch.full_like(losses, float("inf")))
        flat_idx = torch.argmin(losses)
        best_l3 = l3_grid.reshape(-1)[flat_idx].clone()
        best_l4 = l4_grid.reshape(-1)[flat_idx].clone()

    l3_opt = best_l3.detach().requires_grad_(True)
    l4_opt = best_l4.detach().requires_grad_(True)
    optimizer = torch.optim.Adam([l3_opt, l4_opt], lr=lr)

    for _ in range(refine_steps):
        optimizer.zero_grad()
        loss = loss_fn(l3_opt, l4_opt)
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()

    return l3_opt.detach(), l4_opt.detach()


# =====================================================================
# MODIFIED GLD CLASS (PYTORCH GPU IMPLEMENTATION)
# =====================================================================

class GLD:
    """PyTorch GPU-accelerated Generalized Lambda Distribution ('RS', 'FMKL', 'VSL')."""

    def __init__(self, param_type: str = "FMKL", device: Optional[torch.device] = None):
        if param_type not in ("RS", "FMKL", "VSL"):
            raise ValueError(f"Unknown parameterization '{param_type}'. Use 'RS', 'FMKL', or 'VSL'.")
        self.param_type = param_type
        if device is None:
            device = _default_device()
        self.device = torch.device(device)

    def _as_tensor(self, x, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x
            if self.device is not None and t.device != self.device:
                t = t.to(self.device)
            return t.to(dtype) if dtype is not None else t
        target_dtype = dtype or torch.float32
        return torch.as_tensor(x, dtype=target_dtype, device=self.device)

    def check_param(self, param) -> bool:
        p = self._as_tensor(param)
        return bool(validate_gld_params(p, self.param_type).all().item())

    def Q(self, y, param) -> torch.Tensor:
        p = self._as_tensor(param)
        y_t = self._as_tensor(y, dtype=p.dtype)
        if torch.any((y_t < 0.0) | (y_t > 1.0)):
            raise ValueError("y must be in range [0, 1].")
        if not self.check_param(p):
            raise ValueError("Parameters are not valid.")
        return compute_quantile(y_t, p, self.param_type)

    def PDF_Q(self, y, param) -> torch.Tensor:
        p = self._as_tensor(param)
        y_t = self._as_tensor(y, dtype=p.dtype)
        if torch.any((y_t < 0.0) | (y_t > 1.0)):
            raise ValueError("y must be in range [0, 1].")
        if not self.check_param(p):
            raise ValueError("Parameters are not valid.")
        return compute_pdf_quantile(y_t, p, self.param_type)

    def supp(self, param) -> Tuple[torch.Tensor, torch.Tensor]:
        p = self._as_tensor(param)
        if not self.check_param(p):
            raise ValueError("Parameters are not valid.")
        bounds = torch.tensor([0.0, 1.0], dtype=p.dtype, device=p.device)
        q_bounds = compute_quantile(bounds, p, self.param_type)
        return q_bounds[0], q_bounds[1]

    def CDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 22) -> torch.Tensor:
        p = self._as_tensor(param)
        x_t = self._as_tensor(x, dtype=p.dtype)
        if not self.check_param(p):
            raise ValueError("Parameters are not valid.")
        return invert_quantile_newton_bisect(x_t, p, self.param_type, xtol=xtol, max_iter=max_iter)

    def PDF_num(self, x, param, xtol: float = 1e-6, max_iter: int = 22) -> torch.Tensor:
        p = self._as_tensor(param)
        x_t = self._as_tensor(x, dtype=p.dtype)
        y = self.CDF_num(x_t, p, xtol=xtol, max_iter=max_iter)
        in_supp = (y > 0.0) & (y < 1.0)
        y_safe = torch.clamp(y, 1e-7, 1.0 - 1e-7)
        pdf = compute_pdf_quantile(y_safe, p, self.param_type)
        return torch.where(in_supp, pdf, torch.zeros_like(pdf))

    def rand(
        self,
        param,
        size: Tuple[int, ...] = (1,),
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        p = self._as_tensor(param)
        if isinstance(size, int):
            size = (size,)
        u = torch.rand(size, generator=generator, dtype=p.dtype, device=p.device)
        return self.Q(u, p)

    def GoF_Q_metric(self, data, param) -> torch.Tensor:
        p = self._as_tensor(param)
        d = self._as_tensor(data, dtype=p.dtype).reshape(-1)
        x_sorted, _ = torch.sort(d)
        n = x_sorted.numel()
        y_emp = (torch.arange(n, dtype=p.dtype, device=p.device) + 0.5) / n
        return torch.mean((x_sorted - self.Q(y_emp, p)) ** 2)

    def fit_LMM(
        self,
        data: torch.Tensor,
        initial_guess: Optional[float] = None,
        n_grid: int = 128,
        refine_steps: int = 40,
    ) -> torch.Tensor:
        """GPU-accelerated L-Moment estimation (exact closed-form for VSL; parallel grid+Adam for RS/FMKL)."""
        d = self._as_tensor(data)
        a1, a2, a3, a4 = compute_sample_lmoments(d)

        if self.param_type == "VSL":
            return solve_vsl_lmm_closed_form(d, a1, a2, a3, a4, initial_guess=initial_guess)

        if self.param_type == "RS":
            def loss_fn(l3, l4):
                L2 = -1.0 / (1.0 + l3) + 2.0 / (2.0 + l3) - 1.0 / (1.0 + l4) + 2.0 / (2.0 + l4)
                t3 = (
                    1.0 / (l3 + 1.0) - 6.0 / (2.0 + l3) + 6.0 / (3.0 + l3)
                    - 1.0 / (l4 + 1.0) + 6.0 / (2.0 + l4) - 6.0 / (3.0 + l4)
                ) / L2
                t4 = (
                    -1.0 / (1.0 + l3) + 12.0 / (2.0 + l3) - 30.0 / (3.0 + l3) + 20.0 / (4.0 + l3)
                    - 1.0 / (1.0 + l4) + 12.0 / (2.0 + l4) - 30.0 / (3.0 + l4) + 20.0 / (4.0 + l4)
                ) / L2
                return (t3 - a3) ** 2 + (t4 - a4) ** 2

            l3, l4 = solve_shape_grid_refine(
                loss_fn, n_grid=n_grid, refine_steps=refine_steps, device=d.device, dtype=a1.dtype
            )
            l2 = (-1.0 / (1.0 + l3) + 2.0 / (2.0 + l3) - 1.0 / (1.0 + l4) + 2.0 / (2.0 + l4)) / a2
            l1 = a1 + (1.0 / (1.0 + l4) - 1.0 / (1.0 + l3)) / l2
            return torch.stack([l1, l2, l3, l4])

        if self.param_type == "FMKL":
            def loss_fn(l3, l4):
                L2 = (
                    -1.0 / ((1.0 + l3) * l3) + 2.0 / ((2.0 + l3) * l3)
                    - 1.0 / ((1.0 + l4) * l4) + 2.0 / ((2.0 + l4) * l4)
                )
                t3 = (
                    (1.0 / (l3 + 1.0) - 6.0 / (2.0 + l3) + 6.0 / (3.0 + l3)) / l3
                    - (1.0 / (l4 + 1.0) - 6.0 / (2.0 + l4) + 6.0 / (3.0 + l4)) / l4
                ) / L2
                t4 = (
                    (-1.0 / (1.0 + l3) + 12.0 / (2.0 + l3) - 30.0 / (3.0 + l3) + 20.0 / (4.0 + l3)) / l3
                    + (-1.0 / (1.0 + l4) + 12.0 / (2.0 + l4) - 30.0 / (3.0 + l4) + 20.0 / (4.0 + l4)) / l4
                ) / L2
                return (t3 - a3) ** 2 + (t4 - a4) ** 2

            l3, l4 = solve_shape_grid_refine(
                loss_fn, n_grid=n_grid, refine_steps=refine_steps, device=d.device, dtype=a1.dtype
            )
            l2 = (
                -1.0 / ((1.0 + l3) * l3) + 2.0 / ((2.0 + l3) * l3)
                - 1.0 / ((1.0 + l4) * l4) + 2.0 / ((2.0 + l4) * l4)
            ) / a2
            l1 = a1 + (1.0 / ((1.0 + l4) * l4) - 1.0 / ((1.0 + l3) * l3) + 1.0 / l3 - 1.0 / l4) / l2
            return torch.stack([l1, l2, l3, l4])

    def fit_PM(
        self,
        data: torch.Tensor,
        u: float = 0.1,
        n_grid: int = 128,
        refine_steps: int = 40,
    ) -> torch.Tensor:
        """GPU-accelerated Percentile Method using torch.quantile + parallel grid/Adam."""
        d = self._as_tensor(data)
        a1, a2, a3, a4 = compute_sample_percentiles(d, u=u)
        u_t = torch.tensor(u, dtype=a1.dtype, device=d.device)
        half = torch.tensor(0.5, dtype=a1.dtype, device=d.device)
        q25 = torch.tensor(0.25, dtype=a1.dtype, device=d.device)
        q75 = torch.tensor(0.75, dtype=a1.dtype, device=d.device)

        if self.param_type == "FMKL":
            def loss_fn(l3, l4):
                num3 = _box_cox_power(half, l3) - _box_cox_power(u_t, l3) + _box_cox_power(1.0 - u_t, l4) - _box_cox_power(half, l4)
                den3 = _box_cox_power(1.0 - u_t, l3) - _box_cox_power(half, l3) + _box_cox_power(half, l4) - _box_cox_power(u_t, l4)
                num4 = _box_cox_power(q75, l3) - _box_cox_power(q25, l3) + _box_cox_power(q75, l4) - _box_cox_power(q25, l4)
                den4 = _box_cox_power(1.0 - u_t, l3) - _box_cox_power(u_t, l3) + _box_cox_power(1.0 - u_t, l4) - _box_cox_power(u_t, l4)
                return (num3 / den3 - a3) ** 2 + (num4 / den4 - a4) ** 2

            l3, l4 = solve_shape_grid_refine(
                loss_fn, grid_min=-2.0, grid_max=4.0, n_grid=n_grid, refine_steps=refine_steps, device=d.device, dtype=a1.dtype
            )
            l2 = (_box_cox_power(1.0 - u_t, l3) - _box_cox_power(u_t, l3) + _box_cox_power(1.0 - u_t, l4) - _box_cox_power(u_t, l4)) / a2
            l1 = a1 - (_box_cox_power(half, l3) - _box_cox_power(half, l4)) / l2
            return torch.stack([l1, l2, l3, l4])

        if self.param_type == "RS":
            def loss_fn(l3, l4):
                den_p2 = (1.0 - u_t) ** l3 - u_t ** l4 - u_t ** l3 + (1.0 - u_t) ** l4
                p3_hat = (half ** l3 - half ** l4 - u_t ** l3 + (1.0 - u_t) ** l4) / (
                    (1.0 - u_t) ** l3 - u_t ** l4 - half ** l3 + half ** l4
                )
                p4_hat = (q75 ** l3 - q25 ** l4 - q25 ** l3 + q75 ** l4) / den_p2
                return (p3_hat - a3) ** 2 + (p4_hat - a4) ** 2

            l3, l4 = solve_shape_grid_refine(
                loss_fn, grid_min=-0.95, grid_max=4.0, n_grid=n_grid, refine_steps=refine_steps, device=d.device, dtype=a1.dtype
            )
            l2 = ((1.0 - u_t) ** l3 - u_t ** l3 + (1.0 - u_t) ** l4 - u_t ** l4) / a2
            l1 = a1 - (half ** l3 - half ** l4) / l2
            return torch.stack([l1, l2, l3, l4])

        if self.param_type == "VSL":
            g = torch.linspace(-2.0, 4.0, n_grid * 8, device=d.device, dtype=a1.dtype)
            err = torch.abs((q75 ** g - q25 ** g) / ((1.0 - u_t) ** g - u_t ** g) - a4)
            # g contains 0.0 where the ratio is 0/0 = NaN, and argmin would pick the NaN.
            err = torch.where(torch.isfinite(err), err, torch.full_like(err, float("inf")))
            l = g[torch.argmin(err)]
            delta = (a3 * ((1.0 - u_t) ** l - half ** l) - half ** l + u_t ** l) / (
                (a3 + 1.0) * ((1.0 - u_t) ** l - 2.0 * (half ** l) + u_t ** l)
            )
            delta = torch.clamp(delta, 0.0, 1.0)
            beta = a2 * l / ((1.0 - u_t) ** l - u_t ** l)
            alpha = a1 - beta * (half ** l - 1.0) * (1.0 - 2.0 * delta) / l
            return torch.stack([alpha, beta, delta, l])

    def fit_ML(
        self,
        data: torch.Tensor,
        init_method: str = "LMM",
        steps: int = 60,
        lr: float = 0.02,
        xtol: float = 1e-6,
    ) -> torch.Tensor:
        """GPU Maximum Likelihood estimation using vectorized Newton-Bisection CDF + autograd."""
        d = self._as_tensor(data).reshape(-1)
        if init_method == "LMM":
            param_init = self.fit_LMM(d)
        elif init_method == "PM":
            param_init = self.fit_PM(d)
        else:
            raise ValueError("init_method must be 'LMM' or 'PM'.")

        param = param_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([param], lr=lr)
        best_param = param_init.clone()
        best_nll = float("inf")

        for _ in range(steps):
            optimizer.zero_grad()
            if not validate_gld_params(param.detach(), self.param_type):
                break
            # Invert Q(y) = d without tracking gradients through root-finding iterations,
            # then apply implicit differentiation via Q(y) and PDF_Q(y)
            with torch.no_grad():
                y = invert_quantile_newton_bisect(d, param.detach(), self.param_type, xtol=xtol)
                y = torch.clamp(y, 1e-6, 1.0 - 1e-6)

            # Implicit function theorem: dy/dtheta = - (dQ/dtheta) / Q'(y) = - (dQ/dtheta) * f_Q(y)
            f_q_detached = compute_pdf_quantile(y, param.detach(), self.param_type)
            y_diff = y - (compute_quantile(y, param, self.param_type) - d) * f_q_detached
            y_diff = torch.clamp(y_diff, 1e-6, 1.0 - 1e-6)

            pdf_vals = compute_pdf_quantile(y_diff, param, self.param_type)
            nll = -torch.mean(torch.log(torch.clamp(pdf_vals, min=1e-12)))

            if not torch.isfinite(nll):
                break
            if nll.item() < best_nll:
                best_nll = nll.item()
                best_param = param.detach().clone()

            nll.backward()
            optimizer.step()

        return best_param
