# Roadmap

Legend: [x] done, [ ] todo

## Phase 0 — Project hygiene
- [x] Review existing code (`gpu.py` untracked, `__init__` lazy loader)
- [x] Migrate `setup.py` → `pyproject.toml` (hatchling), uv dev workflow
- [x] `CLAUDE.md`, `ROADMAP.md`, `.gitignore`

## Phase 1 — Verify GPU backend (TDD)
- [x] Working torch (cu126) on GTX 1070, confirm `sm_61` support
- [x] Parity tests GPU vs CPU: `check_param`, `Q`, `PDF_Q`, `supp`, `CDF_num`, `PDF_num`, `rand`, `GoF_Q_metric`
- [x] Stats helper tests: L-moments, percentiles
- [x] Fit tests: `fit_LMM`, `fit_PM`, `fit_ML` for RS / FMKL / VSL
- [x] Fix every bug the tests uncover (log each below)
- [x] Default device = CUDA when available

## Phase 2 — Speed
- [x] `benchmarks/bench_gpu.py`: Q, rand, CDF_num, PDF_num, fit_* vs N; float32 vs float64
- [x] `pytest -m gpu_speed` regression checks (GPU must beat CPU on large N)
- [x] Record 1070 results in README

## Phase 3 — Release as `gldpy-gpu`
- [x] README: install section (torch/Pascal note), `gldpy.gpu` usage, upstream credit, conflict with `gldpy`
- [x] `uv build`, `twine check`, wheel installs and runs on CUDA (name `gldpy-gpu` verified free on PyPI 2026-11-02)
- [x] Upload to PyPI (gldpy-gpu 0.3.0 live, verified 2026-11-04)
- [x] Tag `v0.3.0` (pushed)

- [x] Rename import package `gldpy` -> `gldpy_gpu` (coexists with upstream)
- [x] Runtime check: torch build lacks kernels for GPU -> warn + CPU fallback

## Phase 4 — Robustness
- [x] torch is an optional extra; without it `gpu.GLD` warns and runs on NumPy (`_numpy_fallback.py`)
- [x] Tails: log-space core, logit-space CDF, new `SF_num`, `ISF`, `log_PDF_num`
- [x] float32/float64: `dtype=` argument, no silent downcast, dtype promotion, float64 uniforms in `rand`
- [x] Fits: float64 multistart Levenberg-Marquardt, `initial_guess` honored, root chosen by Q-Q fit, `correct_supp`
- [x] Correct the defects of `gldpy` 0.2 in `_gld_module.py`, with PyPI `gldpy` as evidence (`docs/upstream-bugs.md`)
- [x] `dtype` per call, no forced precision
- [x] Validation against known distributions (`docs/validation.md`), optimizations and benchmarks (`docs/benchmarks.md`)
- [ ] First CI run (the workflow has not run yet), including Python 3.9
- [ ] Release 0.4.0 (version set, not published)

## Later / ideas
- GPU versions of `fit_MM`, `fit_MPS`, `fit_starship`, `fit_curve`
- `mean`, `var`, `std`, `skewness`, `kurtosis`, `median`, `GoF_tests` on the torch class
- Quasi-Newton polish for `fit_ML`: Adam may not have converged at 300 steps (lognormal in `docs/validation.md`)
- Split the code as described under "Structure" below

## Structure
Candidates, not done, in order of value:
1. One math core. `_torch_backend.py` and `_gld_module.py` implement the same formulas twice, and fixes have had to be made in both. A module written against an array namespace could serve both.
2. Split `_torch_backend.py` (about 1000 lines) into `core`, `solve`, `stats`, `fit` modules, keeping `GLD` as the deep interface.
3. A separate result type for fits, in place of the `disp_*`/`test_gof` flags that mix computing with printing in `_gld_module.py`.
4. Separate plotting from computation, so that matplotlib becomes optional as well.
- Batched fitting (many datasets at once)
- CI (GitHub Actions, CPU-only tests)

## Bug log
- VSL `fit_PM`: grid contained l=0 -> NaN won `argmin` -> invalid params. Fixed (non-finite -> inf).
- `GLD(device=None)` stored None; now defaults to CUDA if available.
- Tails: `CDF_num` bracket stopped at eps (tail 1e-12 was off by 1e5), `xtol` was a global max over the batch, `PDF_num` clamped y and reported a constant density far out. Fixed.
- float32 `rand`: 2**-24 uniform step gave `-inf` about once per 16M draws and cut the upper tail at Q(1 - 2**-24). Fixed. Same seed now gives different draws.
- NumPy float64 input was silently cast to float32 (`Q(1 - 1e-10)` returned `inf`); mixed dtypes rounded to `param`'s. Fixed.
- Near-zero lambda: `(u**l - 1)/l` lost half its digits in float32 and jumped at the |l| = 1e-7 switch to log. Fixed with `expm1`.
- `supp` with batched params paired bounds with params element-wise and returned wrong values. Fixed.
- RS validity overflowed float32 for lambda ~ 30. Fixed (compared in log space).
- RS/FMKL fits: fixed-lr Adam stopped ~1e-2 short (RS heavy: 3.8e-2 residual); `fit_LMM` ignored `initial_guess`; `fit_PM` could return a far exact root (FMKL 0.2, 0.3 -> 3.55, 2.88). Fixed.
- `fit_PM` signature now matches the CPU class: `(data, initial_guess, u)`. Positional `u` breaks.
- `fit_ML`: `lr` was in raw parameter units, points outside the support got a fake density. Fixed (standardized data, `correct_supp`, rejected steps).
- `torch.quantile` raised above 2**24 elements, killing `fit_PM` on large data. Fixed.
- Fits now raise `ValueError` on NaN, inf or constant data.
