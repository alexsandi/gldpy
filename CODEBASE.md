# CODEBASE index

Read this before exploring. Fork of upstream `gldpy` adding a PyTorch/CUDA backend; PyPI `gldpy-gpu`, import `gldpy_gpu`.

| Path | Role | Entry point |
|---|---|---|
| `gldpy_gpu/__init__.py` | Exports CPU `GLD`, `__version__`; lazy-loads `gpu` (PEP 562) | `gldpy_gpu.GLD`, `gldpy_gpu.gpu` |
| `gldpy_gpu/_gld_module.py` | Corrected NumPy/SciPy `GLD`, derived from `gldpy` 0.2; corrections are listed in `docs/upstream-bugs.md` | `GLD` |
| `gldpy_gpu/gpu.py` | Backend selector: torch if importable, else NumPy fallback. `BACKEND`; forwards other names to the backend | `gpu.GLD` |
| `gldpy_gpu/_torch_backend.py` | Torch implementation | `GLD` |
| `gldpy_gpu/_numpy_fallback.py` | Torch-free `gpu.GLD`: the NumPy class with the torch call signatures, warns on construction | `GLD` |
| `tests/` | pytest, GPU-vs-CPU parity; `device` fixture = cpu (+ cuda if usable) | `conftest.py` |
| `benchmarks/bench_gpu.py` | Timings across backends and sizes, and speed factors; output in `benchmarks/results/` (gitignored) | `main()` |
| `benchmarks/compare_analytic.py` | Fits of known distributions on every backend; tail accuracy | `main()` |
| `benchmarks/make_tables.py` | Writes the result tables into `docs/validation.md` and `docs/benchmarks.md` | `main()` |
| `docs/` | `upstream-bugs.md`, `numerics.md`, `validation.md`, `benchmarks.md` | |
| `CHANGELOG.md` | All changes | |
| `.github/workflows/ci.yml` | Tests without torch and with CPU torch, and the build | |
| `pyproject.toml` | hatchling build, deps, pytest config, uv torch cu126 source (dev only) | |
| `uv.lock` | Lockfile (generated, do not edit) | |
| `dist/` | Build output (generated, gitignored) | |
| `ROADMAP.md` | Phases and bug log | |

## _torch_backend.py
- Device: `_cuda_usable()` (probes kernels, warns and falls back to CPU), `_default_device()`.
- Core works on log(y) and log(1-y): `_box_cox`, `_pow_minus_one`, `_pow_from_log`, `_quantile_from_logs`, `_density_terms`, `_log_density_from_logs`, plus y-based `compute_quantile`, `compute_pdf_quantile`. Sampling: `_sample_log_uniform`.
- CDF: `solve_logit_cdf` (safeguarded Newton on logit(y), `_table_start`, retires finished elements, optional warm start), wrapper `invert_quantile_newton_bisect`.
- Statistics: `compute_sample_lmoments`, `compute_sample_percentiles`, `_require_fit_data`.
- Fit solvers run on the CPU, in the data's dtype: `solve_shape_multistart` (grid, then `_levenberg_marquardt` from each local minimum, then lowest Q-Q among exact roots), `_lmm_equations`, `_pm_equations`, `_solve_vsl_pm`, `solve_vsl_lmm_closed_form`, `_with_support_covering` (for `fit_ML`).
- Dtype rule: `GLD._resolve_dtype`, `_as_dtype`.
- `GLD` class: `check_param`, `Q`, `ISF`, `PDF_Q`, `supp`, `CDF_num`, `SF_num`, `PDF_num`, `log_PDF_num`, `rand`, `GoF_Q_metric`, `correct_supp`, `fit_LMM`, `fit_PM`, `fit_ML`. Param types: RS, FMKL, VSL.
- Not ported: `mean`, `var`, `std`, `skewness`, `kurtosis`, `median`, `GoF_tests`, `plot_*`, `grid_search`, `fit_MM`, `fit_MPS`, `fit_starship`, `fit_curve` (NumPy `GLD` only; see ROADMAP "Later").

## tests/
- `test_gpu_core.py` Q/PDF/CDF/rand/supp parity; `test_gpu_stats.py` L-moments, percentiles; `test_gpu_fit.py` fit_LMM/PM/ML basics.
- `test_upstream_bugs.py` each defect of PyPI `gldpy` 0.2 as a pair (original fails, ours passes), and parity with it elsewhere.
- `test_analytic.py` fits of exactly known distributions and published values.
- `test_gpu_tails.py` tails against closed forms (logistic, exponential, Pareto), float32/float64, `rand` tails.
- `test_gpu_dtype.py` the dtype rule, with a `TorchFunctionMode` that fails on any tensor of the wrong precision.
- `test_gpu_fast_paths.py` optimized paths against the general ones, solver iteration budgets.
- `test_gpu_fit_quality.py` fits against the CPU oracle, multiple roots, scale invariance, bad data, support.
- `test_optional_torch.py` torch-blocked subprocess runs and the NumPy fallback; runs without torch (conftest then skips `test_gpu_*`).
- `test_gpu_device.py` device selection/fallback; `test_lazy_import.py` PEP 562 loader; `test_packaging.py` version equality with pyproject.
- `test_speed.py` marked `gpu_speed`, and the longest tests marked `slow`: both deselected by default.
