# CODEBASE index

Read this before exploring. Fork of upstream `gldpy` adding a PyTorch/CUDA backend; PyPI `gldpy-gpu`, import `gldpy_gpu`.

| Path | Role | Entry point |
|---|---|---|
| `gldpy_gpu/__init__.py` | Exports CPU `GLD`, `__version__`; lazy-loads `gpu` (PEP 562, needs torch) | `gldpy_gpu.GLD`, `gldpy_gpu.gpu` |
| `gldpy_gpu/_gld_module.py` | Upstream NumPy/SciPy `GLD` (1653 lines). CPU reference = test oracle | `GLD` |
| `gldpy_gpu/gpu.py` | Torch backend (577 lines) | `gpu.GLD` |
| `tests/` | pytest, GPU-vs-CPU parity; `device` fixture = cpu (+ cuda if usable) | `conftest.py` |
| `benchmarks/bench_gpu.py` | CPU-vs-GPU timing table; output in `benchmarks/results/` (gitignored, data) | `main()` |
| `pyproject.toml` | hatchling build, deps, pytest config, uv torch cu126 source (dev only) | |
| `uv.lock` | Lockfile (generated, do not edit) | |
| `dist/` | Build output (generated, gitignored) | |
| `ROADMAP.md` | Phases and bug log | |

## gpu.py
- Helpers: `_cuda_usable()` (probes kernels, warns and falls back to CPU), `_default_device()`, `validate_gld_params`, `compute_quantile`, `compute_pdf_quantile`, `invert_quantile_newton_bisect` (CDF inversion), `compute_sample_lmoments`, `compute_sample_percentiles`, `solve_vsl_lmm_closed_form`, `solve_shape_grid_refine`.
- `GLD` class: `check_param`, `Q`, `PDF_Q`, `supp`, `CDF_num`, `PDF_num`, `rand`, `GoF_Q_metric`, `fit_LMM`, `fit_PM`, `fit_ML`. Param types: RS, FMKL, VSL.
- Not ported to GPU: `fit_MM`, `fit_MPS`, `fit_starship` (CPU only; see ROADMAP "Later").

## tests/
- `test_gpu_core.py` Q/PDF/CDF/rand/supp parity; `test_gpu_stats.py` L-moments, percentiles; `test_gpu_fit.py` fit_LMM/PM/ML.
- `test_gpu_device.py` device selection/fallback; `test_lazy_import.py` PEP 562 loader; `test_packaging.py` version equality with pyproject.
- `test_speed.py` marked `gpu_speed`, deselected by default.
