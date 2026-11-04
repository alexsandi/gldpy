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

## Later / ideas
- GPU versions of `fit_MM`, `fit_MPS`, `fit_starship`
- Batched fitting (many datasets at once)
- CI (GitHub Actions, CPU-only tests)

## Bug log
- VSL `fit_PM`: grid contained l=0 -> NaN won `argmin` -> invalid params. Fixed (non-finite -> inf).
- `GLD(device=None)` stored None; now defaults to CUDA if available.
