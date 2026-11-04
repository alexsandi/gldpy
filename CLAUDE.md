# gldpy-gpu

Fork of [alexsandi/gldpy](https://github.com/alexsandi/gldpy) (Generalized Lambda Distribution)
adding a PyTorch/CUDA backend. PyPI name: **gldpy-gpu**; import name is **`gldpy_gpu`**
(distinct from upstream `gldpy`, so both can coexist).

## Layout
- `gldpy_gpu/_gld_module.py` — NumPy/SciPy `GLD`, corrected from `gldpy` 0.2 (see `docs/upstream-bugs.md`).
- `gldpy_gpu/gpu.py` — picks the backend, loaded lazily via `gldpy_gpu.gpu` (PEP 562 in `__init__.py`).
- `gldpy_gpu/_torch_backend.py` — torch `GLD` class + helpers.
- `gldpy_gpu/_numpy_fallback.py` — used when torch is missing: warns, same calls, runs on NumPy.
- `tests/` — pytest. GPU-vs-CPU parity tests; device fixture runs on `cpu` and, if present, `cuda`.
- `benchmarks/` — CPU-vs-GPU speed scripts (`bench_gpu.py`), results in `benchmarks/results/` (gitignored).
- `docs/` — bug analysis, numerical methods, validation, benchmarks. `CHANGELOG.md` — all changes. `ROADMAP.md` — plan and status.

## Workflow: test-driven
Write/extend a failing test first, see it fail for the right reason, then implement, then refactor.
Oracles, most independent first: closed forms and published values (`tests/test_analytic.py`, `tests/test_gpu_tails.py`); PyPI `gldpy` 0.2, which is frozen,
outside its defects (`tests/test_upstream_bugs.py`); `gldpy_gpu.GLD` for parity between backends. A newly found defect of the original gets a pair of tests there and an entry in `docs/upstream-bugs.md`.
Precision is the caller's: no code path may convert to another dtype (`tests/test_gpu_dtype.py` enforces it). float32 is the default only for plain Python numbers.

## Commands (uv)
- `uv sync` — create `.venv`, install deps + dev group, which includes torch (from the cu126 index, see below).
- `uv run pytest` — fast suite (speed and slow tests excluded); `uv run pytest -m "not gpu_speed"` adds the slow ones.
- `uv run pytest -m gpu_speed -s` — GPU-vs-CPU speed checks (needs CUDA).
- `uv run python benchmarks/bench_gpu.py`, `benchmarks/compare_analytic.py`, then `benchmarks/make_tables.py` — regenerate the measured tables in `docs/`.
- `uv version --bump minor` (or `patch`, `major`) — the only way to change the version; add a `CHANGELOG.md` section.
- `uv build` → `uv run twine check dist/*` — build and validate distributions.
- Publish (only when the user says so): `uv publish` (token via `UV_PUBLISH_TOKEN`); try TestPyPI first.

## Hardware gotcha: GTX 1070 (Pascal, sm_61)
Newer default PyTorch wheels (CUDA 12.8+/13) dropped sm_61 kernels. `[tool.uv.sources]` in
`pyproject.toml` pins torch to the **cu126** index for local dev. That setting is NOT published. In the wheel, torch is the optional extra `gldpy-gpu[gpu]` (`torch>=2.0`), so the user's installer picks the build for their machine.
Old-GPU users install a cu126 torch themselves (see README); `gpu._cuda_usable()` detects a mismatch and falls back to CPU with a warning.
Verify with `torch.cuda.get_arch_list()` containing `sm_61`.

## Conventions
- Packaging is `pyproject.toml` only (hatchling). No `setup.py`. `[project].version` is the only source of the version; `__version__` reads the installed metadata.
- Numbers in `docs/` come from measurements on the current code, and tables between `<!-- table:... -->` markers are generated. Never edit those by hand.
- Never commit/push/publish without being asked.
- Keep upstream author credit and MIT license.

## Working rules
- Stay inside this folder. Files win over docs: if CLAUDE.md/README/ROADMAP disagree with the code, fix the doc.
- Facts vs hypotheses: state verified values with their source file; label guesses.
- `CODEBASE.md` is the file index; read it before exploring and update it when modules change.
- Generated, do not edit: `uv.lock`, `dist/`, `.venv/`, `benchmarks/results/`.
- Remote: `origin` = git@github.com:Kramer84/gldpy-gpu.git. No secrets/`.env` in this repo.

Last validated against the real files: 2026-11-04 (`uv run pytest -m "not gpu_speed"`: 1313 passed, 6 skipped; `-m gpu_speed`: 2 passed).
