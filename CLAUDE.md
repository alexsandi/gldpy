# gldpy-gpu

Fork of [alexsandi/gldpy](https://github.com/alexsandi/gldpy) (Generalized Lambda Distribution)
adding a PyTorch/CUDA backend. PyPI name: **gldpy-gpu**; import name is **`gldpy_gpu`**
(distinct from upstream `gldpy`, so both can coexist).

## Layout
- `gldpy_gpu/_gld_module.py` — original NumPy/SciPy `GLD` class (CPU reference; treat as ground truth).
- `gldpy_gpu/gpu.py` — torch `GLD` class + helpers. Loaded lazily via `gldpy_gpu.gpu` (PEP 562 in `__init__.py`).
- `tests/` — pytest. GPU-vs-CPU parity tests; device fixture runs on `cpu` and, if present, `cuda`.
- `benchmarks/` — CPU-vs-GPU speed scripts (`bench_gpu.py`), results in `benchmarks/results/` (gitignored).
- `ROADMAP.md` — plan and status.

## Workflow: test-driven
Write/extend a failing test first, see it fail for the right reason, then implement, then refactor.
CPU `gldpy_gpu.GLD` is the oracle for parity tests. Use float64 in tests; float32 is the runtime default.

## Commands (uv)
- `uv sync` — create `.venv`, install deps + dev group (torch comes from the cu126 index, see below).
- `uv run pytest` — fast suite (speed tests excluded).
- `uv run pytest -m gpu_speed -s` — GPU-vs-CPU speed checks (needs CUDA).
- `uv run python benchmarks/bench_gpu.py` — full benchmark table.
- `uv build` → `uv run twine check dist/*` — build and validate distributions.
- Publish (only when the user says so): `uv publish` (token via `UV_PUBLISH_TOKEN`); try TestPyPI first.

## Hardware gotcha: GTX 1070 (Pascal, sm_61)
Newer default PyTorch wheels (CUDA 12.8+/13) dropped sm_61 kernels. `[tool.uv.sources]` in
`pyproject.toml` pins torch to the **cu126** index for local dev. That setting is NOT published (the wheel only requires `torch>=2.0`, so the user's installer picks the build for their machine).
Old-GPU users install a cu126 torch themselves (see README); `gpu._cuda_usable()` detects a mismatch and falls back to CPU with a warning.
Verify with `torch.cuda.get_arch_list()` containing `sm_61`.

## Conventions
- Packaging is `pyproject.toml` only (hatchling). No `setup.py`. Keep `__version__` in `gldpy_gpu/__init__.py`
  equal to `[project].version` (a test enforces it).
- Never commit/push/publish without being asked.
- Keep upstream author credit and MIT license.

## Working rules
- Stay inside this folder. Files win over docs: if CLAUDE.md/README/ROADMAP disagree with the code, fix the doc.
- Facts vs hypotheses: state verified values with their source file; label guesses.
- `CODEBASE.md` is the file index; read it before exploring and update it when modules change.
- Generated, do not edit: `uv.lock`, `dist/`, `.venv/`, `benchmarks/results/`.
- Remote: `origin` = git@github.com:Kramer84/gldpy-opt.git. No secrets/`.env` in this repo.

Last validated against the real files: 2026-11-04 (`uv run pytest`: 152 passed, 4 skipped).
