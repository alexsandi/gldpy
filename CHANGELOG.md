# Changelog

Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions follow [Semantic Versioning](https://semver.org/);
before 1.0, a minor version may contain breaking changes, and they are marked **Breaking** here.

To release a new version, run `uv version --bump minor` (or `patch`, or `major`): `pyproject.toml` is the only file that
sets the version, and `gldpy_gpu.__version__` reads it from the installed metadata.

## [0.4.0] - unreleased

### Corrections to the original gldpy

`gldpy_gpu.GLD` is no longer an unmodified copy of `gldpy` 0.2: 14 defects are corrected, and
[docs/upstream-bugs.md](docs/upstream-bugs.md) explains each one, with measurements. `tests/test_upstream_bugs.py` tests every
defect against the real PyPI package and shows the correction, and it checks that nothing else changed.

- U01 RS `fit_PM` left its equations unsolved (missing `abs`), with a KS distance up to 2000x worse.
- U02 `CDF_num` and `PDF_num` were wrong in the tails, by a factor of 5 at probability 1e-6 and 6e6 at 1e-12.
- U03 Moments were NaN at lambda = 0, and kurtosis was 2.4e4 instead of 4.2 at lambda = 1e-5.
- U04 FMKL fits started at lambda = 0 returned NaN.
- U05 `rand` and `fit_curve` ignored seed 0 and reseeded the global generator.
- U06 `correct_supp` changed its argument, and truncated integer parameters.
- U07 `GoF_tests` raised on data outside the support.
- U08 `PDF_num` raised on 2-D input.
- U09 `Q` and `PDF_Q` gave NaN at exact endpoints, where the limit exists.
- U10 RS `grid_search` returned invalid parameters.
- U11 `fit_MPS` left out the two end spacings, so fits excluded observed data from the support.
- U12 `fit_MPS` was not reproducible with tied data.
- U13 Wrong signs in the documented RS and FMKL densities, and a mislabeled statistic in `fit_starship`.
- U14 `Q` lost digits as lambda approached 0.

### Added

- **`dtype` on every method** of `gpu.GLD`, and on the constructor. The order is: call, constructor, tensor or NumPy inputs, float32
  for plain Python numbers. Accepts `torch.float32`, `torch.float64`, `"float32"` or `"float64"`. Nothing is computed in another
  precision, and tests enforce it.
- **Tail methods** `SF_num`, `ISF` and `log_PDF_num`, on both `gpu.GLD` and `gldpy_gpu.GLD`.
- **`gpu.GLD` methods** `correct_supp`, and `initial_guess` on `fit_LMM`, `fit_PM` and `fit_ML`.
- **Optional PyTorch.** `pip install gldpy-gpu` no longer installs torch, and `pip install gldpy-gpu[gpu]` does. Without torch,
  `gpu.GLD` warns and runs on NumPy with the same calls. `gpu.BACKEND` says which backend is active.
- **Documentation.** [docs/numerics.md](docs/numerics.md), [docs/validation.md](docs/validation.md), [docs/benchmarks.md](docs/benchmarks.md) and
  [docs/upstream-bugs.md](docs/upstream-bugs.md).
- **Validation** against exactly known distributions (`tests/test_analytic.py`, `benchmarks/compare_analytic.py`), and against the
  frozen `gldpy` 0.2 (`tests/test_upstream_bugs.py`).
- **GitHub Actions** workflow: tests without torch and with CPU torch.

### Changed

- **Breaking:** torch must be requested explicitly, as above.
- **Breaking:** `gpu.GLD.fit_PM` is now `(data, initial_guess=None, u=0.1, ...)`, the same order as `gldpy_gpu.GLD`. A positional `u` no longer works.
- **Breaking:** `gpu.GLD.fit_ML` defaults to `steps=300`, because 60 often does not improve on the start.
- **Breaking:** NumPy float64 input stays float64, where it was silently converted to float32, and mixed inputs promote to the wider dtype.
- **Breaking:** `gldpy_gpu.GLD.CDF_num`, `SF_num`, `PDF_num`: `xtol` is the relative accuracy of the smaller tail probability, and defaults to 1e-8 (was an absolute tolerance on y, 1e-5). They now keep the shape of their input.
- **Breaking:** `gpu.GLD.CDF_num`: `xtol` has the same new meaning, and a floor of 32 machine epsilons. `max_iter` defaults to 64 (was 22).
- **Breaking:** the same seed gives different `gpu.GLD.rand` draws than 0.3.0. `gldpy_gpu.GLD.rand` keeps its stream.
- `gpu.solve_logit_cdf` and `invert_quantile_newton_bisect` are not differentiable. Use the implicit function theorem, as `fit_ML` does.
- `gpu.solve_vsl_lmm_closed_form` takes a prepared sample instead of raw data, and `solve_shape_grid_refine` is replaced by `solve_shape_multistart`. These are internal helpers.
- The torch code moved to `_torch_backend.py`, and `gpu.py` selects the backend.

### Fixed in the torch backend

- **Tails.** The CDF stopped at machine epsilon, and `PDF_num` clamped y, so far points reported a constant density.
- **float32 `rand`** returned `-inf` about once per 16.8M draws, and cut the upper tail at `Q(1 - 2**-24)`.
- **Batches.** `CDF_num`'s `xtol` was a maximum over the whole batch, so one far point changed the iteration for all. `supp` with batched parameters returned wrong values.
- **float32 validity.** RS `check_param` overflowed for lambda around 30.
- **Fits.** `fit_LMM` ignored `initial_guess` for RS and FMKL. Fixed-rate Adam left errors around 1e-2 in the L-moment and percentile fits. `fit_PM` could return a far root (FMKL 0.2, 0.3 became 3.55, 2.88).
- **`fit_ML`.** `lr` depended on the data's unit, and points outside the support received an artificial density.
- **Large data.** `fit_PM` failed above 2**24 points because of `torch.quantile`.
- **Bad data.** Fits raise `ValueError` on NaN, infinite or constant data.

### Performance

Measured on a GTX 1070, in [docs/benchmarks.md](docs/benchmarks.md). The log-space evaluation has more operations per element, and the optimizations more than recover that cost.

- Skip the series branch of the Box-Cox term unless a gradient in lambda is needed (6.5x faster).
- CDF solver: a table start, retiring finished elements, and a tolerance floor.
- Shape solves run on the CPU, and stop when nothing progresses.
- A direct form for the density at interior points.

## [0.3.0] - 2026-11-04

### Added

- PyTorch/CUDA backend `gldpy_gpu.gpu`, with `Q`, `PDF_Q`, `supp`, `CDF_num`, `PDF_num`, `rand`, `GoF_Q_metric`, `fit_LMM`, `fit_PM` and `fit_ML` for RS, FMKL and VSL.
- Device selection with a CPU fallback when the installed torch has no kernels for the GPU.
- Parity tests against the NumPy class, and a benchmark script.

### Changed

- Fork of `gldpy` 0.2 by Aleksandra Alekseeva, published as `gldpy-gpu`, imported as `gldpy_gpu`, so that it can coexist with `gldpy`.
- Packaging moved from `setup.py` to `pyproject.toml`.

[0.4.0]: https://github.com/Kramer84/gldpy-gpu/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/Kramer84/gldpy-gpu/releases/tag/v0.3.0
