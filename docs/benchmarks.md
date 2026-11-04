# Benchmarks: what determines speed, and what a GPU gains

## Summary

For large samples, a consumer GPU from 2016 runs the array operations of this package about 20 to 80 times faster than a 4-core CPU with the same PyTorch code, and several hundred times faster than NumPy. The **same GPU is slower below about $10^5$ points**, because every operation pays a launch cost. The advantage therefore grows with sample size, and for the solvers it comes on top of a large algorithmic gain over the original.

Three separate effects are at work, and only the first is due to the GPU:

1. **Parallel hardware**, for operations on large arrays.
2. **Vectorization**: NumPy's `CDF_num` calls a root finder once per point, in Python. The batched solver is more than 1000 times faster at $10^5$ points on the *CPU* (`torch CPU f32` against `NumPy` in the `CDF_num` table).
3. **Algorithm**: a table start and retiring of finished elements reduce the work per point, on any device.

## Method

**Environment.**

<!-- table:meta -->
| Item | Value |
|---|---|
| gldpy-gpu | 0.4.0 |
| PyTorch | 2.14.1+cu126 |
| NumPy | 2.4.6 |
| Python | 3.11.15 |
| GPU | NVIDIA GeForce GTX 1070 |
| CPU threads used by torch | 4 |
| Date | 2026-11-04 |
<!-- /table -->

The CPU is a Xeon E5-1620 v2 (4 cores, 8 threads, 3.7 GHz, 2013, AVX without AVX2). RAM is DDR3-1600. The GPU has 8 GiB, of which about 7 GiB was free, and its compute capability is 6.1 (Pascal).

**Procedure** (`benchmarks/bench_gpu.py`).

- Each case runs once as an untimed warm-up, then up to 7 times or for 4 s. Reported values are medians.
- CUDA runs are bracketed by `torch.cuda.synchronize()`, since CUDA calls return before the work is done.
- All inputs are on the measured device before the clock starts, so **transfer time is excluded**. It has its own table below.
- The distribution is FMKL(0, 1, 0.2, 0.3), and the other parameterizations are compared separately.

**Limitations of these measurements.**

- The machine was running its usual background services and a desktop, and it was not isolated. RAM was tight, with about 4 GiB free out of 46 GiB. So the timings have noise, and the largest NumPy runs may include memory pressure: from $10^6$ to $10^7$ points, `Q` in NumPy grows about 25-fold, where linear scaling would give 10.
- A different GPU, CPU or PyTorch version will change these numbers. Compare ratios and trends, not absolute values. A newer GPU should move the crossover to smaller $N$, and float64 relatively closer to float32, but this was not measured here.
- `fit_LMM` and `fit_PM` do different work in the two families. NumPy refines from one given start and the torch backend searches for all roots, so their ratio is not a pure hardware comparison.

## Results

Time per call, median. Empty cells were not run, because of memory or duration limits.

### Quantile function and sampling

**`Q`**

<!-- table:sweep_Q -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 44 µs | 53 µs | 262 µs | 235 µs | 805 µs | 798 µs |
| 1000 | 107 µs | 107 µs | 301 µs | 309 µs | 803 µs | 808 µs |
| 1e+04 | 703 µs | 579 µs | 480 µs | 605 µs | 793 µs | 800 µs |
| 1e+05 | 6.67 ms | 5.29 ms | 1.56 ms | 1.79 ms | 796 µs | 1.04 ms |
| 1e+06 | 83.9 ms | 72.6 ms | 14.8 ms | 30.7 ms | 983 µs | 2.51 ms |
| 1e+07 | 1.7 s | 1.8 s | 237 ms | 324 ms | 6.08 ms | 20.2 ms |
| 3e+07 |  |  |  |  | 17.4 ms | 55.3 ms |
| 1e+08 |  |  |  |  | 57.7 ms |  |
<!-- /table -->

**`PDF_Q`**

<!-- table:sweep_PDF_Q -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 32 µs | 50 µs | 345 µs | 313 µs | 1.08 ms | 1.04 ms |
| 1000 | 91 µs | 109 µs | 377 µs | 362 µs | 1.04 ms | 1.03 ms |
| 1e+04 | 644 µs | 672 µs | 702 µs | 775 µs | 1.04 ms | 1.04 ms |
| 1e+05 | 6.07 ms | 6.29 ms | 1.79 ms | 2.29 ms | 1.04 ms | 1.05 ms |
| 1e+06 | 69.5 ms | 75.9 ms | 9.13 ms | 28.0 ms | 1.25 ms | 2.52 ms |
| 1e+07 | 1.5 s | 1.6 s | 198 ms | 322 ms | 7.08 ms | 19.5 ms |
| 3e+07 |  |  |  |  | 19.9 ms | 58.3 ms |
| 1e+08 |  |  |  |  | 65.6 ms |  |
<!-- /table -->

**`rand`**

<!-- table:sweep_rand -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 48 µs | 57 µs | 572 µs | 299 µs | 995 µs | 958 µs |
| 1000 | 121 µs | 116 µs | 396 µs | 541 µs | 1.08 ms | 987 µs |
| 1e+04 | 779 µs | 649 µs | 1.19 ms | 1.37 ms | 1.40 ms | 1.40 ms |
| 1e+05 | 7.42 ms | 6.27 ms | 4.36 ms | 5.17 ms | 1.42 ms | 1.43 ms |
| 1e+06 | 93.0 ms | 85.3 ms | 24.2 ms | 41.0 ms | 1.73 ms | 3.34 ms |
| 1e+07 | 1.9 s | 2.1 s | 377 ms | 550 ms | 9.64 ms | 26.1 ms |
| 3e+07 |  |  |  |  | 27.0 ms | 73.7 ms |
| 1e+08 |  |  |  |  | 88.9 ms |  |
<!-- /table -->

**`GoF_Q_metric`** (includes a sort)

<!-- table:sweep_GoF_Q_metric -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 80 µs | 86 µs | 384 µs | 350 µs | 1.12 ms | 1.05 ms |
| 1000 | 181 µs | 171 µs | 467 µs | 460 µs | 1.05 ms | 1.03 ms |
| 1e+04 | 1.36 ms | 1.08 ms | 1.54 ms | 1.68 ms | 1.06 ms | 1.07 ms |
| 1e+05 | 12.7 ms | 10.8 ms | 9.34 ms | 10.4 ms | 1.06 ms | 1.08 ms |
| 1e+06 | 230 ms | 143 ms | 99.0 ms | 120 ms | 2.07 ms | 4.29 ms |
| 1e+07 | 3.5 s | 3.8 s | 1.3 s | 1.5 s | 16.3 ms | 38.4 ms |
| 3e+07 |  |  |  |  | 48.6 ms |  |
<!-- /table -->

### Numerical inversion

**`CDF_num`**

<!-- table:sweep_CDF_num -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 42.9 ms | 28.8 ms | 2.98 ms | 2.81 ms | 7.27 ms | 7.24 ms |
| 1000 | 422 ms | 281 ms | 2.61 ms | 2.64 ms | 7.36 ms | 7.28 ms |
| 1e+04 | 4.2 s | 2.9 s | 4.72 ms | 5.99 ms | 7.36 ms | 7.60 ms |
| 1e+05 |  | 29 s | 16.1 ms | 23.3 ms | 7.84 ms | 7.65 ms |
| 1e+06 |  |  | 155 ms | 256 ms | 11.3 ms | 22.0 ms |
| 1e+07 |  |  | 2.4 s | 4.5 s | 86.3 ms | 185 ms |
| 3e+07 |  |  |  |  | 252 ms |  |
<!-- /table -->

**`PDF_num`**

<!-- table:sweep_PDF_num -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 40.9 ms | 29.0 ms | 3.23 ms | 2.90 ms | 9.71 ms | 10.2 ms |
| 1000 | 418 ms | 294 ms | 3.51 ms | 3.51 ms | 10.0 ms | 9.85 ms |
| 1e+04 | 4.1 s | 2.8 s | 7.87 ms | 9.01 ms | 9.90 ms | 11.0 ms |
| 1e+05 |  |  | 26.8 ms | 39.2 ms | 10.2 ms | 10.5 ms |
| 1e+06 |  |  | 212 ms | 328 ms | 14.0 ms | 26.6 ms |
| 1e+07 |  |  | 3.1 s | 5.0 s | 104 ms | 232 ms |
| 3e+07 |  |  |  |  | 304 ms |  |
<!-- /table -->

### Fits

**`fit_LMM`**

<!-- table:sweep_fit_LMM -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 5.09 ms | 4.88 ms | 66.2 ms | 103 ms | 65.7 ms | 101 ms |
| 1000 | 15.9 ms | 15.6 ms | 13.9 ms | 106 ms | 83.7 ms | 84.7 ms |
| 1e+04 | 139 ms | 133 ms | 73.7 ms | 107 ms | 93.7 ms | 107 ms |
| 1e+05 | 1.3 s | 1.3 s | 111 ms | 126 ms | 81.6 ms | 110 ms |
| 1e+06 |  |  | 284 ms | 344 ms | 82.5 ms | 121 ms |
| 1e+07 |  |  | 2.4 s | 3.1 s | 118 ms | 179 ms |
| 3e+07 |  |  |  |  | 192 ms |  |
<!-- /table -->

**`fit_PM`**

<!-- table:sweep_fit_PM -->
| N | gldpy 0.2 | NumPy | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|---|---|
| 100 | 5.74 ms | 5.31 ms | 373 ms | 364 ms | 277 ms | 376 ms |
| 1000 | 4.85 ms | 4.10 ms | 284 ms | 364 ms | 220 ms | 266 ms |
| 1e+04 | 4.67 ms | 4.59 ms | 229 ms | 366 ms | 172 ms | 363 ms |
| 1e+05 | 12.7 ms | 12.8 ms | 233 ms | 349 ms | 346 ms | 342 ms |
| 1e+06 | 101 ms | 103 ms | 477 ms | 545 ms | 218 ms | 374 ms |
| 1e+07 |  |  | 2.1 s | 2.4 s | 274 ms | 390 ms |
| 3e+07 |  |  |  |  | 314 ms |  |
<!-- /table -->

**`fit_ML`** with `steps=60`. The default of 300 steps costs about 3 times as much, see below.

<!-- table:sweep_fit_ML -->
| N | torch CPU f32 | torch CPU f64 | CUDA f32 | CUDA f64 |
|---|---|---|---|---|
| 100 | 450 ms | 463 ms | 1.3 s | 1.3 s |
| 1000 | 458 ms | 549 ms | 1.4 s | 1.5 s |
| 1e+04 | 796 ms | 969 ms | 2.9 s | 1.5 s |
| 1e+05 | 2.6 s | 3.0 s | 3.1 s | 1.7 s |
| 1e+06 |  |  | 2.8 s | 3.2 s |
<!-- /table -->

## The advantage of the GPU

CUDA float32 against the alternatives, at the largest size both ran. "CUDA wins from" is the smallest tested size, on a decade grid, at which CUDA is faster.

<!-- table:speedup -->
| Operation | Against | At N | Its time | CUDA f32 | Speedup | CUDA wins from N = |
|---|---|---|---|---|---|---|
| Q | torch CPU f32 | 1e+07 | 237 ms | 6.08 ms | 39x | 1e+05 |
| Q | NumPy | 1e+07 | 1.8 s | 6.08 ms | 296x | 1e+05 |
| PDF_Q | torch CPU f32 | 1e+07 | 198 ms | 7.08 ms | 28x | 1e+05 |
| PDF_Q | NumPy | 1e+07 | 1.6 s | 7.08 ms | 232x | 1e+05 |
| rand | torch CPU f32 | 1e+07 | 377 ms | 9.64 ms | 39x | 1e+05 |
| rand | NumPy | 1e+07 | 2.1 s | 9.64 ms | 219x | 1e+05 |
| CDF_num | torch CPU f32 | 1e+07 | 2.4 s | 86.3 ms | 28x | 1e+05 |
| CDF_num | NumPy | 1e+05 | 29 s | 7.84 ms | 3681x | 1e+02 |
| PDF_num | torch CPU f32 | 1e+07 | 3.1 s | 104 ms | 30x | 1e+05 |
| PDF_num | NumPy | 1e+04 | 2.8 s | 9.90 ms | 285x | 1e+02 |
| GoF_Q_metric | torch CPU f32 | 1e+07 | 1.3 s | 16.3 ms | 78x | 1e+04 |
| GoF_Q_metric | NumPy | 1e+07 | 3.8 s | 16.3 ms | 234x | 1e+04 |
| fit_LMM | torch CPU f32 | 1e+07 | 2.4 s | 118 ms | 20x | 1e+02 |
| fit_LMM | NumPy | 1e+05 | 1.3 s | 81.6 ms | 16x | 1e+04 |
| fit_PM | torch CPU f32 | 1e+07 | 2.1 s | 274 ms | 7.7x | 1e+02 |
| fit_PM | NumPy | 1e+06 | 103 ms | 218 ms | 0.5x | not reached |
| fit_ML | torch CPU f32 | 1e+05 | 2.6 s | 3.1 s | 0.8x | not reached |
<!-- /table -->

### Why the advantage grows with $N$

The time of an operation is approximately

$$T(N) \approx T_\text{fixed} + \frac{N}{R},$$

with a fixed cost and a throughput $R$. The two devices are at opposite ends:

| | Fixed cost per operation | Throughput $R$, single addition, float32 |
|---|---|---|
| **CUDA** | high: about 43 µs per launch; `Q` has a floor of about 0.8 ms, equal to about 18 launches | about $2\cdot10^{10}$ elements/s |
| **CPU** | low: about 5 µs | about $10^{9}$ elements/s |

The crossover, where the two lines meet, is thus around $N \approx 10^4$ to $10^5$. Well above it, the ratio tends to the ratio of throughputs. The flat line in the CUDA columns below $10^5$ points is the launch cost, and it is the signature of this behavior: up to $10^5$ points, `Q` on CUDA takes about 0.8 ms *whatever the size*.

**Practical rule.** For a single evaluation of $Q$ below about $3\cdot10^4$ points, use the CPU. For repeated calls, the same applies per call. Above $10^5$ the GPU wins everywhere, and at $10^7$ it is decisive, with `Q` at 6 ms against 240 ms.

### What only large samples can do

Within the 7 GiB this GPU had free, one call at float32 can handle the sizes below. So, for example, $10^8$ draws take about 0.09 s and 3 GiB. On this machine's CPU the same call would take around 4 s, by extrapolation from $10^7$.

<!-- table:memory -->
| Operation | N | Peak GPU memory, float32 | Bytes per element | Largest N in 7 GiB (extrapolated) |
|---|---|---|---|---|
| Q | 1e+08 | 2.61 GiB | 28 | 2.7e+08 |
| rand | 1e+08 | 3.08 GiB | 33 | 2.3e+08 |
| CDF_num | 3e+07 | 2.99 GiB | 107 | 7.0e+07 |
| fit_LMM | 3e+07 | 1.56 GiB | 56 | 1.3e+08 |
| fit_PM | 3e+07 | 1.13 GiB | 40 | 1.9e+08 |
<!-- /table -->

The practical application is Monte Carlo with far tails: resolving a probability of $10^{-6}$ with a 1% standard error takes $10^{10}$ draws. In batches of $10^8$, that is 100 batches and about 9 s on this GPU for sampling alone. At the CPU's rate it would take several minutes, and with NumPy about 35 minutes.

Equally, for a tail probability, a *direct* computation is better than simulation: `SF_num` returns $10^{-30}$ with relative accuracy $10^{-14}$ in well under a microsecond per point ([validation.md](validation.md#4-tail-accuracy)).

## What acts on speed

### Launch cost and transfers

<!-- table:factor_launch -->
| Device | Trivial op on 1 element |
|---|---|
| CUDA | 43 µs |
| CPU | 5 µs |
<!-- /table -->

<!-- table:factor_transfer -->
| N, float32 | CPU to GPU | GPU to CPU |
|---|---|---|
| 1e+04 | 60 µs | 57 µs |
| 1e+06 | 617 µs | 608 µs |
| 1e+07 | 6.32 ms | 23.8 ms |
| 3e+07 | 18.6 ms | 73.3 ms |
<!-- /table -->

This is why the shape solves of `fit_LMM` and `fit_PM` run on the CPU: they are many operations on arrays of 16 elements, where launch cost is everything. `fit_LMM` on CUDA is close to constant up to $10^6$ points, because the shape solve dominates and depends only on the grid.

**Transfers** are often the true bottleneck. Moving $10^7$ float32 values to the GPU costs about 6 ms, the same as evaluating `Q` on them, and moving them back costs about 24 ms. The asymmetry above $10^6$ points is measured; the cause was not investigated. Keep data on the GPU between calls, and bring only summaries back. Note that a NumPy input is copied to the device at every call.

### Floating-point precision

Time for one operation on $10^7$ values:

<!-- table:factor_primitives -->
| Device | dtype | add | mul | exp | log | expm1 | sort |
|---|---|---|---|---|---|---|---|
| cpu | float32 | 9.22 ms | 8.49 ms | 11.6 ms | 13.1 ms | 42.4 ms | 960 ms |
| cpu | float64 | 18.3 ms | 20.7 ms | 24.1 ms | 24.7 ms | 48.3 ms | 995 ms |
| cuda | float32 | 421 µs | 424 µs | 420 µs | 421 µs | 422 µs | 6.23 ms |
| cuda | float64 | 877 µs | 879 µs | 1.61 ms | 3.00 ms | 2.06 ms | 15.1 ms |
<!-- /table -->

- **On the CPU**, float64 costs about 2x for basic arithmetic: the vector registers hold half as many elements. `expm1` has about the same cost at both precisions, which suggests that it is not vectorized, unlike `exp` and `log`.
- **On this GPU**, float64 costs 2x for `add` and `mul`, 4x for `exp`, and 7x for `log`. Pascal consumer GPUs have limited double-precision throughput, and the transcendental functions take the largest penalty.
- The float32 advantage therefore grows with the share of `log` and `exp`, which is why `Q` is about 3x faster in float32.

### Solver settings

`CDF_num` on $10^6$ points, for various `xtol`:

<!-- table:factor_xtol -->
| xtol | float32 | float64 |
|---|---|---|
| 1e-03 | 8.85 ms | 13.6 ms |
| 1e-04 | 12.7 ms | 21.6 ms |
| 1e-06 | 11.6 ms | 21.6 ms |
| 1e-08 | 12.1 ms | 25.7 ms |
| 1e-10 | 13.4 ms | 29.3 ms |
| 1e-12 | 11.4 ms | 29.1 ms |
<!-- /table -->

A tighter `xtol` is nearly free: Newton's convergence is quadratic, and past the first few iterations it adds at most one more. There is little reason to loosen it for speed. float32's floor of $32\epsilon \approx 4\cdot10^{-6}$ makes the smaller values there equivalent.

**Starting point.** The table start is worth a factor of about 3:

<!-- table:factor_start -->
| dtype | Table start | Start at t = 0 | Gain |
|---|---|---|---|
| float32 | 11.0 ms | 34.2 ms | 3.1x |
| float64 | 20.8 ms | 56.6 ms | 2.7x |
<!-- /table -->

**Where the data lie** does not matter much, thanks to the logit-space formulation:

<!-- table:factor_region -->
| Where the 1e6 points lie | CDF_num, float64 |
|---|---|
| center: y in (0.4, 0.6) | 20.6 ms |
| all: y in (0, 1) | 21.1 ms |
| far tail: y < 1e-6 | 20.1 ms |
<!-- /table -->

**The parameterization** has little influence. RS and VSL cost slightly more than FMKL in the density:

<!-- table:factor_parameterization -->
| Type | Shape | Q, 1e7 | PDF_Q, 1e7 | Q + CDF_num, 1e6 |
|---|---|---|---|---|
| FMKL | (0.2, 0.3) | 6.04 ms | 7.15 ms | 12.0 ms |
| FMKL | (-0.2, -0.1) | 6.05 ms | 7.08 ms | 11.9 ms |
| RS | (0.1, 0.15) | 7.09 ms | 8.95 ms | 15.6 ms |
| VSL | (0.4, 0.3) | 7.10 ms | 7.18 ms | 13.7 ms |
<!-- /table -->

### CPU threads

<!-- table:factor_threads -->
| Threads | Q, N = 1e7 | Scaling |
|---|---|---|
| 1 | 661 ms | 1.0x |
| 2 | 375 ms | 1.8x |
| 4 | 238 ms | 2.8x |
| 8 | 253 ms | 2.6x |
<!-- /table -->

Scaling is sublinear, and 8 threads run on 4 physical cores. Running with 4 threads gives most of the available speed. Memory bandwidth on this DDR3 machine is the likely limit, though this was not isolated.

### Composition of a fit

`fit_LMM` on CUDA, float32. The sample statistics scale with $N$, and the rest, the shape solve, is constant at about 85 ms:

<!-- table:factor_fit_composition -->
| N | Sort | L-moments, including sort | fit_LMM, total |
|---|---|---|---|
| 1e+04 | 114 µs | 1.17 ms | 92.1 ms |
| 1e+06 | 701 µs | 2.87 ms | 83.9 ms |
| 1e+07 | 6.16 ms | 25.1 ms | 119 ms |
<!-- /table -->

So up to about $10^7$ points, more data costs almost nothing, which is the real benefit for L-moment fitting. On the CPU the sort alone takes about 1 s at $10^7$, against 6 ms on the GPU.

`fit_PM` has the same structure with a larger constant, about 170 to 350 ms, because its residuals are costlier and it has more local minima to refine. Without multistart, a single NumPy solve is faster at small $N$. Pass `initial_guess` to skip the grid when a good start is known.

`fit_ML` cost is proportional to `steps`:

<!-- table:factor_ml_steps -->
| steps | fit_ML, N = 1e5, float32 |
|---|---|
| 30 | 2.0 s |
| 60 | 3.1 s |
| 300 | 9.6 s |
<!-- /table -->

Each step needs one CDF solve, warm-started from the previous one, and one backward pass, and the many small solver iterations are dominated by launch cost. So **`fit_ML` gains little from this GPU in this range**: at $10^5$ points the CPU in float32 takes 2.6 s, CUDA in float32 3.1 s, and CUDA in float64 1.8 s. Float32 needs about 1.7 times as many solver calls per step as float64, on very small arrays of elements near rounding level, so on this GPU float32 is not faster than float64 until about $10^6$ points. The fitted parameters agree between the two precisions. The CPU was not run at $10^6$.

## What the optimizations did

These were found by profiling the first log-space implementation. Comparison on the same machine in one process, FMKL(0.2, 0.3), float32 unless stated:

| Operation | v0.3.0 | Log space, before optimization | After |
|---|---|---|---|
| `Q`, $10^7$ | 8.0 ms | 20 ms | 6.1 ms |
| `PDF_Q`, $10^7$ | 3.3 ms | 7.9 ms | 7.0 ms |
| `rand`, $10^7$ | 7.5 ms | 24 ms | 10 ms |
| `CDF_num`, $10^5$ | 32 ms | 40 ms | 7.4 ms |
| `CDF_num`, $3\cdot10^6$, float64 | 422 ms | 459 ms | 61 ms |
| `fit_LMM`, $10^5$ | 344 ms | 132 ms | 90 ms |
| `fit_PM`, $10^5$ | 843 ms | 578 ms | 217 ms |
| `fit_ML`, $10^5$, 60 steps | 2.8 s | 6.1 s | 3.1 s |

These are best of 3 (one run for `fit_ML`), measured with a one-off script against the v0.3.0 source and a snapshot of the code before optimization, so they are not reproduced by `bench_gpu.py`. `PDF_Q` and `rand` remain 2.1x and 1.4x slower than v0.3.0, the price of their handling of zero coefficients and of tails. Also, v0.3.0's fast numbers came with wrong tails and unconverged fits, so these are not like-for-like comparisons of correct results.

| Change | Mechanism | Measured effect |
|---|---|---|
| Skip the series branch of the Box-Cox term unless $\partial_\lambda$ is needed | `expm1` alone is accurate at all $\lambda$; the branch only protects the gradient | 6.5x on that function, 3x on `Q` |
| Check $\lambda = 0$ once, in Python, not per element | Removes `where` from the common path | Part of the above |
| Interpolation table for the solver's start | Newton needs 2 steps, not 5 or more | 3x on `CDF_num` |
| Retire finished elements | Stragglers no longer cost full-sized iterations | Large when few remain |
| Tiny Newton steps exempt from the halving rule | Rounding noise was triggering a bisection of a wide bracket, about 24 extra iterations for a few elements | 8x on float32 `CDF_num` at $10^5$ |
| Floor of $32\epsilon$ on `xtol` | Stops iterating inside rounding noise | 2x on float32 `fit_ML` |
| Interior path of the log density | Skips the endpoint handling inside the solver | Part of the `CDF_num` gain |
| Shape solves on the CPU, with exact Jacobians in place of Adam | Launch cost dominates on 16 elements | 2.6x on `fit_LMM`, from v0.3.0 |
| Stop Levenberg-Marquardt when no start progresses, and the fast Box-Cox path | Stuck starts used to run all 40 iterations | Together 2.7x on `fit_PM` |

## Reproducing

```bash
uv run python benchmarks/bench_gpu.py           # about 10 minutes; writes benchmarks/results/benchmarks.json
uv run python benchmarks/compare_analytic.py    # about 5 minutes; writes benchmarks/results/analytic.json
uv run python benchmarks/make_tables.py         # puts both into docs/validation.md and docs/benchmarks.md
uv run pytest -m gpu_speed                      # checks that the GPU still beats the CPU on large batches
```
