# Validation

What supports the claim that `gldpy_gpu` computes the right numbers, with the protocol and measured results. The methods are described in
[numerics.md](numerics.md).

## 1. Three independent references

No single reference is enough, because the original package is wrong in places and a closed form exists only for special cases.

| Reference | Used for | Where |
|---|---|---|
| **Closed forms**: distributions the GLD contains exactly, and published values | Tails, moments, fit recovery. Independent of any GLD implementation | `tests/test_gpu_tails.py`, `tests/test_analytic.py` |
| **`gldpy` 0.2 from PyPI**, frozen | Everything outside its known defects. Evidence for each defect | `tests/test_upstream_bugs.py` |
| **Corrected NumPy class** against the torch class | Parity between backends, on CPU and CUDA, in float32 and float64 | `tests/test_gpu_core.py`, `tests/test_gpu_fit*.py` |

Run the whole suite with `uv run pytest -m "not gpu_speed"`. The default `uv run pytest` leaves out the slow tests.

## 2. Fit quality against known distributions

**Protocol** (`benchmarks/compare_analytic.py`). The input is the exact quantile grid $x_k = Q_\text{true}((k + \frac12) / n)$, with $n = 20\,000$, or $n = 5\,000$ for maximum likelihood. This grid has **no sampling noise**: all that remains is the error of the estimator, of the implementation, and of the GLD family as an approximation.

The metric is the Kolmogorov distance on 1999 probabilities in $(0.0005, 0.9995)$, evaluated with the float64 torch CDF:

$$\text{KS} = \max_p |F_\text{fit}(Q_\text{true}(p)) - p|.$$

Because the grid has $n$ points, even a perfect fit has KS of order $1/(2n) = 2.5\cdot10^{-5}$. That value is the floor for the exact members below.

**Backends.** *orig.* is PyPI `gldpy` 0.2; *NumPy* is the corrected `gldpy_gpu.GLD`; both start from $(0.1, 0.1)$, or $0.3$ for VSL's percentile fit. *f64* and *f32* are the torch backend in each precision, on CUDA, with no start given. "fails" means invalid parameters came back.

### RS

<!-- table:ks_RS -->
| Distribution | LMM orig. | LMM NumPy | LMM f64 | LMM f32 | PM orig. | PM NumPy | PM f64 | PM f32 | MM NumPy | ML f64 | ML f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 5.6e-02 | 2.5e-05 | 2.5e-05 | 2.4e-05 | 1.5e-09 | 1.7e-04 | 1.7e-04 |
| logistic | 3.7e-06 | 3.7e-06 | 2.6e-06 | 2.4e-06 | 2.0e-05 | 2.0e-05 | 2.0e-05 | 3.0e-05 | 2.4e-04 | 3.1e-04 | 3.1e-04 |
| exponential | fails | fails | 6.5e-05 | 1.1e-03 | fails | fails | 2.9e-05 | 2.8e-04 | fails | 1.8e-01 | 1.6e-01 |
| normal | 3.9e-04 | 3.9e-04 | 3.9e-04 | 3.9e-04 | 3.5e-03 | 5.1e-04 | 5.1e-04 | 5.1e-04 | 1.0e-03 | 1.4e-03 | 1.4e-03 |
| student_t5 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 9.7e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 8.7e-02 | 2.0e-03 | 2.0e-03 |
| gamma_2 | 8.2e-03 | 8.2e-03 | 8.2e-03 | 8.2e-03 | 1.4e-02 | 1.4e-02 | 1.4e-02 | 1.4e-02 | 1.3e-02 | 1.7e-02 | 1.7e-02 |
| lognormal_0.25 | 5.6e-03 | 5.6e-03 | 5.6e-03 | 5.6e-03 | 5.5e-03 | 5.5e-03 | 5.5e-03 | 5.5e-03 | 1.0e-02 | 8.1e-03 | 8.1e-03 |
| beta_2_2 | 5.9e-03 | 5.9e-03 | 4.4e-04 | 4.4e-04 | 2.4e-02 | 1.0e-03 | 1.0e-03 | 1.0e-03 | 3.8e-04 | 5.2e-04 | 5.2e-04 |
| beta_2_5 | 7.7e-03 | 7.7e-03 | 7.7e-03 | 7.7e-03 | 2.2e-02 | 1.3e-02 | 1.3e-02 | 1.3e-02 | 1.0e-02 | 2.7e-02 | 2.7e-02 |
<!-- /table -->

### FMKL

<!-- table:ks_FMKL -->
| Distribution | LMM orig. | LMM NumPy | LMM f64 | LMM f32 | PM orig. | PM NumPy | PM f64 | PM f32 | MM NumPy | ML f64 | ML f32 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.6e-09 | 9.3e-04 | 4.3e-04 |
| logistic | 2.6e-06 | 2.6e-06 | 2.6e-06 | 2.6e-06 | 2.0e-05 | 2.0e-05 | 2.0e-05 | 2.0e-05 | 2.4e-04 | 3.9e-04 | 4.0e-04 |
| exponential | 1.3e-02 | 1.3e-02 | 5.3e-04 | 5.3e-04 | 1.3e-02 | 1.3e-02 | 2.4e-03 | 2.4e-03 | 2.5e-02 | 4.2e-03 | 4.2e-03 |
| normal | 3.9e-04 | 3.9e-04 | 3.9e-04 | 3.9e-04 | 5.1e-04 | 5.1e-04 | 5.1e-04 | 5.1e-04 | 1.0e-03 | 1.4e-03 | 1.4e-03 |
| student_t5 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 7.7e-03 | 2.4e-03 | 2.4e-03 |
| gamma_2 | 8.4e-03 | 8.4e-03 | 8.4e-03 | 8.4e-03 | 1.2e-02 | 1.2e-02 | 1.2e-02 | 1.2e-02 | 3.0e-02 | 2.1e-02 | 1.9e-02 |
| lognormal_0.25 | 3.2e-03 | 3.2e-03 | 3.2e-03 | 3.2e-03 | 4.7e-03 | 4.7e-03 | 4.7e-03 | 4.7e-03 | 5.2e-03 | 4.0e-02 | 4.1e-02 |
| beta_2_2 | 4.4e-04 | 4.4e-04 | 4.4e-04 | 4.4e-04 | 1.0e-03 | 1.0e-03 | 1.0e-03 | 1.0e-03 | 3.8e-04 | 5.0e-04 | 5.0e-04 |
| beta_2_5 | 3.2e-03 | 3.2e-03 | 3.2e-03 | 3.2e-03 | 5.1e-03 | 5.1e-03 | 5.1e-03 | 5.1e-03 | 5.9e-03 | 8.0e-03 | 7.5e-03 |
<!-- /table -->

### VSL

<!-- table:ks_VSL -->
| Distribution | LMM orig. | LMM NumPy | LMM f64 | LMM f32 | PM orig. | PM NumPy | PM f64 | PM f32 | ML f64 | ML f32 |
|---|---|---|---|---|---|---|---|---|---|---|
| uniform | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 2.5e-05 | 9.3e-04 | 1.4e-03 |
| logistic | 2.6e-06 | 2.6e-06 | 2.6e-06 | 2.6e-06 | 2.0e-05 | 2.0e-05 | 2.0e-05 | 2.0e-05 | 2.6e-04 | 2.8e-04 |
| exponential | 6.5e-05 | 6.5e-05 | 6.5e-05 | 6.4e-05 | 2.2e-05 | 2.2e-05 | 2.2e-05 | 2.1e-05 | 4.0e-04 | 4.2e-04 |
| normal | 3.9e-04 | 3.9e-04 | 3.9e-04 | 3.9e-04 | 5.1e-04 | 5.1e-04 | 5.1e-04 | 5.1e-04 | 1.3e-03 | 1.3e-03 |
| student_t5 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 1.6e-03 | 2.4e-03 | 2.4e-03 |
| gamma_2 | 6.7e-03 | 6.7e-03 | 6.7e-03 | 6.7e-03 | 1.1e-02 | 1.1e-02 | 1.1e-02 | 1.1e-02 | 1.3e-02 | 1.3e-02 |
| lognormal_0.25 | 3.7e-03 | 3.7e-03 | 3.7e-03 | 3.7e-03 | 3.5e-03 | 3.5e-03 | 3.5e-03 | 3.5e-03 | 5.0e-03 | 5.1e-03 |
| beta_2_2 | 4.4e-04 | 4.4e-04 | 4.4e-04 | 4.4e-04 | 1.0e-03 | 1.0e-03 | 1.0e-03 | 1.0e-03 | 2.9e-03 | 3.1e-03 |
| beta_2_5 | 3.4e-03 | 3.4e-03 | 3.4e-03 | 3.4e-03 | 6.0e-03 | 6.0e-03 | 6.0e-03 | 6.0e-03 | 3.5e-03 | 3.5e-03 |
<!-- /table -->

### Reading the tables

- **Exact members reach the floor.** Uniform and logistic in all three types, and exponential in VSL, have KS near $2.5\cdot10^{-5}$ with L-moments and percentiles. Moment matching on the uniform goes lower, because its grid moments are nearly exact.
- **Non-members show the family's own limit.** All implementations reach about the same KS for the normal ($4\cdot10^{-4}$), Student t(5) ($1.6\cdot10^{-3}$) and the skewed gamma(2) ($7$ to $8\cdot10^{-3}$). This limit belongs to the GLD, so more computation will not lower it. The three parameterizations differ little.
- **U01 is visible** in the RS percentile columns: *orig.* against *NumPy* on the uniform, normal, Student t and beta.
- **Multistart matters.** RS on Beta(2, 2): the local fit from $(0.1, 0.1)$ gives $5.9\cdot10^{-3}$, and the torch fit, which explores all roots, $4.4\cdot10^{-4}$. FMKL on the exponential: $1.3\cdot10^{-2}$ against $5.3\cdot10^{-4}$. RS on the exponential: the local fit returns invalid parameters.
- **Maximum likelihood is not a KS minimizer.** On non-members its KS is often higher than that of L-moments, since it trades error in the center for the likelihood's sensitivity to the tails and support. See section 5 for limits.

**Published value.** Moment matching to the normal gives RS$(0, 0.1979, 0.1353, 0.1353)$ at $n = 20\,000$, in both the original and the corrected class. With $n = 500\,000$ the test `test_ramberg_schmeiser_normal_approximation` reproduces the published $(0, 0.1975, 0.1349, 0.1349)$ of Ramberg & Schmeiser (1974) within $10^{-3}$.

## 3. Influence of the precision

**Fits.** The difference between float32 and float64 runs of the same fit, over all 27 combinations of 9 distributions and 3 types:

<!-- table:precision -->
| Method | Cases | Median |ΔKS| | Largest |ΔKS| |
|---|---|---|---|
| LMM | 27 | 2.1e-08 | 1.1e-03 (RS, exponential) |
| PM | 27 | 1.9e-08 | 2.5e-04 (RS, exponential) |
| ML | 27 | 5.8e-06 | 2.1e-02 (RS, exponential) |
<!-- /table -->

The median difference is far below any practical concern, and the fit in float32 is as good as in float64. The largest differences occur for RS on the exponential, which is not a member of RS but a limiting shape with $\lambda_3, \lambda_4 \to$ extreme values.

**Quantile function.** On $y \in [10^{-3}, 1 - 10^{-3}]$, the largest relative difference from the original across 7 parameter sets is $9.4\cdot10^{-5}$ in float32 and $3.8\cdot10^{-13}$ in float64, the largest where $Q$ crosses 0, since a relative error is ill-defined at a zero. For the density it is $1.6\cdot10^{-5}$ and $1.0\cdot10^{-15}$.

**Data with a large offset.** float32 stores about 7 significant digits. Data around 5000 with unit spread have a rounding step of $5\cdot10^{-4}$. The computations cannot recover what storage has lost, so such data should be float64 in the first place. Calling `fit_LMM(d32, dtype=torch.float64)` runs the computation in float64, but on the already rounded values.

**Recommendation.** float32 suffices for fitting, for sampling, and for tail probabilities on infinite tails. Use float64 for bounded tails below about $10^{-6}$, for a large offset relative to the spread, and whenever the fourth significant digit of a quantile matters.

## 4. Tail accuracy

**Protocol.** For a tail probability $s$, compute $x = Q(s)$ and recover $s' = F(x)$, then likewise $x = Q(1-s)$ through `ISF` and recover $s'$ through `SF_num`. The table has the worst relative error $|s'/s - 1|$ of the two tails. The original has no upper-tail method, so only its lower tail can be compared, and it was not run below $10^{-12}$. A test against closed forms, independent of the round trip, is in `tests/test_gpu_tails.py`: logistic, exponential and Pareto, down to $10^{-300}$ in float64.

<!-- table:tails -->
| Case | s | Original (lower only) | NumPy | torch f64 | torch f32 |
|---|---|---|---|---|---|
| FMKL infinite tails (-0.2, -0.1) | 1e-04 | 3.0e-03 | 2.5e-12 | 1.6e-15 | 1.0e-06 |
| FMKL infinite tails (-0.2, -0.1) | 1e-08 | 6.0e+02 | 2.1e-12 | 1.0e-15 | 3.6e-07 |
| FMKL infinite tails (-0.2, -0.1) | 1e-12 | 6.0e+06 | 2.1e-09 | 4.2e-15 | 2.4e-06 |
| FMKL infinite tails (-0.2, -0.1) | 1e-30 | not run | 2.1e-10 | 1.6e-14 | 9.4e-08 |
| FMKL infinite tails (-0.2, -0.1) | 1e-100 | not run | 2.3e-11 | 3.9e-14 | below floor |
| RS infinite tails (-0.1, -0.15) | 1e-04 | 1.8e-04 | 8.0e-11 | 0 | 0 |
| RS infinite tails (-0.1, -0.15) | 1e-08 | 6.0e+02 | 4.1e-11 | 1.0e-15 | 3.6e-07 |
| RS infinite tails (-0.1, -0.15) | 1e-12 | 6.0e+06 | 3.7e-10 | 4.2e-15 | 2.4e-06 |
| RS infinite tails (-0.1, -0.15) | 1e-30 | not run | 4.4e-11 | 1.6e-14 | 9.4e-08 |
| RS infinite tails (-0.1, -0.15) | 1e-100 | not run | 2.3e-11 | 3.9e-14 | below floor |
| VSL infinite tails (0.4, -0.2) | 1e-04 | 3.0e-03 | 7.9e-10 | 1.6e-15 | 0 |
| VSL infinite tails (0.4, -0.2) | 1e-08 | 6.0e+02 | 2.4e-09 | 4.4e-15 | 3.6e-07 |
| VSL infinite tails (0.4, -0.2) | 1e-12 | 6.0e+06 | 2.3e-11 | 7.8e-16 | 2.4e-06 |
| VSL infinite tails (0.4, -0.2) | 1e-30 | not run | 7.8e-10 | 2.2e-15 | 9.4e-08 |
| VSL infinite tails (0.4, -0.2) | 1e-100 | not run | 1.0e-09 | 3.9e-14 | below floor |
| FMKL bounded (0.2, 0.3) | 1e-04 | 4.7e-03 | 1.2e-11 | 0 | 1.0e-06 |
| FMKL bounded (0.2, 0.3) | 1e-08 | 6.0e+02 | 4.2e-11 | 1.1e-13 | 1.3e-05 |
| FMKL bounded (0.2, 0.3) | 1e-12 | 6.0e+06 | 2.0e-09 | 2.0e-12 | 1.4e-04 |
| FMKL bounded (0.2, 0.3) | 1e-30 | not run | 1.1e-07 | 4.3e-08 | 1.2e+00 |
| FMKL bounded (0.2, 0.3) | 1e-100 | not run | 1.0e+00 | 1.0e+00 | below floor |
| RS bounded (0.1, 0.15) | 1e-04 | 8.9e-03 | 2.3e-12 | 1.9e-15 | 1.0e-06 |
| RS bounded (0.1, 0.15) | 1e-08 | 6.0e+02 | 1.3e-10 | 8.1e-15 | 3.6e-06 |
| RS bounded (0.1, 0.15) | 1e-12 | 6.0e+06 | 1.9e-13 | 7.9e-15 | 8.0e-06 |
| RS bounded (0.1, 0.15) | 1e-30 | not run | 3.7e-11 | 1.8e-11 | 1.6e-02 |
| RS bounded (0.1, 0.15) | 1e-100 | not run | 3.1e-01 | 1.0e+00 | below floor |
| VSL bounded (0.4, 0.3) | 1e-04 | 8.6e-04 | 9.3e-10 | 5.6e-15 | 4.8e-06 |
| VSL bounded (0.4, 0.3) | 1e-08 | 6.0e+02 | 2.2e-09 | 1.8e-13 | 3.7e-05 |
| VSL bounded (0.4, 0.3) | 1e-12 | 6.0e+06 | 5.4e-13 | 1.9e-12 | 3.0e-04 |
| VSL bounded (0.4, 0.3) | 1e-30 | not run | 3.6e-08 | 1.9e-07 | 1.0e+00 |
| VSL bounded (0.4, 0.3) | 1e-100 | not run | 1.0e+00 | 1.0e+00 | below floor |
<!-- /table -->

**Reading the table.**

- **Infinite tails** keep relative accuracy down to the floor: about $10^{-14}$ in float64 and $10^{-6}$ in float32. "below floor" means $s$ itself cannot be represented in float32, below about $10^{-38}$.
- **Bounded tails** degrade as $s$ decreases, and reach an error of order 1 once $c\,s^\lambda$ falls below the rounding of $x$. This is a limit of the input $x$, not of the solver ([numerics.md](numerics.md#3-the-cdf-root-finding-on-the-log-odds)). In float64 it arrives around $s \sim 10^{-100}$ for $\lambda \approx 0.2$; in float32 around $10^{-30}$.
- **The original's error** saturates at $y \approx 6\cdot10^{-6}$, so it equals $6\cdot10^{-6}/s$ for smaller $s$ whatever the distribution. That is why those rows agree to two digits across the parameterizations.
- **The NumPy class** has an error of about $10^{-9}$, set by its default `xtol = 1e-8`.

**Sampling.** The tests `test_rand_float32_stays_finite_and_reaches_past_the_float32_grid` and `test_rand_upper_tail_frequency_is_right` cover the sampler: no infinite values in $4\cdot10^7$ float32 draws, draws beyond the $Q(1 - 2^{-24})$ barrier of a plain float32 uniform, and a frequency of $s = 10^{-4}$ within 5 standard deviations.

## 5. Known limitations

These are measured and not hidden. Each is also a reason to check results before relying on them in production.

1. **Maximum likelihood is a local method, with a default of 300 Adam steps that may not converge.** On the deterministic grid, against the original's Nelder-Mead (mean NLL above the entropy, FMKL, $n = 5000$): lognormal(0.25) $+0.0079$ against $+0.0042$; Beta(2, 5) $+0.00084$ against $+0.00096$; gamma(2) $+0.00221$ against $+0.00220$. On the lognormal, 600 steps reach about $+0.0009$. Raise `steps` where it matters, and compare with the L-moment fit. A quasi-Newton polish is on the roadmap.
2. **RS on the exponential with maximum likelihood** stops at a poor local optimum, with KS above $0.1$ where L-moments reach $6.5\cdot10^{-5}$. The exponential is not in the RS family, except as a limit.
3. **Parameters are not always identified.** FMKL(1, 1) and FMKL(2, 2) are the same uniform distribution, and the RS logistic limit has $\lambda \to 0$ with $\lambda_2 \to 0$. The distribution is then well determined, the parameter values are not. Compare distributions, not parameter vectors.
4. **Goodness-of-fit p-values are not calibrated** when the parameters were estimated from the same data ([U07](upstream-bugs.md#u07-gof_tests-raises-on-data-outside-the-support)).
5. **The percentile estimators use only 5 sample quantiles**, so they have a larger variance than L-moments. The protocol here, without sampling noise, does not show this.
6. **`fit_MPS`, `fit_starship`, `fit_MM` and `fit_curve` exist only in the NumPy class.** Tests cover `fit_MPS`'s support and `fit_MM` parity. `fit_starship`, `fit_curve` and the plotting methods are only as tested as they were upstream, apart from the seed and statistics corrections.
7. **Tested platforms.** The full suite ran on Linux, with Python 3.11, PyTorch 2.14.1+cu126 and a GTX 1070 (CUDA 12.6), on both the CPU and CUDA. The torch-free part ran from a clean environment on Python 3.11 and 3.12. Python 3.9 and 3.10 are allowed by the package metadata but have not been run: the CI workflow, which adds 3.9, has not run yet. No other GPU, operating system or PyTorch version has been tested.
