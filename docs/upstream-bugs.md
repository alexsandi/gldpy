# Defects in the original gldpy 0.2, and their correction

**Reference version.** `gldpy` 0.2 from PyPI (released 2020-10-27). It is frozen, so this analysis will not go
out of date. `gldpy_gpu/_gld_module.py` started as a copy of it: the two files were identical up to line
endings when this analysis began.

**Evidence.** Every defect below has a pair of tests in `tests/test_upstream_bugs.py`: `test_Unn_original_*` runs
against the real PyPI package and shows the misbehavior, and `test_Unn_fixed_*` shows `gldpy_gpu.GLD` does not
have it. If a statement here stops being true, a test fails. The same file checks that the corrections changed nothing
else: `Q`, `PDF_Q`, `supp`, the moments, the seeded `rand` stream and the fits from a common start all agree
with the original, within the tolerances stated there.

**Notation.** $Q$ is the quantile function, $y$ the lower tail probability, $s = 1 - y$. Shape parameters are
$\lambda_3, \lambda_4$ (RS, FMKL) and $\lambda$ (VSL, whose $\delta$ is the third parameter). KS is the
Kolmogorov distance $\sup_p |F_\text{fit}(Q_\text{true}(p)) - p|$, with the protocol of
[validation.md](validation.md).

## Summary

| ID | Method | Defect | Consequence | Severity |
|---|---|---|---|---|
| U01 | `fit_PM` (RS) | Missing `abs` in the objective | Equations not solved; KS up to 2000x worse | High: wrong results, silently |
| U02 | `CDF_num`, `PDF_num` | Absolute tolerance on $y$ | Tail probabilities wrong by factors of 5 to $6\cdot10^6$ | High: wrong results, silently |
| U03 | `mean`, `var`, `skewness`, `kurtosis` | Division by $\lambda^k$ | NaN at $\lambda=0$; kurtosis $2.4\cdot10^4$ for a true 4.2 | High |
| U04 | `fit_LMM`, `fit_PM`, `fit_MM` (FMKL) | Same singularity in the objective | NaN parameters from a start at $\lambda=0$ | Medium |
| U05 | `rand`, `fit_curve` | `if random_state:`; global reseeding | Seed 0 ignored; caller's random stream altered | Medium |
| U06 | `correct_supp` | Writes into its argument | Input changed; integer parameters truncated | Medium |
| U07 | `GoF_tests` | Histogram drops outside points | `ValueError` from SciPy | Medium |
| U08 | `PDF_num` | Flattened result, unflattened mask | `IndexError` on 2-D input | Low |
| U09 | `Q`, `PDF_Q` | $0\cdot\infty$ at the endpoints | NaN where the limit exists | Low |
| U10 | `grid_search` (RS) | Sign of $\lambda_2$ forgotten | Invalid parameters returned | Medium |
| U11 | `fit_MPS` | Two end spacings left out | Observed data outside the fitted support | High: wrong results, silently |
| U12 | `fit_MPS` | Tie jitter from the global generator | Not reproducible | Low |
| U13 | Class docstring | Wrong sign in two density formulas | Misleading documentation | Low |
| U14 | `Q` | Cancellation as $\lambda \to 0$ | Lost digits in the quantile | Low in float64 |

## Details

### U01. RS `fit_PM` does not solve its equations

The objective is meant to be $\max(|\hat\rho_4 - a_4|, |\hat\rho_3 - a_3|)$. The code reads

```python
return np.max([( (0.75**l3 - 0.25**l4 - 0.25**l3 +0.75**l4)/((1-u)**l3-u**l4 - u**l3+(1-u)**l4) - a4),
               np.abs((0.5**l3 - 0.5**l4 - u**l3 +(1-u)**l4)/((1-u)**l3 - u**l4 - 0.5**l3 +0.5**l4) - a3)])
```

The first term has no `abs`, so any negative error in $\rho_4$ counts as zero: the optimizer stops as soon as $\rho_3$ is
matched and $\rho_4$ is on the low side. The FMKL branch has the `abs`, which shows the intent.

*Measured residual in $\rho_4$ left behind* (5000 draws, seed 11, start at the true shape): $-1.5\cdot10^{-4}$ for RS(1, 2),
$-2.4\cdot10^{-3}$ for RS(0.1, 0.15), $-4.3\cdot10^{-3}$ for RS(-0.1, -0.15). The residual in $\rho_3$ is about $10^{-13}$.

*Effect on the fit, KS on the exact quantile grid of each distribution (RS, `fit_PM`):*

| Distribution | Original | Corrected |
|---|---|---|
| Uniform | $5.6\cdot10^{-2}$ | $2.5\cdot10^{-5}$ |
| Beta(2, 2) | $2.4\cdot10^{-2}$ | $1.0\cdot10^{-3}$ |
| Student t(5) | $9.7\cdot10^{-3}$ | $1.6\cdot10^{-3}$ |
| Normal | $3.5\cdot10^{-3}$ | $5.1\cdot10^{-4}$ |

*Correction.* The missing `abs`.

### U02. `CDF_num` loses all accuracy in the tails

`CDF_num` minimizes $(Q(y) - x)^2$ over $y \in [0, 1]$ with `scipy.optimize.fminbound` and an absolute tolerance
`xtol = 1e-5` on $y$. Two facts combine. Near a tail probability below `xtol` the result is
meaningless by construction, and `fminbound` never evaluates the endpoints, so it does not return $0$ or $1$ either. In
practice the result stays near $6\cdot10^{-6}$ for any point further out. `PDF_num` inherits the error, because it evaluates
the density at that $y$, and so do `GoF_tests`, `fit_ML`, `fit_MPS` and `fit_starship`.

*Measured, FMKL(0, 1, -0.2, -0.1), relative error of the recovered tail probability $x = Q(s)$:*

| $s$ | $10^{-2}$ | $10^{-4}$ | $10^{-6}$ | $10^{-8}$ | $10^{-10}$ | $10^{-12}$ |
|---|---|---|---|---|---|---|
| Lower tail | $1.5\cdot10^{-4}$ | $3.0\cdot10^{-3}$ | $5.0$ | $6.0\cdot10^{2}$ | $6.0\cdot10^{4}$ | $6.0\cdot10^{6}$ |

*Correction.* A root find on $t = \operatorname{logit}(y)$ instead of a minimization on $y$, and evaluation of $Q$ from
$\log y$ and $\log s$; see [numerics.md](numerics.md). New methods `SF_num`, `ISF` and `log_PDF_num` expose the upper tail,
which $1 - F$ and $Q(1 - s)$ cannot represent below machine epsilon. The argument `xtol` keeps its name but now means
the relative accuracy of the smaller tail probability, and its default changed from `1e-5` to `1e-8`.

### U03. Moments are singular at $\lambda = 0$

For FMKL and VSL, `__sum_Ez` expands $E[Z^k]$ into a sum of Beta functions with a prefactor $1/(\lambda_3\lambda_4)^k$ or
$1/\lambda^k$. The GLD is regular at $\lambda = 0$, where it has logarithmic terms, but this expression is $0/0$ there,
and for small $\lambda$ the sum loses about $\epsilon / \lambda^{k}$ of relative accuracy.

*Measured.* At $\lambda = 0$ all four moments are NaN, including for the logistic, FMKL(0, 1, 0, 0), and the
exponential, VSL(0, 1, 1, 0). Kurtosis of FMKL($\lambda$, $\lambda$), whose limit is the logistic's 4.2:

| $\lambda$ | $0.05$ and above | $10^{-3}$ | $10^{-5}$ |
|---|---|---|---|
| Original | correct | $4.18493$ (should be $4.18561$) | $2.4\cdot10^{4}$ (should be $4.1999$) |

*Correction.* Below $|\lambda| = 0.02$, integrate $\int_0^1 Q(y)^k\,dy$ numerically, on $(0, \tfrac12)$ and $(\tfrac12, 1)$
separately because of the integrable logarithmic singularity at each end. The threshold comes from
$(\epsilon/10^{-8})^{1/4} \approx 0.012$ for $k = 4$, rounded up. Tests compare with the closed-form moments of the logistic and the
exponential, and with direct numerical moments on both sides of the switch.

### U04. Fits started at $\lambda = 0$ return NaN

The FMKL equations of `fit_LMM` and `fit_PM` divide by $\lambda$, and `fit_MM` inherits U03. Nelder-Mead sees NaN at the start, stays
there and returns `[nan, nan, 0, 0]` with no error. A start at zero is the natural choice for logistic-like data.

*Correction.* The singularity is removable in closed form. For example, for the L-moment terms,

$$\frac{1}{\lambda}\left(-\frac{1}{1+\lambda} + \frac{2}{2+\lambda}\right) = \frac{1}{(1+\lambda)(2+\lambda)},\qquad
\frac{1}{\lambda}\left(\frac{1}{1+\lambda} - \frac{6}{2+\lambda} + \frac{6}{3+\lambda}\right) = \frac{\lambda - 1}{(1+\lambda)(2+\lambda)(3+\lambda)},$$

$$\frac{1}{\lambda}\left(-\frac{1}{1+\lambda} + \frac{12}{2+\lambda} - \frac{30}{3+\lambda} + \frac{20}{4+\lambda}\right) = \frac{(\lambda - 1)(\lambda - 2)}{(1+\lambda)(2+\lambda)(3+\lambda)(4+\lambda)},$$

and $(c^\lambda - 1)/\lambda$ is computed as `expm1(λ log c) / λ` in the percentile equations.

### U05. `random_state`

`if random_state:` treats the seed 0 as no seed. Also, `np.random.seed` resets the process-wide generator, which silently changes the
random stream of any code that runs afterwards. `fit_curve` has the same two problems.

*Correction.* `random_state is None` is the test, and a private `RandomState` draws the numbers. It produces the same stream as the
original for a given nonzero seed, which `test_parity_rand_stream_with_a_seed` verifies, so existing seeded results do not change.

### U06. `correct_supp` changes its input

`param[:2] = x` writes into the caller's array. If the array has an integer dtype, the assignment also truncates: starting from
`np.array([0, 1, 1, 2])` with data on $[-9, 9]$, the result is `[0, 0, 1, 2]`, where $\lambda_2 = 0$ is not a distribution at all.

*Correction.* A new float array is returned. $Q$ is affine in location and scale, so the two end conditions $Q(\varepsilon) = A$ and
$Q(1-\varepsilon) = B$ have an exact solution and the iterative `fmin` is no longer needed.

### U07. `GoF_tests` raises on data outside the support

The chi-square test histograms the data on bins with edges $Q(i/k)$. A point outside a finite support falls in no bin, so the
observed counts no longer sum to the expected counts, and `scipy.stats.chisquare` raises `ValueError`. This can happen right after a
fit, because the moment-based methods do not constrain the support.

*Correction.* Points are clipped to the support, so they count in the end bins and lower the p-value as they should.

*Statistical caveat, unchanged.* When `param` was estimated from the same data, neither p-value is calibrated: the chi-square test
uses $k - 1$ degrees of freedom, not $k - 1 - 4$, and the Kolmogorov-Smirnov distribution assumes a fully specified $F$. Both tests
are too lenient in that situation. The method cannot know how `param` was obtained, so it remains the caller's responsibility.

### U08. `PDF_num` on 2-D input

`CDF_num` flattens its input, and `PDF_num` then indexes the flat result with a mask of the original shape: `IndexError`.

*Correction.* `CDF_num`, `SF_num`, `PDF_num` and `log_PDF_num` keep the input shape. A scalar still gives an array of length 1, as before.

### U09. NaN at the exact endpoints

The formulas multiply a zero coefficient by an infinite power: VSL with $\delta = 1$, $\lambda < 0$ at $y = 0$ has
$(1 - \delta)(0^\lambda - 1)/\lambda = 0 \cdot \infty$, and RS with $\lambda_3 = 0$ at $y = 0$ has $\lambda_3 y^{\lambda_3 - 1}$. Both are
terms that vanish identically, so the correct value exists: for instance $Q(0) = 0$ for the Pareto case VSL(0, 1, 1, -1).

*Correction.* A term with a zero coefficient is left out.

### U10. RS `grid_search` returns invalid parameters

For RS, shape pairs with negative values need $\lambda_2 < 0$. The grid loop handles that (`param[1] = -1`), but the final refinement
restarts from $\lambda_2 = 1$. The objective is then infinite at the start, `fmin` does not move, and an invalid parameter
vector comes back, e.g. `[0.24, 1, -0.25, -0.25]` on data from RS(0, -0.2, -0.1, -0.15).

*Correction.* The final refinement applies the same sign rule.

### U11. `fit_MPS` leaves out the end spacings

Maximum product of spacings maximizes the geometric mean of the $n + 1$ spacings

$$D_i = F(x_{(i)}) - F(x_{(i-1)}),\quad i = 1, \dots, n + 1,\qquad F(x_{(0)}) = 0,\ F(x_{(n+1)}) = 1,$$

which is what makes it usable where likelihood is not (definition as in Chalabi, Scott & Wuertz 2012; the method is from Cheng & Amin 1983 and Ranneby 1984). The original uses `np.diff(CDF_num(data))`, which has only the $n - 1$ inner spacings.
Without $D_1$ and $D_{n+1}$, nothing penalizes a point outside the support: its CDF is 0 or 1, and its inner spacing stays positive.

*Measured.* FMKL(0, 1, 0.5, 0.5), $n = 40$: in 5 of 6 seeds, the fitted support leaves out at least one observed point, so the fitted distribution says
that the data are impossible.

*Correction.* All $n + 1$ spacings, with the last one from `SF_num`, so that it does not lose accuracy.

### U12. `fit_MPS` with ties is not reproducible

Tied observations, for which a spacing would be 0, receive Gaussian jitter from the global generator, so two calls on the same data give different results.

*Correction.* The jitter comes from a private generator with a fixed seed.

### U13. Wrong signs in the class docstring

The docstring gives $f(y) = \lambda_2 / (\lambda_3 y^{\lambda_3-1} - \lambda_4 (1-y)^{\lambda_4-1})$ for RS and the same minus sign for FMKL. The code correctly has a plus,
since $Q'(y)$ is a sum: the derivative of $-(1-y)^{\lambda_4}$ is $+\lambda_4 (1-y)^{\lambda_4 - 1}$.

*Correction.* The documentation now has the plus. Also, `fit_starship` printed its Anderson-Darling statistic with the label "KS-statistic", which is corrected too.

### U14. Cancellation in `Q` near $\lambda = 0$

$(y^\lambda - 1)/\lambda$ subtracts two numbers close to 1. In float64 the relative error reaches about $8\cdot10^{-6}$ at $\lambda = 10^{-12}$. RS's $y^{\lambda_3} - s^{\lambda_4}$ has the same problem.

*Correction.* `expm1(λ log y) / λ` and, for RS, the identity $y^{\lambda_3} - s^{\lambda_4} = (y^{\lambda_3} - 1) - (s^{\lambda_4} - 1)$ written with `expm1`.

## Suspects that were checked and cleared

These looked wrong on reading, and tests or algebra showed they are not:

- **RS parameter regions in `check_param`.** Sampling $2\cdot10^5$ random parameter sets gave 78 321 valid ones, and every one of them had a monotone $Q$ on 4001 points.
- **Moment formulas.** Beta-function sums agree with direct numerical integration to the printed 5 digits for RS, FMKL and VSL away from $\lambda = 0$.
- **RS `fit_MM`'s swap of $\lambda_3, \lambda_4$ with a sign change.** Algebraically it represents the same distribution with the same mean.
- **`fit_PM` at the ends of its $u$ range.** It returns a result for $u = 0$ and $u = 0.25$ without failing. At $u = 0.25$, $\rho_4 = 1$ carries no information, so this is a poor choice by the caller, not a defect. The torch backend requires $0 < u < 0.25$.

## Behavior of the corrected class that differs on purpose

For anyone comparing numbers with the original:

- `CDF_num` and everything built on it give different values in the tails, by U02. Inside $y \in [0.05, 0.95]$ they agree with the original within its $10^{-4}$ tolerance.
- `fit_ML`, `fit_MPS` and `fit_starship` therefore can give somewhat different estimates, and `fit_MPS` is a different objective, by U11.
- `fit_PM` with RS gives different estimates, by U01.
- `correct_supp` gives the exact solution of its two end conditions, by U06, where the original stops at `fmin`'s tolerance.
