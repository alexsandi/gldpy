# Numerical methods

How `gldpy_gpu` computes each quantity, why, and what limits its accuracy. Measured results are in [validation.md](validation.md) and speed in
[benchmarks.md](benchmarks.md).

## 1. The distribution

A GLD is defined by its quantile function. With $y \in [0, 1]$ and $s = 1 - y$:

| Type | $Q(y)$ | Parameters |
|---|---|---|
| RS | $\lambda_1 + (y^{\lambda_3} - s^{\lambda_4}) / \lambda_2$ | valid only in six regions of $(\lambda_2, \lambda_3, \lambda_4)$ |
| FMKL | $\lambda_1 + \big(\frac{y^{\lambda_3} - 1}{\lambda_3} - \frac{s^{\lambda_4} - 1}{\lambda_4}\big) / \lambda_2$ | $\lambda_2 > 0$ |
| VSL | $\alpha + \beta\big((1-\delta)\frac{y^{\lambda} - 1}{\lambda} - \delta\frac{s^{\lambda} - 1}{\lambda}\big)$ | $\beta > 0$, $0 \le \delta \le 1$ |

Fractions of the form $(u^\lambda - 1)/\lambda$ take their limit $\log u$ at $\lambda = 0$. The density quantile function is
$f(y) = 1 / Q'(y)$, with $Q'(y) = c_3\, y^{e_3} + c_4\, s^{e_4}$ up to scale. Neither the CDF nor the density at a given $x$ has a closed form.

Three special cases have exact closed forms, and the tests use them as oracles:

| Distribution | Parameters | Exact |
|---|---|---|
| Logistic | FMKL$(0, 1, 0, 0)$, VSL$(0, 2, \tfrac12, 0)$ | $F(x) = 1/(1 + e^{-x})$ |
| Exponential | VSL$(0, 1, 1, 0)$ | $1 - F(x) = e^{-x}$ |
| Pareto | VSL$(0, 1, 1, -1)$ | $1 - F(x) = 1/(1 + x)$ |
| Uniform | RS$(0, 1, 1, 1)$, FMKL$(0, 1, 1, 1)$, FMKL$(0, 1, 2, 2)$, VSL$(0, 1, \tfrac12, 1)$ and others | |

## 2. Evaluation in log space

**Problem.** Writing $s = 1 - y$ destroys the upper tail: in float64, $1 - 10^{-17} = 1$, and in float32 this happens already at $3\cdot10^{-8}$.
Powers of tiny numbers can also underflow, and $(y^\lambda - 1)/\lambda$ cancels as $\lambda \to 0$.

**Method.** Every formula takes $\log y$ and $\log s$ as inputs, never $y$ and $s$. Both are known accurately from whichever of the following the caller has:

| Input | $\log y$ | $\log s$ |
|---|---|---|
| Lower tail probability $y$ (`Q`) | `log(y)` | `log1p(-y)` |
| Upper tail probability $s$ (`ISF`) | `log1p(-s)` | `log(s)` |
| Log-odds $t$ (the CDF solver) | `logsigmoid(t)` | `logsigmoid(-t)` |

The building blocks are then:

$$\frac{u^\lambda - 1}{\lambda} = \frac{\operatorname{expm1}(\lambda \log u)}{\lambda},\qquad
y^{\lambda_3} - s^{\lambda_4} = \operatorname{expm1}(\lambda_3 \log y) - \operatorname{expm1}(\lambda_4 \log s).$$

The second is an identity, which replaces the difference of two numbers near 1 by a difference of two small numbers known to full relative accuracy.

**Density.** $\log f = \log(\text{scale}) - \log|c_3 y^{e_3} + c_4 s^{e_4}|$. The sum is evaluated with its larger term factored out,
$\log|a + b| = m + \log|\operatorname{sign}(a)e^{\log|a| - m} + \operatorname{sign}(b)e^{\log|b| - m}|$ with $m = \max(\log|a|, \log|b|)$, so it does not overflow.
This is needed where the signs differ (RS regions 5 and 6), which excludes a plain `logaddexp`.

**Autograd.** In `_box_cox`, $\partial_\lambda$ of `expm1(λ log u) / λ` itself cancels as $\lambda \to 0$. When $\lambda$ carries a gradient and $|\lambda \log u| < 10^{-3}$,
a third-order series replaces it; the limit of the derivative, $\tfrac12 \log^2 u$, is tested. The maximum $m$ above is detached: it is a scaling constant, and $\partial_c \log|c|$ at $c = 0$ (VSL with $\delta$ on a bound) would otherwise produce NaN.

## 3. The CDF: root finding on the log-odds

**Problem.** $F(x)$ solves $Q(y) = x$. On $y$, an absolute tolerance cannot resolve tail probabilities smaller than itself ([U02](upstream-bugs.md#u02-cdf_num-loses-all-accuracy-in-the-tails)).

**Method.** Solve for $t = \operatorname{logit}(y) = \log(y / s)$, with

$$\frac{dt}{dx} = \frac{f(y)}{y\,s}.$$

Since $\log y \approx t$ in the lower tail and $\log s \approx -t$ in the upper one, an absolute tolerance $\tau$ on $t$ is a *relative* tolerance $\tau$ on the smaller tail probability. Both $F = \sigma(t)$ and $1 - F = \sigma(-t)$
then come out without cancellation.

The torch solver `solve_logit_cdf` is a vectorized safeguarded Newton iteration (Numerical Recipes' `rtsafe` strategy):

1. **Start** from linear interpolation in a table of 512 values of $Q(t)$ for $t \in [-40, 40]$, found by binary search, because $Q$ is monotone. A caller can pass a warm start instead (`fit_ML` uses the previous step's solution).
2. **Bracket.** Each element keeps $[\text{low}, \text{high}]$, initially $\pm\log(\text{tiny})$, updated from the sign of $Q(t) - x$.
3. **Step.** Take the Newton step if it stays in the bracket and at most halves the previous step. Otherwise bisect. This guarantees at worst the $\log_2$ convergence of bisection, about 31 iterations in float64 at $\tau = 10^{-6}$.
4. **Stop** an element when $|Q(t) - x| \le 2\epsilon(|x| + |Q(t)|)$, or the step or bracket falls below $\tau$. Here $\tau = \max(\text{xtol}, 32\epsilon)$: below that floor the steps are rounding noise and do not settle.
5. **Retire** finished elements: once a quarter of those remaining have finished, they leave the arrays, so an iteration costs in proportion to the elements still unresolved.

Elements at or beyond the support, and NaNs, are decided before the iteration and never enter it.

The NumPy class uses `scipy.optimize.brentq` per element on the same equation, with the same bracket idea at $t = \pm 700$.

**Limits.** Beyond the iteration's control:

- *Floor.* $t$ saturates at $\pm\log(\text{tiny})$: tail probabilities below about $10^{-38}$ (float32) or $10^{-308}$ (float64) report as that floor.
- *Conditioning.* On a bounded tail $x \approx x_\text{sup} - c\,s^{\lambda}$, so a relative change $\epsilon$ in $x$ moves $s$ by a relative $\epsilon\,|x| / (\lambda\, c\, s^{\lambda})$. Once $c\,s^\lambda < \epsilon |x|$, the input $x$ cannot distinguish $s$ from any smaller value. No solver can recover this: it is a property of $x$ at that precision. The measured limits are in [validation.md](validation.md#tail-accuracy).

## 4. Random numbers

Inverse-transform sampling, $X = Q(U)$, needs $\log U$ and $\log(1 - U)$. A plain uniform has a fixed step, $2^{-24}$ in float32, with two consequences: $U = 0$ occurs about once in 16.8M draws and gives $Q = -\infty$ on an infinite support, and the upper tail ends sharply at $Q(1 - 2^{-24})$.

`_sample_log_uniform` removes both, in any precision. It works with the distance $V = \min(U, 1 - U)$ to the nearer end and refines small values: if $V < c$ then $V / c$ is again uniform given that event, so $V$ is replaced by $c \cdot V'$ with a fresh $V'$, with $c = 2^{-12}$. Only a fraction $2^{-11}$ of the elements enter the second round, so the extra work is negligible, and $\log V$ accumulates with no underflow.

## 5. Sample statistics

**L-moments.** The unbiased estimators of Hosking are linear in the order statistics, with weights that are polynomials in the rank. `compute_sample_lmoments` computes them in $O(n \log n)$ for the sort, and two things keep them accurate in float32:

- Weights are products of ratios such as $\frac{k}{n}\cdot\frac{k-1}{n-1}$, each in $[0, 1]$, so nothing overflows.
- The data are centered first. The weights of $l_2, l_3, l_4$ sum to 0, so centering changes nothing mathematically but removes the rounding error of large offsets.

**Percentiles.** Linear interpolation like `numpy.quantile`. Written out because `torch.quantile` raises for inputs above $2^{24}$ elements, and with the index computed in a Python float, so that it stays exact in any precision.

**Plotting positions** $(k + \frac12) / n$ for `GoF_Q_metric`: $\log y = \log(k + \frac12) - \log n$ and $\log s = \log(n - k - \frac12) - \log n$. The upper values keep their relative accuracy.

## 6. Estimators

Each estimator is a system of equations in the two shape parameters, or one for VSL's percentile method. Location and scale then follow in closed form.

| Method | Equations | Solver, torch backend | Solver, NumPy class |
|---|---|---|---|
| `fit_LMM`, RS and FMKL | $\tau_3, \tau_4$ match the sample | multistart Levenberg-Marquardt | Nelder-Mead from a given start |
| `fit_LMM`, VSL | same | closed form (van Staden & Loots 2009) | same closed form |
| `fit_PM`, RS and FMKL | $\rho_3, \rho_4$ match the sample | multistart Levenberg-Marquardt | Nelder-Mead from a given start |
| `fit_PM`, VSL | $\rho_4$ matches; one unknown | all roots by bisection | Nelder-Mead from a given start |
| `fit_ML` | maximize the likelihood | Adam, implicit gradients | Nelder-Mead |
| `fit_MM`, `fit_MPS`, `fit_starship` | | not available | Nelder-Mead |

For FMKL, the L-moment equations use the closed forms of [U04](upstream-bugs.md#u04-fits-started-at-lambda--0-return-nan), which are regular at $\lambda = 0$.

### Multistart Levenberg-Marquardt

The shape equations may have **several exact roots**, for instance $(\lambda_3, \lambda_4) \approx (3.55, 2.88)$ besides $(0.2, 0.3)$ for percentile statistics of data drawn from FMKL(0.2, 0.3). So without a start from the caller, a single local solve is not enough. `solve_shape_multistart`:

1. Evaluates the residual on a $128 \times 128$ grid, and marks non-finite or invalid points as infinite.
2. Takes up to 16 local minima of the grid as starts.
3. Refines all of them in one batched Levenberg-Marquardt iteration, with Jacobians from autograd. A finite-difference step would be $\epsilon^{1/3} \approx 5\cdot10^{-3}$ in float32, larger than the root itself for logistic data, whose RS root lies near $\lambda = 10^{-4}$. Measured effect of this choice: KS of the float32 RS percentile fit to the logistic went from $3.7\cdot10^{-2}$ to $3.0\cdot10^{-5}$.
4. Among the results with squared residual below $\max(10^{-10}, (2000\epsilon)^2)$, keeps the one with the smallest Q-Q discrepancy on the data.

With `initial_guess`, it refines from there only, as the NumPy class does, and falls back to the grid with a warning if that fails.

**Choosing among roots is a decision the equations themselves do not make.** It can matter: FMKL(1, 1) and FMKL(2, 2) are the same uniform distribution, so no criterion can tell them apart, and the parameter values are not identified there, while the distribution is. Pass `initial_guess` to control which root comes back.

These solves are small and run on the CPU even when the data are on the GPU, because launch overhead makes small batches slower on a GPU. See [benchmarks.md](benchmarks.md#what-acts-on-speed).

### Maximum likelihood

Adam minimizes the mean negative log-likelihood, with these specifics:

- **Standardized data.** $z = (x - \text{median}) / \text{sd}$, so that `lr` means the same in any unit and a large offset does not consume float32 precision. Results are transformed back exactly. A test checks invariance over a change of unit by $10^{\pm5}$.
- **Implicit gradients.** The CDF solve is not differentiated. With $t^*$ the solution under `no_grad`, the implicit function theorem gives $\partial_\theta t = -(\partial_\theta Q) \cdot dt/dx$, which is what $\tilde t = t^* - (Q(t^*; \theta) - z)\, (dt/dx)_\text{detached}$ yields under autograd, with value $t^*$.
- **Support.** A point outside the support has zero likelihood. After each step, `_with_support_covering` applies the smallest location and scale change that puts the data inside, with a margin of $10^{-6}$ of the range. Since $Q$ is affine in those two parameters, this is exact and leaves the shape parameters free. Two simpler alternatives failed in testing: rejecting such steps freezes the shape parameters (RS on exponential data stopped at NLL 1.19 against a true 1.0), and `correct_supp`, which moves a full tail quantile, is too large a jump to apply at every step.
- **Result.** The iterate with the lowest negative log-likelihood seen, so the result is never worse than the start.

**Limits.** Adam is a local method, and a likelihood with boundary-dependent support is not smooth at the boundary. On non-members it can stop at a local optimum: RS on exponential data stays above KS $0.1$, where the L-moment fit has $6.5\cdot10^{-5}$. At the default 300 steps it may also not have converged ([validation.md](validation.md#5-known-limitations)). For bounded-support shapes with $\lambda > 1$ the density diverges at the support's end, so the likelihood is theoretically unbounded there. The margin above limits that, but check a maximum-likelihood result against the L-moment fit, and consider `fit_MPS` of the NumPy class if the support is the question.

## 7. Precision

Each call computes in one float dtype, chosen in this order: the method's `dtype` argument; the constructor's `dtype`; the widest float dtype among tensor or NumPy inputs; float32 for plain Python numbers. `tests/test_gpu_dtype.py` runs each public method under a `TorchFunctionMode` that fails on any tensor created in the other precision, so the rule is enforced, not merely documented. The one exemption is the scalar step counter inside `torch.optim.Adam`, which is bookkeeping.

The NumPy class always computes in float64.
