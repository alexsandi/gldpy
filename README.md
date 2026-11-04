# gldpy-gpu

The Generalized Lambda Distribution for Python, with an optional PyTorch/CUDA backend and accurate tail probabilities.

This is a fork of [gldpy](https://github.com/alexsandi/gldpy) by Aleksandra Alekseeva (MIT). It corrects
[14 defects of the original](docs/upstream-bugs.md) and adds the GPU backend. The import name is `gldpy_gpu`, so it can be
installed next to `gldpy`.

| Document | Content |
|---|---|
| [CHANGELOG.md](CHANGELOG.md) | All changes, including breaking ones |
| [docs/upstream-bugs.md](docs/upstream-bugs.md) | Each defect of `gldpy` 0.2: cause, evidence, correction |
| [docs/numerics.md](docs/numerics.md) | The numerical methods, and what limits them |
| [docs/validation.md](docs/validation.md) | Fits against known distributions, tails, precision, known limitations |
| [docs/benchmarks.md](docs/benchmarks.md) | Speed, what acts on it, and what a GPU gains |

## Install

```bash
pip install gldpy-gpu            # NumPy/SciPy only
pip install "gldpy-gpu[gpu]"     # adds PyTorch for the GPU backend
```

Pascal GPUs (GTX 10xx, compute capability 6.1) need a specific torch, because newer wheels dropped their kernels:
`pip install torch --index-url https://download.pytorch.org/whl/cu126`. If the installed torch can't run on your GPU,
`gpu.GLD` warns and uses torch on the CPU.

## Usage

```python
import torch
from gldpy_gpu import gpu

gld = gpu.GLD("FMKL")                 # CUDA when available, else CPU
p = torch.tensor([0.0, 1.0, 0.2, 0.3], device=gld.device)
x = gld.rand(p, size=10_000_000)      # sampling on the GPU
params = gld.fit_LMM(x)               # also fit_PM, fit_ML
```

The NumPy class has the same interface as upstream's, plus the tail methods: `from gldpy_gpu import GLD`. It also has
`mean`, `var`, `skewness`, `kurtosis`, `fit_MM`, `fit_MPS`, `fit_starship`, `fit_curve`, `GoF_tests` and the plots, which the GPU class doesn't.

**Without PyTorch**, `gpu.GLD` warns and runs on NumPy: the same calls work, and return ndarrays. `gpu.BACKEND` is `"torch"` or
`"numpy"`. Arguments that only make sense for torch (`device`, `dtype`, `n_grid`, `refine_steps`, `lr`) are accepted and ignored, and
fits use the slower SciPy optimizers.

## Tail probabilities

Don't compute an upper tail as `1 - CDF_num(x)`, or as `Q(1 - s)`: `1 - s` rounds to 1 for small `s`.

| Task | Call | Note |
|---|---|---|
| P(X <= x) | `CDF_num(x, p)` | relative accuracy in the lower tail |
| P(X > x) | `SF_num(x, p)` | relative accuracy in the upper tail |
| quantile at upper-tail probability `s` | `ISF(s, p)` | exact where `1 - s` rounds to 1 |
| log density | `log_PDF_num(x, p)` | valid where `PDF_num` underflows to 0 |

Infinite tails keep a relative accuracy of about 1e-14 in float64 and 1e-6 in float32, down to the smallest representable
number. Bounded tails run out sooner, for a reason no solver can fix: `x` becomes indistinguishable from the end of the support.
Measurements are in [docs/validation.md](docs/validation.md#4-tail-accuracy).

## Precision

Every method takes `dtype`, and computes in that precision only:

```python
gld.fit_LMM(data, dtype=torch.float64)    # or "float64"
gpu.GLD("FMKL", dtype=torch.float64)      # a default for this instance
```

The order is: the method's `dtype`, then the constructor's, then the widest float dtype among tensor or NumPy inputs, then
float32 for plain Python numbers. Results have that dtype. Fits are as good in float32 as in float64 (median difference in KS distance
about 2e-8, over 54 fits of known distributions). Use float64 for bounded tails below about 1e-6, or when the data have a large offset relative to their spread.

## Speed

GTX 1070, FMKL, float32, median. CUDA loses below about 1e5 points because each operation pays a launch cost, and it wins by a growing margin above that:

| Operation | N | NumPy | torch on CPU | CUDA |
|---|---|---|---|---|
| `Q` | 1e4 | 0.6 ms | 0.5 ms | 0.8 ms |
| `Q` | 1e7 | 1.8 s | 237 ms | 6.1 ms |
| `rand` | 1e7 | 2.1 s | 377 ms | 9.6 ms |
| `CDF_num` | 1e5 | 29 s | 16 ms | 7.8 ms |
| `CDF_num` | 1e7 | not run | 2.4 s | 86 ms |
| `fit_LMM` | 1e7 | not run | 2.4 s | 118 ms |

[docs/benchmarks.md](docs/benchmarks.md) explains the method and what acts on speed.

---

# gldpy

_______

gldpy is Python package that implements  tools for using Generalized Lambda Distribution (GLD) of three different parameterization types.

GLD is flexible family of continuous probability distributions with wide variety of density shapes under one functional form. It specified by four parameters which determine location, scale and shape (skewness and kurtosis) of the distribution.

GLD is easily described in terms of quantile function which makes it easy to use in simulation studies. 

It can approximate a lot of well-known distributions such as normal, uniform, exponential, Student's, gamma, some of beta distributions and many others. 

GLD shapes can be symmetric or asymmetric  and have different combination of finite and infinite tails, they include unimodal, U-shaped, J-shaped and monotone density functions. So it is particularly useful tool for data fitting applications.

For more information, see the documentation.



# Usage examples

___

https://github.com/alexsandi/using_gldpy



# References

_______

1. Chalabi, Y., Scott, D.J., & Wuertz, D. 2012. Flexible distribution modeling with the generalized lambda distribution.

2. Cheng, R.C.H., & Amin, N.A.K. 1983. Estimating parameters in continuous univariate distributions with a shifted origin. Journal of the Royal Statistical Society: Series B (Methodological), 45(3), 394–403.

3. Freimer, M., Kollia, G., Mudholkar, G.S., & Lin, C.T. 1988. A study of the generalized Tukey lambda family. Communications in Statistics-Theory and Methods, 17, 3547–3567. 

4. Karian, Z.A., Dudewicz, E.J. 2000. Fitting statistical distributions: the generalized lambda distribution and generalized bootstrap methods. Chapman and Hall/CRC.

5. Karvanen, J. and Nuutinen, A. 2008. Characterizing the generalized lambda distribution by L-moments. Computational Statistics & Data Analysis, 52(4):1971–1983. 

6. King, R. A. R., and MacGillivray, H. L. 1999. "A Starship Estimation Method for the Generalized Lambda Distributions," Australian and New Zealand Journal of Statistics, 41, 353–374. 
7.   Ramberg, J.S., & Schmeiser, B.W. 1974. An approximate method for generating asymmetric random variables. Communications of the ACM, 17(2), 78–82   
8.   Ranneby, B. 1984. The maximum spacing method. an estimation method related to the maximum likelihood method. Scandinavian Journal of Statistics, 93–112.
9.   Su, S. 2007. Numerical maximum log likelihood estimation for generalized lambda distributions. Computational Statistics & Data Analysis, 51(8), 3983–3998. 
10.  Van Staden, Paul J., & M.T. Loots. 2009. Method of L-moment estimation for generalized lambda distribution. Third Annual ASEARC Conference. Newcastle, Australia.