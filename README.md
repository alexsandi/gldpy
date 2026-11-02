# gldpy-gpu

Fork of [gldpy](https://github.com/alexsandi/gldpy) (Aleksandra Alekseeva, MIT) that adds a PyTorch/CUDA backend, `gldpy.gpu`.
The import name is `gldpy_gpu`, so it can be installed next to upstream `gldpy`.

## Install

```bash
pip install gldpy-gpu          # or: uv add gldpy-gpu
```

PyTorch is a plain dependency (`torch>=2.0`): pip/uv picks the wheel for *your* machine, and
modern GPUs work with the default build. Only Pascal GPUs (GTX 10xx, compute capability 6.1) need an
older CUDA 12.6 build, because newer wheels dropped those kernels. If the installed torch can't run
on your GPU, `gldpy_gpu.gpu.GLD` warns and falls back to CPU. Fix with:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

## GPU usage

```python
import torch
from gldpy_gpu import gpu

gld = gpu.GLD("FMKL")                 # uses CUDA when available, else CPU
p = torch.tensor([0.0, 1.0, 0.2, 0.3], device=gld.device)
x = gld.rand(p, size=10_000_000)       # sampling on the GPU
params = gld.fit_LMM(x[:100_000])      # also fit_PM, fit_ML
```

The original NumPy/SciPy API is unchanged: `from gldpy_gpu import GLD`.

Benchmarks: `uv run python benchmarks/bench_gpu.py`.

## Speed (GTX 1070, torch 2.14.1+cu126, FMKL, best of runs)

| Operation | N | CPU (NumPy) | GPU (float32) | Speedup |
|---|---|---|---|---|
| `Q` | 10 000 | 0.7 ms | 1.1 ms | 0.7x (launch overhead) |
| `Q` | 1 000 000 | 116 ms | 1.5 ms | 77x |
| `Q` | 10 000 000 | 1.63 s | 7.6 ms | 215x |
| `CDF_num` | 10 000 | 5.6 s | 32 ms | 177x |
| `CDF_num` | 100 000 | 49 s | 32 ms | ~1500x |
| `fit_LMM` (100k samples) | VSL / RS / FMKL | 1.6 s / 1.3 s / 3.2 s | 6 ms / 311 ms / 386 ms | 271x / 4x / 8x |

Small batches (below ~50k points) are faster on the CPU. float64 is 4-5x slower than float32 on this card.

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