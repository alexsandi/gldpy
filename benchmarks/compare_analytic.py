"""Fit known distributions with every backend and report the distance from the exact answer.

Usage: uv run python benchmarks/compare_analytic.py [--n 20000] [--out benchmarks/results/analytic.json]

The "sample" is the exact population quantile grid Q_true((k + 0.5) / n), so there is no sampling
noise: what remains is the error of the method and of the implementation. Metrics, on the same
grid of probabilities p, using the float64 torch CDF to evaluate the fitted distribution:

  ks     sup |F_fit(Q_true(p)) - p|, the Kolmogorov distance, p in (0.0005, 0.9995)
  qerr   max |Q_fit(p) - Q_true(p)| / sd_true, p in [0.01, 0.99]
  tail   Q_fit(1 - 1e-4) / Q_true(1 - 1e-4) - 1, after centering at the true median

It also records the tail accuracy of the CDF, as the round trip s -> Q(s) -> F(Q(s)).
"""
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from gldpy import GLD as Original
from scipy import stats

from gldpy_gpu import GLD
from gldpy_gpu import gpu

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() and gpu._cuda_usable() else "cpu"
FIT_KW = dict(disp_optimizer=False, disp_fit=False, test_gof=False)

DISTRIBUTIONS = {
    "uniform": stats.uniform(-1, 2),
    "logistic": stats.logistic(),
    "exponential": stats.expon(),
    "normal": stats.norm(),
    "student_t5": stats.t(5),
    "gamma_2": stats.gamma(2),
    "lognormal_0.25": stats.lognorm(0.25),
    "beta_2_2": stats.beta(2, 2),
    "beta_2_5": stats.beta(2, 5),
}
TYPES = ("RS", "FMKL", "VSL")
CPU_START = {"RS": [0.1, 0.1], "FMKL": [0.1, 0.1], "VSL": [0.3]}


def metrics(ptype, fit, dist):
    fit = np.asarray(fit, dtype=float)
    if not np.isfinite(fit).all() or not GLD(ptype).check_param(fit):
        return dict(ks=None, qerr=None, tail=None, param=[float(v) for v in fit], valid=False)
    g = gpu.GLD(ptype, device=DEVICE)
    p64 = torch.as_tensor(fit, dtype=torch.float64, device=DEVICE)
    probs = np.linspace(0.0005, 0.9995, 1999)
    F = g.CDF_num(torch.as_tensor(dist.ppf(probs), dtype=torch.float64, device=DEVICE), p64, xtol=1e-10).cpu().numpy()
    inner = np.linspace(0.01, 0.99, 981)
    q = GLD(ptype).Q(inner, fit)
    median = dist.ppf(0.5)
    tail = (GLD(ptype).ISF(1e-4, fit) - median) / (dist.isf(1e-4) - median) - 1.0
    return dict(
        ks=float(np.abs(F - probs).max()), qerr=float(np.abs(q - dist.ppf(inner)).max() / dist.std()),
        tail=float(tail), param=[float(v) for v in fit], valid=True,
    )


def fit_all(ptype, sample, n_ml):
    """(backend, method) -> parameters, or the exception's name."""
    start = None if ptype == "VSL" else CPU_START[ptype]
    runs = {}
    for label, cls in (("original", Original), ("corrected", GLD)):
        g = cls(ptype)
        runs[label, "LMM"] = lambda g=g: g.fit_LMM(sample, start, xtol=1e-7, **FIT_KW)
        runs[label, "PM"] = lambda g=g: g.fit_PM(sample, CPU_START[ptype], xtol=1e-7, **FIT_KW)
        if ptype != "VSL":
            runs[label, "MM"] = lambda g=g: g.fit_MM(sample, np.array(CPU_START[ptype]), xtol=1e-7, **FIT_KW)
    for dtype, label in ((torch.float64, "torch_f64"), (torch.float32, "torch_f32")):
        g = gpu.GLD(ptype, device=DEVICE, dtype=dtype)
        d = torch.as_tensor(sample, dtype=dtype, device=DEVICE)
        thin = torch.as_tensor(sample[:: max(1, len(sample) // n_ml)], dtype=dtype, device=DEVICE)
        runs[label, "LMM"] = lambda g=g, d=d: g.fit_LMM(d).cpu().numpy()
        runs[label, "PM"] = lambda g=g, d=d: g.fit_PM(d).cpu().numpy()
        runs[label, "ML"] = lambda g=g, thin=thin: g.fit_ML(thin).cpu().numpy()
    return runs


TAIL_CASES = {
    "FMKL infinite tails (-0.2, -0.1)": ("FMKL", [0.0, 1.0, -0.2, -0.1]),
    "RS infinite tails (-0.1, -0.15)": ("RS", [0.0, -0.2, -0.1, -0.15]),
    "VSL infinite tails (0.4, -0.2)": ("VSL", [0.0, 1.0, 0.4, -0.2]),
    "FMKL bounded (0.2, 0.3)": ("FMKL", [0.0, 1.0, 0.2, 0.3]),
    "RS bounded (0.1, 0.15)": ("RS", [0.0, 0.2, 0.1, 0.15]),
    "VSL bounded (0.4, 0.3)": ("VSL", [0.0, 1.0, 0.4, 0.3]),
}
TAIL_PROBS = [1e-4, 1e-8, 1e-12, 1e-30, 1e-100]


def tail_accuracy():
    """Relative error of a tail probability s after the round trip x = Q(s), s' = F(x), worst of both tails."""
    rows = []
    for name, (ptype, p) in TAIL_CASES.items():
        for s in TAIL_PROBS:
            row = dict(case=name, s=s)
            for label, dtype in (("f64", torch.float64), ("f32", torch.float32)):
                g = gpu.GLD(ptype, device=DEVICE)
                pt = torch.as_tensor(p, dtype=dtype, device=DEVICE)
                st = torch.as_tensor([s], dtype=dtype, device=DEVICE)
                if st.item() == 0.0 or s < torch.finfo(dtype).tiny:
                    row[label] = None
                    continue
                low = abs(g.CDF_num(g.Q(st, pt), pt).item() / st.item() - 1.0)
                up = abs(g.SF_num(g.ISF(st, pt), pt).item() / st.item() - 1.0)
                row[label] = max(low, up)
            np_g = GLD(ptype)
            row["numpy"] = max(
                abs(np_g.CDF_num(np_g.Q(s, p), p)[0] / s - 1.0), abs(np_g.SF_num(np_g.ISF(s, p), p)[0] / s - 1.0)
            )
            if s >= 1e-12:
                orig = Original(ptype)
                # The original has no upper-tail method, so only the lower tail can be compared.
                row["original_lower"] = abs(orig.CDF_num(orig.Q(np.array([s]), p), p)[0] / s - 1.0)
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--n-ml", type=int, default=5_000)
    ap.add_argument("--out", default="benchmarks/results/analytic.json")
    args = ap.parse_args()
    results = []
    for name, dist in DISTRIBUTIONS.items():
        sample = dist.ppf((np.arange(args.n) + 0.5) / args.n)
        for ptype in TYPES:
            for (backend, method), run in fit_all(ptype, sample, args.n_ml).items():
                try:
                    row = metrics(ptype, run(), dist)
                except Exception as exc:  # the table reports failures instead of hiding them
                    row = dict(ks=None, qerr=None, tail=None, param=None, valid=False, error=type(exc).__name__)
                results.append(dict(distribution=name, type=ptype, backend=backend, method=method, **row))
                ks = "   fail " if row["ks"] is None else f"{row['ks']:.2e}"
                print(f"{name:16s}{ptype:5s}{backend:11s}{method:4s} ks {ks}  qerr {row['qerr'] if row['qerr'] is None else round(row['qerr'],5)}  tail {row['tail'] if row['tail'] is None else round(row['tail'],4)}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tails = tail_accuracy()
    out.write_text(json.dumps(dict(n=args.n, n_ml=args.n_ml, device=DEVICE, results=results, tails=tails), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
