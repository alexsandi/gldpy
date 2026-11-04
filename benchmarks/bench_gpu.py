"""Benchmarks behind docs/benchmarks.md. Usage: uv run python benchmarks/bench_gpu.py [--quick]

Writes benchmarks/results/benchmarks.json. Method:
- Each case first runs once as a warm-up, which is not timed (CUDA context, lazy kernels, page faults).
- Then it repeats until REPEAT_SECONDS or REPEATS, whichever is first; the median and min are kept.
- CUDA timings call torch.cuda.synchronize() before and after, since CUDA calls return early.
- Every data array is built before the clock starts, and already on the device being measured.
"""
import argparse
import json
import platform
import statistics
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from gldpy import GLD as Original

from gldpy_gpu import GLD as Corrected
from gldpy_gpu import _torch_backend as backend
from gldpy_gpu import __version__, gpu

warnings.filterwarnings("ignore")
PARAM = [0.0, 1.0, 0.2, 0.3]
PTYPE = "FMKL"
REPEATS = 7
REPEAT_SECONDS = 4.0
FIT_KW = dict(disp_optimizer=False, disp_fit=False, test_gof=False)
GPU = torch.cuda.is_available() and gpu._cuda_usable()


def measure(fn, cuda):
    """(median seconds, min seconds, repeats)."""

    def once():
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter() - start

    once()
    times, spent = [], 0.0
    while len(times) < REPEATS and (spent < REPEAT_SECONDS or len(times) < 2):
        times.append(once())
        spent += times[-1]
    return statistics.median(times), min(times), len(times)


def torch_cases(device, dtype, ptype=PTYPE, param=PARAM):
    g = gpu.GLD(ptype, device=device, dtype=dtype)
    p = torch.as_tensor(param, dtype=dtype, device=device)
    return g, p


def sample(n, device, dtype, ptype=PTYPE, param=PARAM):
    g, p = torch_cases(device, dtype, ptype, param)
    return g.rand(p, size=n, generator=torch.Generator(device=device).manual_seed(1))


def op(name, g, p, y, x):
    return {
        "Q": lambda: g.Q(y, p),
        "PDF_Q": lambda: g.PDF_Q(y, p),
        "rand": lambda: g.rand(p, size=y.numel()),
        "CDF_num": lambda: g.CDF_num(x, p),
        "PDF_num": lambda: g.PDF_num(x, p),
        "GoF_Q_metric": lambda: g.GoF_Q_metric(x, p),
        "fit_LMM": lambda: g.fit_LMM(x),
        "fit_PM": lambda: g.fit_PM(x),
        "fit_ML": lambda: g.fit_ML(x, steps=60),
    }[name]


# Largest N per operation and backend: set by RAM (~4 GiB free), GPU memory (~7 GiB) and patience.
LIMITS = {
    "Q": dict(original=10**7, corrected=10**7, cpu=10**7, cuda_f32=10**8, cuda_f64=3 * 10**7),
    "PDF_Q": dict(original=10**7, corrected=10**7, cpu=10**7, cuda_f32=10**8, cuda_f64=3 * 10**7),
    "rand": dict(original=10**7, corrected=10**7, cpu=10**7, cuda_f32=10**8, cuda_f64=3 * 10**7),
    "CDF_num": dict(original=10**4, corrected=10**5, cpu=10**7, cuda_f32=3 * 10**7, cuda_f64=10**7),
    "PDF_num": dict(original=10**4, corrected=10**4, cpu=10**7, cuda_f32=3 * 10**7, cuda_f64=10**7),
    "GoF_Q_metric": dict(original=10**7, corrected=10**7, cpu=10**7, cuda_f32=3 * 10**7, cuda_f64=10**7),
    "fit_LMM": dict(original=10**5, corrected=10**5, cpu=10**7, cuda_f32=3 * 10**7, cuda_f64=10**7),
    "fit_PM": dict(original=10**6, corrected=10**6, cpu=10**7, cuda_f32=3 * 10**7, cuda_f64=10**7),
    "fit_ML": dict(cpu=10**5, cuda_f32=10**6, cuda_f64=10**6),
}
SIZES = [10**k for k in range(2, 9)] + [3 * 10**7]


def numpy_call(name, g, y, x):
    start = [0.2, 0.3]
    return {
        "Q": lambda: g.Q(y, PARAM),
        "PDF_Q": lambda: g.PDF_Q(y, PARAM),
        "rand": lambda: g.rand(PARAM, size=y.size),
        "CDF_num": lambda: g.CDF_num(x, PARAM),
        "PDF_num": lambda: g.PDF_num(x, PARAM),
        "GoF_Q_metric": lambda: g.GoF_Q_metric(x, PARAM),
        "fit_LMM": lambda: g.fit_LMM(x, start, **FIT_KW),
        "fit_PM": lambda: g.fit_PM(x, start, **FIT_KW),
    }[name]


def sweep(sizes, rows):
    rng = np.random.default_rng(1)
    for name, limits in LIMITS.items():
        for n in sorted(sizes):
            if n >= 10:
                # NumPy backends
                for label, cls in (("original", Original), ("corrected", Corrected)):
                    if n <= limits.get(label, 0):
                        y = rng.random(n)
                        x = Corrected(PTYPE).Q(rng.random(n), PARAM)
                        med, low, k = measure(numpy_call(name, cls(PTYPE), y, x), cuda=False)
                        rows.append(dict(op=name, n=n, backend=label, median=med, min=low, repeats=k))
                        print(f"{name:13s}{n:>11,d} {label:10s}{med * 1e3:12.3f} ms", flush=True)
                # torch backends
                targets = [("torch_cpu_f32", "cpu", torch.float32, "cpu"), ("torch_cpu_f64", "cpu", torch.float64, "cpu")]
                if GPU:
                    targets += [("cuda_f32", "cuda", torch.float32, "cuda_f32"), ("cuda_f64", "cuda", torch.float64, "cuda_f64")]
                for label, device, dtype, key in targets:
                    if n > limits.get(key, 0):
                        continue
                    g, p = torch_cases(device, dtype)
                    y = torch.rand(n, dtype=dtype, device=device)
                    x = sample(n, device, dtype)
                    if device == "cuda":
                        torch.cuda.reset_peak_memory_stats()
                    med, low, k = measure(op(name, g, p, y, x), cuda=device == "cuda")
                    peak = torch.cuda.max_memory_allocated() if device == "cuda" else None
                    rows.append(dict(op=name, n=n, backend=label, median=med, min=low, repeats=k, peak_bytes=peak))
                    print(f"{name:13s}{n:>11,d} {label:14s}{med * 1e3:12.3f} ms", flush=True)
                    del y, x
                    if device == "cuda":
                        torch.cuda.empty_cache()


def factors(out):
    """What acts on speed, one factor at a time. FMKL(0.2, 0.3), unless stated otherwise."""
    device = "cuda" if GPU else "cpu"
    cuda = device == "cuda"

    # launch latency: the same trivial op on a tiny array
    a = torch.ones(1, device=device)
    out["launch_seconds"] = dict(zip(("median", "min", "repeats"), measure(lambda: a + a, cuda)))
    cpu_a = torch.ones(1)
    out["launch_seconds_cpu"] = dict(zip(("median", "min", "repeats"), measure(lambda: cpu_a + cpu_a, False)))

    if cuda:
        rows = []
        for n in (10**4, 10**6, 10**7, 3 * 10**7):
            host = torch.rand(n)
            on_gpu = host.to("cuda")
            up = measure(lambda: host.to("cuda"), True)[0]
            down = measure(lambda: on_gpu.cpu(), True)[0]
            rows.append(dict(n=n, to_gpu=up, to_cpu=down))
        out["transfer"] = rows

    # precision on each device
    rows = []
    for dev in ("cpu", device) if cuda else ("cpu",):
        for dtype in (torch.float32, torch.float64):
            y = torch.rand(10**7, dtype=dtype, device=dev)
            med = {}
            for label, fn in (
                ("log", lambda: torch.log(y)), ("expm1", lambda: torch.expm1(y)), ("exp", lambda: torch.exp(y)),
                ("add", lambda: y + y), ("mul", lambda: y * y), ("sort", lambda: torch.sort(y)),
            ):
                med[label] = measure(fn, dev == "cuda")[0]
            rows.append(dict(device=dev, dtype=str(dtype)[6:], **med))
            del y
    out["primitives_1e7"] = rows

    # CPU threads
    rows = []
    saved = torch.get_num_threads()
    y = torch.rand(10**7, dtype=torch.float32)
    p = torch.as_tensor(PARAM, dtype=torch.float32)
    g = gpu.GLD(PTYPE, device="cpu")
    for threads in (1, 2, 4, 8):
        torch.set_num_threads(threads)
        rows.append(dict(threads=threads, Q=measure(lambda: g.Q(y, p), False)[0]))
    torch.set_num_threads(saved)
    out["cpu_threads_Q_1e7"] = rows

    # solver: xtol, start, where the data sit
    n = 10**6
    rows = []
    for dtype in (torch.float32, torch.float64):
        g, p = torch_cases(device, dtype)
        x = sample(n, device, dtype)
        for xtol in (1e-3, 1e-4, 1e-6, 1e-8, 1e-10, 1e-12):
            rows.append(dict(dtype=str(dtype)[6:], xtol=xtol, seconds=measure(lambda: g.CDF_num(x, p, xtol=xtol), cuda)[0]))
    out["solver_xtol_1e6"] = rows

    rows = []
    for dtype in (torch.float32, torch.float64):
        g, p = torch_cases(device, dtype)
        x = sample(n, device, dtype)
        table = measure(lambda: backend.solve_logit_cdf(x, p, PTYPE), cuda)[0]
        zeros = torch.zeros_like(x)
        cold = measure(lambda: backend.solve_logit_cdf(x, p, PTYPE, t0=zeros), cuda)[0]
        rows.append(dict(dtype=str(dtype)[6:], table_start=table, cold_start=cold))
    out["solver_start_1e6"] = rows

    rows = []
    for ptype, param in (("FMKL", PARAM), ("FMKL", [0.0, 1.0, -0.2, -0.1]), ("RS", [0.0, 0.2, 0.1, 0.15]), ("VSL", [0.0, 1.0, 0.4, 0.3])):
        g, p = torch_cases(device, torch.float32, ptype, param)
        y = torch.rand(10**7, dtype=torch.float32, device=device)
        rows.append(dict(
            type=ptype, shape=param[2:], Q=measure(lambda: g.Q(y, p), cuda)[0], PDF_Q=measure(lambda: g.PDF_Q(y, p), cuda)[0],
            CDF_1e6=measure(lambda: g.CDF_num(g.Q(y[:n], p), p), cuda)[0],
        ))
    out["parameterization"] = rows

    # where the data sit in the distribution
    rows = []
    g, p = torch_cases(device, torch.float64, "FMKL", [0.0, 1.0, -0.2, -0.1])
    for label, low, high in (("center: y in (0.4, 0.6)", 0.4, 0.6), ("all: y in (0, 1)", 0.0, 1.0), ("far tail: y < 1e-6", 1e-30, 1e-6)):
        y = low + (high - low) * torch.rand(n, dtype=torch.float64, device=device)
        x = g.Q(y, p)
        rows.append(dict(region=label, seconds=measure(lambda: g.CDF_num(x, p), cuda)[0]))
    out["data_region_1e6"] = rows

    # composition of the fits
    rows = []
    for n_fit in (10**4, 10**6, 10**7):
        x = sample(n_fit, device, torch.float32)
        g = gpu.GLD(PTYPE, device=device)
        sort = measure(lambda: torch.sort(x), cuda)[0]
        moments = measure(lambda: gpu.compute_sample_lmoments(x), cuda)[0]
        total = measure(lambda: g.fit_LMM(x), cuda)[0]
        rows.append(dict(n=n_fit, sort=sort, lmoments=moments, fit_LMM=total))
    out["fit_LMM_composition"] = rows

    # fit_ML: warm start, steps
    rows = []
    x = sample(10**5, device, torch.float32)
    g = gpu.GLD(PTYPE, device=device)
    for steps in (30, 60, 300):
        rows.append(dict(steps=steps, seconds=measure(lambda: g.fit_ML(x, steps=steps), cuda)[0]))
    out["fit_ML_steps_1e5"] = rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only up to N = 1e5")
    ap.add_argument("--out", default="benchmarks/results/benchmarks.json")
    args = ap.parse_args()
    sizes = [s for s in SIZES if not args.quick or s <= 10**5]
    meta = dict(
        version=__version__, torch=torch.__version__, numpy=np.__version__, python=platform.python_version(),
        cuda_device=torch.cuda.get_device_name(0) if GPU else None, cpu_threads=torch.get_num_threads(),
        date=time.strftime("%Y-%m-%d"), param=PARAM, type=PTYPE,
    )
    rows, extra = [], {}
    sweep(sizes, rows)
    factors(extra)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(meta=meta, rows=rows, factors=extra), indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
