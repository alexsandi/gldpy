"""CPU (NumPy/SciPy) vs GPU (PyTorch) benchmark. Usage: uv run python benchmarks/bench_gpu.py"""
import json
import pathlib
import platform
import time

import numpy as np
import torch

from gldpy_gpu import GLD as CpuGLD
from gldpy_gpu import gpu

P = {"FMKL": [0.0, 1.0, 0.2, 0.3], "RS": [0.0, 0.2, 0.1, 0.15], "VSL": [0.0, 1.0, 0.4, 0.3]}
SIZES = [10_000, 100_000, 1_000_000, 10_000_000]
CDF_SIZES = [1_000, 10_000, 100_000]  # CPU root-finding is slow


def timeit(fn, repeat=3, cuda=False):
    fn()
    best = float("inf")
    for _ in range(repeat):
        if cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if cuda:
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best


def main():
    assert torch.cuda.is_available(), "CUDA device required"
    dev = torch.device("cuda")
    rows = []

    def record(op, ptype, n, dtype, cpu_s, gpu_s):
        rows.append(dict(op=op, ptype=ptype, n=n, dtype=dtype, cpu_s=cpu_s, gpu_s=gpu_s, speedup=cpu_s / gpu_s))
        print(f"{op:8s} {ptype:5s} n={n:>10,d} {dtype:7s} cpu={cpu_s*1e3:10.2f} ms  gpu={gpu_s*1e3:9.2f} ms  x{cpu_s/gpu_s:7.1f}")

    for ptype, p in P.items():
        cpu, g = CpuGLD(ptype), gpu.GLD(ptype, device=dev)
        for dtype in (torch.float32, torch.float64):
            pt = torch.tensor(p, dtype=dtype, device=dev)
            for n in SIZES:
                u = np.random.default_rng(0).random(n)
                ut = torch.as_tensor(u, dtype=dtype, device=dev)
                record("Q", ptype, n, str(dtype)[6:], timeit(lambda: cpu.Q(u, np.array(p)), 2),
                       timeit(lambda: g.Q(ut, pt), cuda=True))
            for n in CDF_SIZES:
                x = cpu.rand(np.array(p), size=n, random_state=1)
                xt = torch.as_tensor(x, dtype=dtype, device=dev)
                record("CDF_num", ptype, n, str(dtype)[6:], timeit(lambda: cpu.CDF_num(x, np.array(p)), 1),
                       timeit(lambda: g.CDF_num(xt, pt), cuda=True))
        # fits (float32 data, 100k samples)
        data = cpu.rand(np.array(p), size=100_000, random_state=2)
        dt = torch.as_tensor(data, dtype=torch.float32, device=dev)
        record("fit_LMM", ptype, len(data), "float32",
               timeit(lambda: cpu.fit_LMM(data, initial_guess=np.array(p[2:]) if ptype != "VSL" else None,
                                          disp_optimizer=False, disp_fit=False), 1),
               timeit(lambda: g.fit_LMM(dt), cuda=True))

    out = pathlib.Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    meta = dict(gpu=torch.cuda.get_device_name(0), torch=torch.__version__, python=platform.python_version())
    (out / "latest.json").write_text(json.dumps(dict(meta=meta, rows=rows), indent=1))
    print(meta)


if __name__ == "__main__":
    main()
