"""Turn benchmarks/results/*.json into the Markdown tables of docs/. Usage: uv run python benchmarks/make_tables.py

Each table replaces the text between `<!-- table:NAME -->` and `<!-- /table -->` in the documents, so the
numbers in docs/validation.md and docs/benchmarks.md are never copied by hand.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "benchmarks" / "results"
DOCS = [ROOT / "docs" / "validation.md", ROOT / "docs" / "benchmarks.md"]


def sci(v):
    if v is None:
        return "fails"
    if v == 0:
        return "0"
    return f"{v:.1e}"


def duration(s):
    if s >= 10:
        return f"{s:.0f} s"
    if s >= 1:
        return f"{s:.1f} s"
    if s >= 0.1:
        return f"{s * 1e3:.0f} ms"
    if s >= 0.01:
        return f"{s * 1e3:.1f} ms"
    if s >= 1e-3:
        return f"{s * 1e3:.2f} ms"
    return f"{s * 1e6:.0f} µs"


def markdown(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def ks_table(results, ptype):
    methods = [("LMM", "original"), ("LMM", "corrected"), ("LMM", "torch_f64"), ("LMM", "torch_f32"),
               ("PM", "original"), ("PM", "corrected"), ("PM", "torch_f64"), ("PM", "torch_f32"),
               ("MM", "corrected"), ("ML", "torch_f64"), ("ML", "torch_f32")]
    names = dict(original="orig.", corrected="NumPy", torch_f64="f64", torch_f32="f32")
    index = {(r["distribution"], r["method"], r["backend"]): r for r in results if r["type"] == ptype}
    methods = [(m, b) for m, b in methods if any(k[1:] == (m, b) for k in index)]
    rows = []
    for dist in dict.fromkeys(r["distribution"] for r in results):
        rows.append([dist] + [sci(index[dist, m, b]["ks"]) for m, b in methods])
    return markdown(["Distribution"] + [f"{m} {names[b]}" for m, b in methods], rows)


def precision_table(results):
    rows = []
    for method in ("LMM", "PM", "ML"):
        diffs = []
        for r in results:
            if r["backend"] == "torch_f32" and r["method"] == method and r["valid"]:
                other = next(
                    x for x in results
                    if x["backend"] == "torch_f64" and x["method"] == method and x["type"] == r["type"] and x["distribution"] == r["distribution"]
                )
                if other["valid"]:
                    diffs.append((abs(r["ks"] - other["ks"]), f"{r['type']}, {r['distribution']}"))
        diffs.sort()
        rows.append([method, len(diffs), sci(diffs[len(diffs) // 2][0]), f"{sci(diffs[-1][0])} ({diffs[-1][1]})"])
    return markdown(["Method", "Cases", "Median |ΔKS|", "Largest |ΔKS|"], rows)


def tail_table(tails):
    rows = []
    for r in tails:
        rows.append([r["case"], f"{r['s']:.0e}", sci(r.get("original_lower")) if "original_lower" in r else "not run",
                     sci(r["numpy"]), sci(r["f64"]), "below floor" if r["f32"] is None else sci(r["f32"])])
    return markdown(["Case", "s", "Original (lower only)", "NumPy", "torch f64", "torch f32"], rows)


BACKENDS = [("original", "gldpy 0.2"), ("corrected", "NumPy"), ("torch_cpu_f32", "torch CPU f32"), ("torch_cpu_f64", "torch CPU f64"),
            ("cuda_f32", "CUDA f32"), ("cuda_f64", "CUDA f64")]


def sweep_table(rows, op):
    index = {(r["n"], r["backend"]): r for r in rows if r["op"] == op}
    sizes = sorted({r["n"] for r in rows if r["op"] == op})
    backends = [(b, label) for b, label in BACKENDS if any(k[1] == b for k in index)]
    out = []
    for n in sizes:
        cells = [f"{n:.0e}" if n >= 10**4 else str(n)]
        for b, _ in backends:
            cells.append(duration(index[n, b]["median"]) if (n, b) in index else "")
        out.append(cells)
    return markdown(["N"] + [label for _, label in backends], out)


def ratio(a, b):
    return "" if a is None or b is None else f"{a / b:.0f}x" if a / b >= 10 else f"{a / b:.1f}x"


def speedup_table(rows):
    """CUDA float32 against the alternatives, at the largest N each alternative ran, and the crossover."""
    out = []
    for op in dict.fromkeys(r["op"] for r in rows):
        index = {(r["n"], r["backend"]): r["median"] for r in rows if r["op"] == op}
        for reference, label in (("torch_cpu_f32", "torch CPU f32"), ("corrected", "NumPy")):
            common = sorted(n for n, b in index if b == reference and (n, "cuda_f32") in index)
            if not common:
                continue
            n = common[-1]
            crossover = next((m for m in common if index[m, "cuda_f32"] < index[m, reference]), None)
            out.append([op, label, f"{n:.0e}", duration(index[n, reference]), duration(index[n, "cuda_f32"]),
                        ratio(index[n, reference], index[n, "cuda_f32"]), f"{crossover:.0e}" if crossover else "not reached"])
    return markdown(["Operation", "Against", "At N", "Its time", "CUDA f32", "Speedup", "CUDA wins from N ="], out)


def memory_table(rows):
    out = []
    for op in ("Q", "rand", "CDF_num", "fit_LMM", "fit_PM"):
        index = {r["n"]: r for r in rows if r["op"] == op and r["backend"] == "cuda_f32"}
        n = max(index)
        bytes_per = index[n]["peak_bytes"] / n
        out.append([op, f"{n:.0e}", f"{index[n]['peak_bytes'] / 2**30:.2f} GiB", f"{bytes_per:.0f}", f"{7 * 2**30 / bytes_per:.1e}"])
    return markdown(["Operation", "N", "Peak GPU memory, float32", "Bytes per element", "Largest N in 7 GiB (extrapolated)"], out)


def factor_tables(f):
    tables = {}
    tables["launch"] = markdown(["Device", "Trivial op on 1 element"], [
        ["CUDA", duration(f["launch_seconds"]["median"])], ["CPU", duration(f["launch_seconds_cpu"]["median"])]])
    if "transfer" in f:
        tables["transfer"] = markdown(["N, float32", "CPU to GPU", "GPU to CPU"], [
            [f"{r['n']:.0e}", duration(r["to_gpu"]), duration(r["to_cpu"])] for r in f["transfer"]])
    tables["primitives"] = markdown(["Device", "dtype", "add", "mul", "exp", "log", "expm1", "sort"], [
        [r["device"], r["dtype"]] + [duration(r[k]) for k in ("add", "mul", "exp", "log", "expm1", "sort")] for r in f["primitives_1e7"]])
    base = f["cpu_threads_Q_1e7"][0]["Q"]
    tables["threads"] = markdown(["Threads", "Q, N = 1e7", "Scaling"], [
        [r["threads"], duration(r["Q"]), f"{base / r['Q']:.1f}x"] for r in f["cpu_threads_Q_1e7"]])
    xtols = sorted({r["xtol"] for r in f["solver_xtol_1e6"]}, reverse=True)
    cell = {(r["dtype"], r["xtol"]): r["seconds"] for r in f["solver_xtol_1e6"]}
    tables["xtol"] = markdown(["xtol", "float32", "float64"], [
        [f"{x:.0e}", duration(cell["float32", x]), duration(cell["float64", x])] for x in xtols])
    tables["start"] = markdown(["dtype", "Table start", "Start at t = 0", "Gain"], [
        [r["dtype"], duration(r["table_start"]), duration(r["cold_start"]), ratio(r["cold_start"], r["table_start"])] for r in f["solver_start_1e6"]])
    tables["parameterization"] = markdown(["Type", "Shape", "Q, 1e7", "PDF_Q, 1e7", "Q + CDF_num, 1e6"], [
        [r["type"], str(tuple(r["shape"])), duration(r["Q"]), duration(r["PDF_Q"]), duration(r["CDF_1e6"])] for r in f["parameterization"]])
    tables["region"] = markdown(["Where the 1e6 points lie", "CDF_num, float64"], [
        [r["region"], duration(r["seconds"])] for r in f["data_region_1e6"]])
    tables["fit_composition"] = markdown(["N", "Sort", "L-moments, including sort", "fit_LMM, total"], [
        [f"{r['n']:.0e}", duration(r["sort"]), duration(r["lmoments"]), duration(r["fit_LMM"])] for r in f["fit_LMM_composition"]])
    tables["ml_steps"] = markdown(["steps", "fit_ML, N = 1e5, float32"], [
        [r["steps"], duration(r["seconds"])] for r in f["fit_ML_steps_1e5"]])
    return tables


def main():
    tables = {}
    analytic = RESULTS / "analytic.json"
    if analytic.exists():
        a = json.loads(analytic.read_text())
        for ptype in ("RS", "FMKL", "VSL"):
            tables[f"ks_{ptype}"] = ks_table(a["results"], ptype)
        tables["precision"] = precision_table(a["results"])
        tables["tails"] = tail_table(a["tails"])
    bench = RESULTS / "benchmarks.json"
    if bench.exists():
        b = json.loads(bench.read_text())
        for op in dict.fromkeys(r["op"] for r in b["rows"]):
            tables[f"sweep_{op}"] = sweep_table(b["rows"], op)
        tables["speedup"] = speedup_table(b["rows"])
        tables["memory"] = memory_table(b["rows"])
        tables.update({f"factor_{k}": v for k, v in factor_tables(b["factors"]).items()})
        m = b["meta"]
        tables["meta"] = markdown(["Item", "Value"], [
            ["gldpy-gpu", m["version"]], ["PyTorch", m["torch"]], ["NumPy", m["numpy"]], ["Python", m["python"]],
            ["GPU", m["cuda_device"]], ["CPU threads used by torch", m["cpu_threads"]], ["Date", m["date"]],
        ])
    for doc in DOCS:
        if not doc.exists():
            continue
        text = doc.read_text()
        for name in re.findall(r"<!-- table:(\w+) -->", text):
            if name not in tables:
                raise SystemExit(f"{doc.name}: no data for table {name}")
            text = re.sub(
                rf"(<!-- table:{name} -->\n).*?(<!-- /table -->)",
                lambda match: match.group(1) + tables[name] + "\n" + match.group(2),
                text, flags=re.S,
            )
        doc.write_text(text)
        print("updated", doc.relative_to(ROOT))


if __name__ == "__main__":
    main()
