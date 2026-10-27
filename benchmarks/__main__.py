"""Benchmark suite: `python -m benchmarks` writes results.json and the docs' figures and table."""

import csv
import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import numpy as np
import scipy
from matplotlib.ticker import EngFormatter

import pyvectorfit as vf

from . import banner, transient
from .cases import CASES, HORIZONTAL, Case
from .style import figure, legend, save

RESULTS = Path(__file__).parent / "results.json"
DOCS = Path(__file__).parents[1] / "docs" / "_static" / "benchmarks"
ORDERS = range(4, 21, 2)
ERRORS = {"yc": "Yc", "h": "H", "terminal": "Terminal admittance"}
TRANSIENTS = {"horizontal": (1e-6, 4000, 10e-6), "cable_flat": (0.25e-6, 6000, 2e-6)}
SWEEPS = {
    "length_km": ("Line length (km)", (1, 3, 10, 30, 100, 300)),
    "earth_ohm_m": ("Earth resistivity (Ω·m)", (10, 100, 1000, 5000)),
    "per_decade": ("Samples per decade", (5, 10, 20, 40)),
}


def grids(band, per_decade=20):
    """Fitting frequencies, and validation frequencies ten times denser that avoid them."""
    f = np.geomspace(*band, round(np.log10(band[1] / band[0]) * per_decade) + 1)
    dense = np.geomspace(*band, 10 * (len(f) - 1) + 1)
    return f, np.sqrt(dense[:-1] * dense[1:])


def fit(case, per_decade=20, **options):
    f, fv = grids(case.band, per_decade)
    options = {"yc_poles": case.yc_poles, "h_poles": case.h_poles, **options}
    start = time.perf_counter()
    line, report = vf.fit_line(f, *case.system.zy(2j * np.pi * f), case.length, **options)
    return line, report, fv, time.perf_counter() - start


def errors(line, case, fv):
    """Relative error per frequency of Yc, H and the terminal admittance."""
    exact = vf.line_functions(*case.system.zy(2j * np.pi * fv), case.length)
    yc, h = line.yc.response(fv), line.h.response(fv)
    terminal = vf.terminal_admittance(yc, h)
    return {
        "yc": vf.relative_error(yc, exact.yc),
        "h": vf.relative_error(h, exact.h),
        "terminal": vf.relative_error(terminal, vf.terminal_admittance(exact.yc, exact.h)),
    }


def lines():
    """Fit, make passive and validate every case."""
    results = {}
    for name, case in CASES.items():
        fitted, report, fv, seconds = fit(case)
        start = time.perf_counter()
        line, correction = vf.enforce_passivity(fitted, fv)
        results[name] = {
            "case": case,
            "line": line,
            "report": report,
            "correction": correction,
            "fv": fv,
            "errors": errors(line, case, fv),
            "seconds": {"fit": seconds, "passivity": time.perf_counter() - start},
        }
    return results


def plot_accuracy(results):
    fig, axes = figure(2, 3, height=2.0, sharex=True, sharey=True)
    for ax, item in zip(axes.flat, results.values(), strict=True):
        for key, label in ERRORS.items():
            ax.loglog(item["fv"], item["errors"][key], lw=1.1, label=label)
        ax.set(title=item["case"].title, ylim=(1e-7, 1e-1), xticks=(1, 1e3, 1e6))
    axes[0, 0].xaxis.set_major_formatter(EngFormatter("Hz"))
    for ax in axes[:, 0]:
        ax.set_ylabel("Relative error")
    legend(fig, axes[0, 0], columns=3)
    save(fig, DOCS / "accuracy")


def orders():
    """Maximum errors of Yc, H and the terminal admittance against the number of poles."""
    table = {}
    for name, case in CASES.items():
        f, fv = grids(case.band)
        samples = vf.line_functions(*case.system.zy(2j * np.pi * f), case.length).yc
        exact = vf.line_functions(*case.system.zy(2j * np.pi * fv), case.length).yc
        row = {key: [] for key in ERRORS}
        for order in ORDERS:
            model = vf.fit(f, samples, poles=order, symmetric=True).model
            row["yc"].append(vf.relative_error(model.response(fv), exact).max())
            line_errors = errors(fit(case, h_poles=order)[0], case, fv)
            row["h"].append(line_errors["h"].max())
            row["terminal"].append(line_errors["terminal"].max())
        table[name] = row
    return table


def plot_orders(table):
    """Errors against poles: Yc poles for Yc, H poles per delay group for H and the terminal."""
    fig, axes = figure(2, 3, height=2.0, sharex=True, sharey=True)
    for ax, (name, row) in zip(axes.flat, table.items(), strict=True):
        for key, label in ERRORS.items():
            ax.semilogy(ORDERS, row[key], label=label)
        ax.set(title=CASES[name].title, xticks=ORDERS[::2])
    for ax in axes[-1]:
        ax.set_xlabel("Poles")
    for ax in axes[:, 0]:
        ax.set_ylabel("Maximum relative error")
    legend(fig, axes[0, 0], columns=3)
    save(fig, DOCS / "orders")


def transients(results):
    """Largest receiving-end differences from the exact line, relative to its peak."""
    table = {}
    for name, (dt, steps, rise) in TRANSIENTS.items():
        case, line = CASES[name], results[name]["line"]
        ends = transient.terminations(case.system.conductors)
        s = transient.frequencies(dt, steps)
        yc, h, _ = vf.line_functions(*case.system.zy(s), case.length)
        exact = transient.invert(transient.response(yc, h, s, rise, *ends), s, dt)[:, 0]
        model = transient.invert(transient.response(line.yc(s), line.h(s), s, rise, *ends), s, dt)
        emt = transient.simulate(line, dt, steps, rise, *ends)[:, 0]
        peak = np.abs(exact).max()
        table[name] = {
            "model": np.abs(model[:, 0] - exact).max() / peak,
            "emt": np.abs(emt - exact).max() / peak,
            "time_step": np.abs(emt - model[:, 0]).max() / peak,
        }
    return table


def sweeps():
    """Maximum errors of the horizontal line against length, earth resistivity and sampling."""
    variants = {
        "length_km": lambda km: (Case("", HORIZONTAL, 1e3 * km), 20),
        "earth_ohm_m": lambda rho: (Case("", replace(HORIZONTAL, earth=rho), 100e3), 20),
        "per_decade": lambda n: (Case("", HORIZONTAL, 100e3), n),
    }
    table = {}
    for study, (_, values) in SWEEPS.items():
        table[study] = {}
        for value in values:
            case, per_decade = variants[study](value)
            line, _, fv, _ = fit(case, per_decade)
            table[study][value] = {key: e.max() for key, e in errors(line, case, fv).items()}
    return table


def plot_sweeps(table):
    fig, axes = figure(1, 3, height=2.6, sharey=True)
    for ax, (study, rows) in zip(axes.flat, table.items(), strict=True):
        values = list(rows)
        for key, label in ERRORS.items():
            ax.loglog(values, [row[key] for row in rows.values()], marker="o", ms=5, label=label)
        ax.set_xticks(values, [str(value) for value in values], minor=False)
        ax.minorticks_off()
        ax.set_xlabel(SWEEPS[study][0])
    axes[0, 0].set_ylabel("Maximum relative error")
    legend(fig, axes[0, 0], columns=3)
    save(fig, DOCS / "sweeps")


def summary(results, transient_table):
    """The docs' table: one row per case."""
    rows = []
    for name, item in results.items():
        line, correction, emt = item["line"], item["correction"], transient_table.get(name)
        rows.append(
            {
                "Case": item["case"].title,
                "Poles Yc / H": f"{line.yc.order} / {line.h.order}",
                "Delays (µs)": ", ".join(f"{1e6 * tau:.0f}" for tau in line.h.delays),
                "Yc error": _percent(item["errors"]["yc"].max()),
                "H error": _percent(item["errors"]["h"].max()),
                "Terminal error": _percent(item["errors"]["terminal"].max()),
                "Passivity fix": _percent(correction.change) if correction.iterations else "none",
                "EMT error": _percent(emt["emt"]) if emt else "—",
            }
        )
    with (DOCS / "summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, rows[0], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _percent(value):
    """Two significant digits, trailing zeros kept: 0.30 %, 0.0047 %."""
    return f"{100 * value:#.2g}".rstrip(".") + " %"


def _rounded(value):
    """JSON-ready values with three significant digits, so reruns diff cleanly."""
    if isinstance(value, dict):
        return {str(k): _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(f"{value:.3g}")
    return value


def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    banner.write()
    results = lines()
    plot_accuracy(results)
    order_table = orders()
    plot_orders(order_table)
    transient_table = transients(results)
    sweep_table = sweeps()
    plot_sweeps(sweep_table)
    summary(results, transient_table)
    report = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": mpl.__version__,
            "pyvectorfit": vf.__version__,
        },
        "cases": {
            name: {
                "title": item["case"].title,
                "conductors": item["case"].system.conductors,
                "length_km": item["case"].length / 1e3,
                "poles": {"yc": item["line"].yc.order, "h": item["line"].h.order},
                "groups": item["report"].groups,
                "delays_us": (1e6 * item["line"].h.delays).tolist(),
                "max_error": {key: e.max() for key, e in item["errors"].items()},
                "passivity": {
                    "iterations": item["correction"].iterations,
                    "change": item["correction"].change,
                },
                "seconds": item["seconds"],
            }
            for name, item in results.items()
        },
        "orders": {"poles": list(ORDERS), **order_table},
        "transients": transient_table,
        "sweeps": sweep_table,
    }
    RESULTS.write_text(
        json.dumps(_rounded(report), indent=2) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
