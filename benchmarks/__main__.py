"""Benchmark suite: `python -m benchmarks` writes figures/*.svg and results.json."""

import json
import platform
import time
from dataclasses import replace
from pathlib import Path

import matplotlib as mpl
import numpy as np
import scipy

import pyvectorfit as vf
from pyvectorfit._line import _sensitivity

from . import transient
from .cases import CASES, HORIZONTAL, Case
from .style import AXIS, MUTED, SERIES, figure, legend, save

RESULTS = Path(__file__).parent / "results.json"
ORDERS = range(4, 21, 2)
ERRORS = {"yc": "Yc", "h": "H", "terminal": "Terminal admittance"}
TRANSIENTS = {"horizontal": (1e-6, 4000, 10e-6), "cable_flat": (0.25e-6, 6000, 2e-6)}
SWEEPS = {
    "length_km": ("Line length (km)", (1, 3, 10, 30, 100, 300)),
    "earth_ohm_m": ("Earth resistivity (ohm m)", (10, 100, 1000, 5000)),
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
            "fitted": fitted,
            "line": line,
            "report": report,
            "correction": correction,
            "fv": fv,
            "errors": errors(line, case, fv),
            "seconds": {"fit": seconds, "passivity": time.perf_counter() - start},
        }
    return results


def plot_modes(results):
    fig, axes = figure(2, 3, height=2.3, sharex=True, sharey=True)
    for ax, item in zip(axes.flat, results.values(), strict=True):
        case = item["case"]
        gamma = vf.line_functions(*case.system.zy(2j * np.pi * item["fv"]), case.length).gamma
        for index, modes in enumerate(item["report"].groups):
            for mode in modes:
                label = f"Delay group {index + 1}" if mode == modes[0] else None
                magnitude = np.exp(-case.length * gamma[:, mode].real)
                ax.loglog(item["fv"], magnitude, color=SERIES[index], label=label)
        ax.set(title=case.title, ylim=(1e-4, 2))
    _labels(axes, "Modal propagation |h|")
    widest = max(axes.flat, key=lambda ax: len(ax.get_legend_handles_labels()[1]))
    legend(fig, widest)
    save(fig, "modes")


def plot_accuracy(results):
    fig, axes = figure(2, 3, height=2.3, sharex=True, sharey=True)
    for ax, item in zip(axes.flat, results.values(), strict=True):
        for key, label in ERRORS.items():
            ax.loglog(item["fv"], item["errors"][key], label=label)
        ax.set(title=item["case"].title, ylim=(1e-7, 1e-1))
    _labels(axes, "Relative error")
    legend(fig, axes[0, 0])
    save(fig, "accuracy")


def plot_iterations(results):
    fig, axes = figure(1, 1, width=5.2, height=3.0)
    ax = axes[0, 0]
    for item in results.values():
        history = item["report"].yc.history
        ax.semilogy(range(len(history)), history, marker="o", label=item["case"].title)
    ax.set(title="Vector fitting of Yc", xlabel="Relocation", ylabel="Weighted relative residual")
    legend(fig, ax, columns=2)
    save(fig, "iterations")


def delay_landscape(item, samples=81):
    """Modal residual against the delay of each group, as minimized by the delay search."""
    case = item["case"]
    f, _ = grids(case.band)
    gamma = vf.line_functions(*case.system.zy(2j * np.pi * f), case.length).gamma
    phase = case.length * gamma[-1].imag / (2 * np.pi * f[-1])
    ratio = np.linspace(0.85, 1.25, samples)
    curves = []
    for modes, chosen in zip(map(list, item["report"].groups), item["line"].h.delays, strict=True):
        modal = np.exp(-case.length * gamma[:, modes])
        tau = phase[modes].min()
        residual = [_residual(f, modal, case.h_poles, delay) for delay in [*ratio * tau, chosen]]
        curves.append((np.array(residual[:-1]), chosen / tau, residual[-1]))
    return ratio, curves


def _residual(f, modal, poles, delay):
    """Objective of the delay search: weighted residual of the delay-removed modal fit."""
    options = {"poles": poles, "weights": _sensitivity(modal), "asymptote": "none"}
    return min(vf.fit(f, modal, delay=delay, **options).report.history)


def plot_delays(item, ratio, curves):
    fig, axes = figure(1, 1, width=5.2, height=3.0)
    ax = axes[0, 0]
    for index, (scan, chosen, best) in enumerate(curves):
        ax.semilogy(ratio, scan, color=SERIES[index], label=f"Delay group {index + 1}")
        ax.semilogy(chosen, best, "o", color=SERIES[index])
    ax.axvline(1.0, color=MUTED, linewidth=0.75, label="Phase delay at the highest frequency")
    ax.set(
        title=f"{item['case'].title}: delay search",
        xlabel="Delay relative to the phase delay at the highest frequency",
        ylabel="Weighted modal residual",
    )
    legend(fig, ax)
    save(fig, "delays")


def orders():
    """Maximum errors of Yc and of the terminal admittance against the number of poles."""
    table = {}
    for name, case in CASES.items():
        f, fv = grids(case.band)
        samples = vf.line_functions(*case.system.zy(2j * np.pi * f), case.length).yc
        exact = vf.line_functions(*case.system.zy(2j * np.pi * fv), case.length).yc
        yc, terminal = [], []
        for order in ORDERS:
            model = vf.fit(f, samples, poles=order, symmetric=True).model
            yc.append(vf.relative_error(model.response(fv), exact).max())
            terminal.append(errors(fit(case, h_poles=order)[0], case, fv)["terminal"].max())
        table[name] = {"yc": yc, "terminal": terminal}
    return table


def plot_orders(table):
    fig, axes = figure(1, 2, height=2.8, sharey=True)
    for name, row in table.items():
        axes[0, 0].semilogy(ORDERS, row["yc"], marker="o", label=CASES[name].title)
        axes[0, 1].semilogy(ORDERS, row["terminal"], marker="o")
    axes[0, 0].set(title="Yc", xlabel="Poles", ylabel="Maximum relative error")
    axes[0, 1].set(title="Terminal admittance", xlabel="H poles per delay group")
    legend(fig, axes[0, 0], columns=3)
    save(fig, "orders")


def plot_passivity(item):
    f = np.geomspace(20.0, 1.5e3, 601)
    fig, axes = figure(1, 2, height=2.8)
    for line, label, color in (
        (item["fitted"], "Fitted", SERIES[1]),
        (item["line"], "Passive", SERIES[0]),
    ):
        margin = [vf.check_passivity(line, [frequency]).margin for frequency in f]
        axes[0, 0].semilogx(f, 1e6 * np.array(margin), color=color, label=label)
    axes[0, 0].axhline(0.0, color=AXIS, linewidth=0.75)
    axes[0, 0].set(
        title=f"{item['case'].title}: terminal admittance",
        xlabel="Frequency (Hz)",
        ylabel="Smallest conductance (uS)",
    )
    fv = item["fv"]
    change = vf.relative_error(item["line"].yc.response(fv), item["fitted"].yc.response(fv))
    axes[0, 1].loglog(fv, change, color=SERIES[0])
    axes[0, 1].set(title="Correction of Yc", xlabel="Frequency (Hz)", ylabel="Relative change")
    legend(fig, axes[0, 0])
    save(fig, "passivity")


def transients(results):
    """Receiving-end energization of the exact line, the fitted model and its EMT simulation."""
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
            "t": dt * np.arange(steps),
            "exact": exact,
            "emt": emt,
            "errors": {
                "model": np.abs(model[:, 0] - exact).max() / peak,
                "emt": np.abs(emt - exact).max() / peak,
                "time_step": np.abs(emt - model[:, 0]).max() / peak,
            },
        }
    return table


def plot_transients(table):
    fig, axes = figure(2, len(table), height=2.2, sharex="col")
    for column, (name, row) in enumerate(table.items()):
        ms = 1e3 * row["t"]
        axes[0, column].plot(ms, row["exact"], linewidth=4, alpha=0.35, label="Exact line")
        axes[0, column].plot(ms, row["emt"], color=SERIES[1], label="Recursive convolution")
        difference = 100 * (row["emt"] - row["exact"]) / np.abs(row["exact"]).max()
        axes[1, column].plot(ms, difference, color=SERIES[1])
        axes[0, column].set_title(CASES[name].title)
        axes[1, column].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("Receiving end (per unit)")
    axes[1, 0].set_ylabel("Difference (% of peak)")
    legend(fig, axes[0, 0])
    save(fig, "transients")


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
        for key, label in ERRORS.items():
            ax.loglog(list(rows), [row[key] for row in rows.values()], marker="o", label=label)
        ax.set_xticks(list(rows), [str(value) for value in rows], minor=False)
        ax.minorticks_off()
        ax.set_xlabel(SWEEPS[study][0])
    axes[0, 0].set_ylabel("Maximum relative error")
    legend(fig, axes[0, 0])
    save(fig, "sweeps")


def _labels(axes, ylabel):
    for ax in axes[-1]:
        ax.set_xlabel("Frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)


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
    results = lines()
    plot_modes(results)
    plot_accuracy(results)
    plot_iterations(results)
    plot_delays(results["cable_flat"], *delay_landscape(results["cable_flat"]))
    plot_passivity(results["cable_trefoil"])
    order_table = orders()
    plot_orders(order_table)
    transient_table = transients(results)
    plot_transients(transient_table)
    sweep_table = sweeps()
    plot_sweeps(sweep_table)
    summary = {
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
        "transients": {name: row["errors"] for name, row in transient_table.items()},
        "sweeps": sweep_table,
    }
    RESULTS.write_text(json.dumps(_rounded(summary), indent=2) + "\n")


if __name__ == "__main__":
    main()
