"""
Modal delays
============

Why :func:`~pyvectorfit.fit_line` searches the delay of each mode group, and what that buys.
"""

# %%
# Three single-core cables in flat formation, 10 km long. Their six modes travel in four
# delay groups, from the coaxial core-sheath modes to the heavily attenuated earth-return
# mode.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

import pyvectorfit as vf
from benchmarks.cases import CASES

case = CASES["cable_flat"]
f = np.geomspace(0.1, 1e6, 141)
z, y = case.system.zy(2j * np.pi * f)
options = {"yc_poles": case.yc_poles, "h_poles": case.h_poles}

line, report = vf.fit_line(f, z, y, case.length, **options)
delays = ", ".join(f"{1e6 * tau:.1f}" for tau in line.h.delays)
print(f"modes {report.groups} travel with delays of {delays} µs")

# %%
# Each group's modal propagation is fitted after removing a delay τ. The residual against τ
# has a narrow valley just below a causality cliff: past the true onset of the wave, the
# delay-removed function is no longer causal. For lightly damped modes the valley sits at the
# phase delay at the highest frequency; for the earth-return mode it lies 13 % above it,
# where no rule of thumb would look.

functions = vf.line_functions(z, y, case.length)
modal = np.exp(-case.length * functions.gamma)
phase = case.length * functions.gamma[-1].imag / (2 * np.pi * f[-1])
ratios = np.linspace(0.85, 1.25, 81)


def residual(h, delay):
    """Smallest residual of a delay-removed modal fit."""
    fitted = vf.fit(f, h, poles=case.h_poles, asymptote="none", delay=delay)
    return min(fitted.report.history)


scans = []
for modes in map(list, report.groups):
    scans.append([residual(modal[:, modes], r * phase[modes].min()) for r in ratios])
    valley = ratios[np.argmin(scans[-1])]
    print(f"modes {modes}: valley at {valley:.3f} times the phase delay")

# %%
# With the phase delays instead, the earth-return mode starts too early, and the terminal
# admittance an EMT program uses is off by up to 72 %. The searched delays keep it within
# 0.4 %.

fixed, _ = vf.fit_line(f, z, y, case.length, delays="phase", **options)
fv = np.geomspace(0.1, 1e6, 1401)
exact = vf.line_functions(*case.system.zy(2j * np.pi * fv), case.length)
reference = vf.terminal_admittance(exact.yc, exact.h)

fig, (top, bottom) = plt.subplots(2, 1)
for scan, tau in zip(scans, line.h.delays, strict=True):
    top.semilogy(ratios, scan, label=f"{1e6 * tau:.0f} µs")
top.axvline(1, color="k", alpha=0.3, lw=1)
top.set(title="Modal residual against delay", xlabel="Delay / phase delay at 1 MHz")
top.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=4)

for model, label in ((line, "Searched delays"), (fixed, "Phase delays")):
    terminal = vf.terminal_admittance(model.yc.response(fv), model.h.response(fv))
    error = vf.relative_error(terminal, reference)
    bottom.loglog(fv, error, lw=1.2, label=f"{label}, max {error.max():.1%}")
bottom.xaxis.set_major_formatter(EngFormatter("Hz"))
bottom.set(title="Terminal admittance error", xlabel="Frequency")
bottom.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2)
plt.show()
