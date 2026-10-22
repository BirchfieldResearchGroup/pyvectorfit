"""
Rational fitting
================

Fit a resonant frequency response with one set of stable poles, and see how the number of
poles decides what the model can represent.
"""

# %%
# The input admittance of a 50 km line open at its far end, :math:`Y_c \tanh(\gamma \ell)`,
# resonates every 2.9 kHz. Skin effect makes the conductor resistance grow with the square
# root of frequency, so no rational function matches it exactly.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

import pyvectorfit as vf


def admittance(f):
    """Input admittance in S of a 50 km open-ended line at frequencies f in Hz."""
    s = 2j * np.pi * f
    z = 2e-4 * np.sqrt(1 + s / (2 * np.pi * 5e3)) + s * 1.3e-6  # ohm/m
    y = 1e-11 + s * 9e-12  # S/m
    return np.sqrt(y / z) * np.tanh(50e3 * np.sqrt(z * y))


f = np.geomspace(1, 2e4, 400)
model, report = vf.fit(f, admittance(f), poles=24)
relocations = len(report.history) - 1
print(model)
print(f"{report.status}, {relocations} relocations, rms error {report.rms_error:.1e}")

# %%
# Each resonance takes one complex pole pair. With 24 poles every peak has its pair and the
# model holds on a validation grid ten times denser; with fewer, the missing peaks set the
# error.

fv = np.linspace(1, 2e4, 4001)
exact = admittance(fv)
pairs = model.poles.imag[model.poles.imag > 0] / (2 * np.pi)
pairs = pairs[pairs < fv[-1]]

fig, (top, bottom) = plt.subplots(2, 1, sharex=True, height_ratios=(3, 2))
top.semilogy(fv, abs(exact), color="C0", alpha=0.25, lw=5, label="Data")
top.semilogy(fv, abs(model.response(fv)), color="C0", lw=1.2, label="Model, 24 poles")
top.plot(pairs, abs(model.response(pairs)), "o", color="C1", ms=5, label="Pole pairs")
top.yaxis.set_major_formatter(EngFormatter("S"))
top.set(title="Input admittance", ylim=(1e-4, 1))
top.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=3)

for poles, alpha in ((8, 0.3), (16, 0.6), (24, 1.0)):
    m = vf.fit(f, admittance(f), poles=poles).model
    error = vf.relative_error(m.response(fv), exact)
    bottom.semilogy(fv, error, color="C0", alpha=alpha, lw=1.2, label=f"{poles} poles")
bottom.xaxis.set_major_formatter(EngFormatter("Hz"))
bottom.set(title="Relative error", xlabel="Frequency", ylim=(1e-12, 10))
bottom.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=3)
plt.show()
