"""
Passivity
=========

Find where a fitted cable model would generate energy, and remove it with the smallest change.
"""

# %%
# Three single-core cables in trefoil, 5 km long. The fitted characteristic admittance is
# passive on its own, yet the terminal admittance built from it and the propagation is not.
# A passive network never delivers net energy, so an EMT simulation of a non-passive model
# can grow without bound.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

import pyvectorfit as vf
from benchmarks.cases import CASES

case = CASES["cable_trefoil"]
f = np.geomspace(0.1, 1e6, 141)
options = {"yc_poles": case.yc_poles, "h_poles": case.h_poles}
fitted, _ = vf.fit_line(f, *case.system.zy(2j * np.pi * f), case.length, **options)

fv = np.geomspace(0.1, 1e6, 1401)
before = vf.check_passivity(fitted, fv)
bands = ", ".join(f"{lo:.0f} to {hi:.0f} Hz" for lo, hi in before.bands)
print(f"Yc passive: {vf.check_passivity(fitted.yc).passive}")
print(f"line passive: {before.passive}, violated from {bands}")

# %%
# :func:`~pyvectorfit.enforce_passivity` perturbs the residues and constant term of Yc,
# keeping its poles and the whole propagation function, until Yc and the terminal admittance
# are passive. The change is the smallest in a weighted least-squares sense.

line, correction = vf.enforce_passivity(fitted, fv)
after = correction.after
print(f"line passive: {after.passive}, smallest eigenvalue {after.margin:.1e} S")
print(f"{correction.iterations} iterations, largest change {correction.change:.1e}")

# %%
# Passivity means the Hermitian part of the terminal admittance, its conductance matrix, has
# no negative eigenvalue. Its smallest eigenvalue dips below zero in two narrow bands (shaded);
# after the correction it does not, while Yc moves by about one part in ten thousand.


def conductance(model, f):
    """Smallest eigenvalue of the Hermitian part of the terminal admittance, in S."""
    y = vf.terminal_admittance(model.yc.response(f), model.h.response(f))
    return np.linalg.eigvalsh((y + y.conj().swapaxes(1, 2)) / 2)[:, 0]


band = np.linspace(50, 1200, 801)
fig, (top, bottom) = plt.subplots(2, 1)
for lo, hi in before.bands:
    top.axvspan(lo, hi, color="C1", alpha=0.12, lw=0)
top.plot(band, 1e6 * conductance(fitted, band), color="C1", label="Fitted")
top.plot(band, 1e6 * conductance(line, band), color="C0", label="Passive")
top.axhline(0, color="k", alpha=0.4, lw=0.8)
top.xaxis.set_major_formatter(EngFormatter("Hz"))
top.set(title="Smallest conductance (µS)", xlabel="Frequency")
top.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2)

change = vf.relative_error(line.yc.response(fv), fitted.yc.response(fv))
bottom.loglog(fv, change, color="C0")
bottom.xaxis.set_major_formatter(EngFormatter("Hz"))
bottom.set(title="Relative change of Yc", xlabel="Frequency")
plt.show()
