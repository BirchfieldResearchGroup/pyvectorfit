"""
EMT simulation
==============

Energize the fitted line in a time-stepping simulation and compare it with the exact line.
"""

# %%
# The 100 km three-phase line again, fitted and made passive.

import matplotlib.pyplot as plt
import numpy as np

import pyvectorfit as vf
from benchmarks import transient
from benchmarks.cases import CASES

case = CASES["horizontal"]
f = np.geomspace(0.1, 1e6, 141)
line, _ = vf.fit_line(f, *case.system.zy(2j * np.pi * f), case.length)
line, _ = vf.enforce_passivity(line, np.geomspace(0.1, 1e6, 1401))

# %%
# A source :math:`1 - e^{-t / 10\,\mu s}` behind 10 Ω energizes phase 1; the far end is open.
# ``benchmarks/transient.py`` steps the line by the method of characteristics: each end is a
# Norton equivalent whose history currents come from :func:`~pyvectorfit.recursive_convolution`
# of Yc and H, exact for inputs that are linear over each 1 µs step. The reference inverts the
# exact line's response with the numerical Laplace transform.

dt, steps, rise = 1e-6, 4000, 10e-6
ends = transient.terminations(case.system.conductors)
emt = transient.simulate(line, dt, steps, rise, *ends)

s = transient.frequencies(dt, steps)
yc, h, _ = vf.line_functions(*case.system.zy(s), case.length)
exact = transient.invert(transient.response(yc, h, s, rise, *ends), s, dt)
peak = np.abs(exact).max()
print(f"largest difference {np.abs(emt - exact).max() / peak:.2%} of the peak voltage")

# %%
# The wave reaches the open end after 0.34 ms and doubles, and the induced voltages on phases
# 2 and 3 follow every reflection. Pale bands are the exact response and lines the
# simulation, which stays on it through every wavefront.

ms = 1e3 * dt * np.arange(steps)
fig, (top, bottom) = plt.subplots(2, 1)
for ax, window in ((top, slice(None)), (bottom, slice(320, 400))):
    for k in range(3):
        ax.plot(ms[window], exact[window, k], color=f"C{k}", alpha=0.25, lw=5)
        ax.plot(ms[window], emt[window, k], color=f"C{k}", lw=1.2, label=f"Phase {k + 1}")
top.set(title="Receiving-end voltage (per unit)", xlabel="Time (ms)")
top.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=3)
bottom.set(title="First arrival", xlabel="Time (ms)")
plt.show()
