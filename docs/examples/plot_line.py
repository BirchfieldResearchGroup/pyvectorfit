"""
Line model
==========

From per-metre line constants to a passive universal line model, ready for an EMT program.
"""

# %%
# A 100 km three-phase line with two shield wires over 100 Ω·m earth. Its per-metre series
# impedance Z and shunt admittance Y include frequency-dependent earth return and skin effect,
# sampled from 0.1 Hz to 1 MHz.

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter

import pyvectorfit as vf
from benchmarks.cases import CASES

case = CASES["horizontal"]
f = np.geomspace(0.1, 1e6, 141)
z, y = case.system.zy(2j * np.pi * f)

line, report = vf.fit_line(f, z, y, case.length)
delays = ", ".join(f"{1e6 * tau:.1f}" for tau in line.h.delays)
print(line.yc)
print(line.h)
print(f"modes {report.groups} travel with delays of {delays} µs")

# %%
# The fit samples Z and Y only; passivity is checked, and enforced where needed, on a grid
# ten times denser. This line is passive as fitted.

fv = np.geomspace(0.1, 1e6, 1401)
line, correction = vf.enforce_passivity(line, fv)
print(f"passive: {correction.after.passive}, corrections: {correction.iterations}")

# %%
# The two aerial modes share one delay and the ground mode takes its own. Against the exact
# line, the characteristic admittance, the propagation and the terminal admittance an EMT
# program sees all hold below 1 % over seven decades.

exact = vf.line_functions(*case.system.zy(2j * np.pi * fv), case.length)
yc, h = line.yc.response(fv), line.h.response(fv)
errors = {
    "Characteristic admittance": vf.relative_error(yc, exact.yc),
    "Propagation": vf.relative_error(h, exact.h),
    "Terminal admittance": vf.relative_error(
        vf.terminal_admittance(yc, h), vf.terminal_admittance(exact.yc, exact.h)
    ),
}

fig, (top, bottom) = plt.subplots(2, 1, sharex=True)
for k, (modes, tau) in enumerate(zip(report.groups, line.h.delays, strict=True)):
    modal = np.exp(-case.length * exact.gamma[:, list(modes)].real)
    top.loglog(fv, modal, color=f"C{k}")[0].set_label(f"τ = {1e6 * tau:.1f} µs")
top.set(title="Modal propagation |h| by delay group", ylim=(1e-3, 2))
top.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=2)

for label, error in errors.items():
    bottom.loglog(fv, error, label=label)
bottom.xaxis.set_major_formatter(EngFormatter("Hz"))
bottom.set(title="Relative error", xlabel="Frequency", ylim=(1e-7, 1e-1))
bottom.legend(loc="lower right", bbox_to_anchor=(1, 1), ncols=3)
plt.show()

# %%
# The model is plain data: poles, residues and delays. Save it as JSON for an EMT program and
# load it back unchanged.

text = json.dumps(line.to_dict())
loaded = vf.LineModel.from_dict(json.loads(text))
same = np.array_equal(loaded.h.response(fv), line.h.response(fv))
print(f"{len(text) / 1e3:.0f} kB of JSON; reloaded model identical: {same}")
