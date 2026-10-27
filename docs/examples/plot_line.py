"""
Line model
==========

From a line in OpenLine's catalog to a passive universal line model, ready for an EMT program.
"""

# %%
# A 100 km, 230 kV H-frame line of Drake conductors over 100 Ω·m earth.
# `OpenLine <https://openline.readthedocs.io>`_ gives its per-metre series impedance Z and shunt
# admittance Y, with earth return and skin effect, from 1 Hz to 1 MHz. As in EMT programs, a
# small shunt conductance keeps the line passive at DC.

import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
from openline import Circuit, Earth, Line, Strung, conductors, electrical, structures
from openline import units as u

import pyvectorfit as vf

frame = structures.frame.TH_230
overhead = Line(
    structure=frame,
    height=u.m(22),
    span=u.m(300),
    circuits=(
        Circuit.uniform(
            conductors.astm.b232.DRAKE,
            at=dict(zip("ABC", frame.phases, strict=True)),
            strung=Strung(u.kn(28), u.celsius(15)),
            insulator_drop=u.m(2.5),
        ),
    ),
    shields=(),
    earth=Earth(u.ohm_m(100), 10),
)
length = 100e3  # m


def constants(f):
    """Per-metre Z and Y at frequencies f, with 1e-11 S/m of shunt conductance."""
    sampled = [
        electrical.parameters(overhead, temperature=u.celsius(50), frequency=u.hz(x)) for x in f
    ]
    z = np.array([p.impedance.to(u.ohm_per_m) for p in sampled])
    y = np.array([p.admittance.to(u.s_per_m) for p in sampled])
    return z, y + 1e-11 * np.eye(3)


f = np.geomspace(1, 1e6, 121)
line, report = vf.fit_line(f, *constants(f), length)
delays = ", ".join(f"{1e6 * tau:.1f}" for tau in line.h.delays)
print(line.yc)
print(line.h)
print(f"modes {report.groups} travel with delays of {delays} µs")

# %%
# The fit samples Z and Y only; passivity is checked, and enforced where needed, on a grid
# ten times denser. This line is passive as fitted.

fv = np.geomspace(1, 1e6, 1201)
line, correction = vf.enforce_passivity(line, fv)
print(f"passive: {correction.after.passive}, corrections: {correction.iterations}")

# %%
# The two aerial modes share one delay and the ground mode takes its own. Against the exact
# line, the characteristic admittance, the propagation and the terminal admittance an EMT
# program sees all hold below 1 % over six decades.

exact = vf.line_functions(*constants(fv), length)
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
    modal = np.exp(-length * exact.gamma[:, list(modes)].real)
    top.loglog(fv, modal, color=f"C{k}")[0].set_label(f"τ = {1e6 * tau:.1f} µs")
top.set(title="Modal propagation |h| by delay group", ylim=(1e-3, 2))
top.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=2)

for label, error in errors.items():
    bottom.loglog(fv, error, label=label)
bottom.xaxis.set_major_formatter(EngFormatter("Hz"))
bottom.set(title="Relative error", xlabel="Frequency", ylim=(1e-7, 1e-1))
bottom.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=3)
plt.show()

# %%
# The model is plain data: poles, residues and delays. Save it as JSON for an EMT program and
# load it back unchanged.

text = json.dumps(line.to_dict())
loaded = vf.LineModel.from_dict(json.loads(text))
same = np.array_equal(loaded.h.response(fv), line.h.response(fv))
print(f"{len(text) / 1e3:.0f} kB of JSON; reloaded model identical: {same}")
