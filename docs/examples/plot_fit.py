"""
Rational fitting
================

Fit a line's characteristic admittance with one set of real poles, the first step of a line
model, and see how the number of poles sets the accuracy.
"""

# %%
# A 60 km, 380 kV line of twin Drake bundles on a lattice tower, over 100 Ω·m earth.
# `OpenLine <https://openline.readthedocs.io>`_ gives its per-metre line constants at 20
# frequencies per decade from 0.1 Hz to 1 MHz, and the midpoints between them are kept to
# validate. As in EMT programs, a small shunt conductance keeps the line passive at DC.

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import EngFormatter
from openline import Bundle, Circuit, Earth, Line, Strung, conductors, electrical, structures
from openline import units as u

import pyvectorfit as vf

tower = structures.tower.D_2_D_2018_3_T1
west = [attachment for attachment in tower.phases if attachment.point.across < u.m(0)]
overhead = Line(
    structure=tower,
    height=u.m(32),
    span=u.m(400),
    circuits=(
        Circuit.uniform(
            Bundle(conductor=conductors.astm.b232.DRAKE, count=2, spacing=u.m(0.4)),
            at=dict(zip("ABC", west, strict=True)),
            strung=Strung(tension=u.kn(25), temperature=u.celsius(15)),
        ),
    ),
    shields=(),
    earth=Earth(resistivity=u.ohm_m(100), relative_permittivity=10),
)


def constants(f):
    """Per-metre Z and Y at frequencies f, with 1e-11 S/m of shunt conductance."""
    sampled = [
        electrical.parameters(overhead, temperature=u.celsius(50), frequency=u.hz(x)) for x in f
    ]
    z = np.array([p.impedance.to(u.ohm_per_m) for p in sampled])
    y = np.array([p.admittance.to(u.s_per_m) for p in sampled])
    return z, y + 1e-11 * np.eye(3)


grid = np.geomspace(0.1, 1e6, 281)
yc = vf.line_functions(*constants(grid), 60e3).yc
f, fv = grid[::2], grid[1::2]
samples, validation = yc[::2], yc[1::2]

model, report = vf.fit(f, samples, poles=12, symmetric=True)
print(model)
print(f"{report.status}, {len(report.history) - 1} relocations, rms error {report.rms_error:.1e}")

# %%
# Each real pole bends the response near its own frequency, so the poles spread over the band
# where Yc changes. Between the samples the model holds to 2 % with 4 poles, 0.3 % with 8 and
# 0.07 % with 12.

fd = np.geomspace(0.1, 1e6, 2001)
response = model.response(fd)
elements = {"Self": (0, 0), "Mutual": (0, 1)}
fig, (top, bottom) = plt.subplots(2, 1, sharex=True, height_ratios=(3, 2))
for k, (label, (i, j)) in enumerate(elements.items()):
    top.loglog(fd, abs(response[:, i, j]), color=f"C{k}", label=label)
for k, (i, j) in enumerate(elements.values()):
    dots = abs(samples[::4, i, j])
    top.loglog(f[::4], dots, "o", mfc="w", mec="k", mew=1, label=None if k else "Samples")
poles = abs(model.poles) / (2 * np.pi)
ticks = {"transform": top.get_xaxis_transform(), "ms": 10, "mew": 1.6, "mec": "C3"}
top.plot(poles, np.full_like(poles, 0.04), "|", label="Poles", **ticks)
top.set(title="Characteristic admittance (S)")
top.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=4)

for order, color in ((4, "C2"), (8, "C1"), (12, "C0")):
    fitted = vf.fit(f, samples, poles=order, symmetric=True).model
    error = vf.relative_error(fitted.response(fv), validation)
    bottom.loglog(fv, error, color=color, label=f"{order} poles")
bottom.xaxis.set_major_formatter(EngFormatter("Hz"))
bottom.set(title="Relative error between the samples", xlabel="Frequency")
bottom.legend(loc="lower left", bbox_to_anchor=(0, 1), ncols=3)
plt.show()
