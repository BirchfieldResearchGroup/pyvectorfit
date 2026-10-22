# pyvectorfit

:::{container} lede
Vector fitting of delayed pole-residue models for electromagnetic transient (EMT) simulation:
rational fitting, universal line models with searched modal delays, and passivity assessment
and enforcement, on plain NumPy arrays.
:::

:::{container} plot
```{image} gallery/images/sphx_glr_plot_line_001.svg
:class: only-light
:alt: Modal propagation by delay group and relative errors of a fitted 100 km three-phase line
```
```{image} gallery/images/sphx_glr_plot_line_001-dark.svg
:class: only-dark
:alt: Modal propagation by delay group and relative errors of a fitted 100 km three-phase line
```
:::

:::{container} caption
A 100 km three-phase line fitted from 0.1 Hz to 1 MHz, from the {doc}`gallery/plot_line` example.
:::

## Install

```bash
pip install pyvectorfit
```

## Quickstart

Per-metre series impedance `z` and shunt admittance `y` of a two-conductor line, sampled at
frequencies `f`, become a passive line model with one delay per mode group:

```python
import numpy as np
import pyvectorfit as vf

f = np.geomspace(1, 1e6, 121)  # Hz
s = 2j * np.pi * f[:, None, None]
r = 3e-5 * np.sqrt(1 + s / (2 * np.pi * 1e3))  # ohm/m, with skin effect
z = r * np.eye(2) + s * 1e-6 * np.array([[1.6, 0.5], [0.5, 1.6]])  # ohm/m
y = 1e-11 + s * 1e-12 * np.array([[8, -2], [-2, 8]])  # S/m

line, report = vf.fit_line(f, z, y, length=80e3)
line, correction = vf.enforce_passivity(line, f)
print(line.h.delays)  # s
```

`line.yc` and `line.h` are {class}`~pyvectorfit.PoleResidue` models: evaluate them with
`response(f)`, save them with `to_dict()`, and step them in time with
{func}`~pyvectorfit.recursive_convolution`.

## Examples

:::{container} examples
{doc}`gallery/plot_fit`
: Poles find the resonances of a line's input admittance.

{doc}`gallery/plot_line`
: Line constants to a passive universal line model, saved as JSON.

{doc}`gallery/plot_modal_delays`
: Why delays are searched, and what phase delays cost.

{doc}`gallery/plot_passivity`
: Find and remove passivity violations of a cable model.

{doc}`gallery/plot_transients`
: Energize the fitted line and compare with the exact response.
:::

The {doc}`guide` covers conventions and methods, {doc}`benchmarks` the accuracy on six lines
and cables, and the {doc}`api` every function.

```{toctree}
:hidden:

gallery/index
guide
benchmarks
api
```
