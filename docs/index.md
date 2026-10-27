---
myst:
  html_meta:
    description: "pyvectorfit: vector fitting in Python for EMT simulation. Rational fits, universal line models with modal delays, and passivity enforcement on NumPy arrays."
---

# pyvectorfit

:::{container} lede
Vector fitting of delayed pole-residue models for electromagnetic transient (EMT) simulation:
rational fitting, universal line models with searched modal delays, and passivity assessment
and enforcement, on plain NumPy arrays.
:::

:::{container} banner
```{image} _static/banner.svg
:class: only-light
:alt: Samples of a line's terminal admittance, the rational model fitted through them, and its poles beneath the resonances
```
```{image} _static/banner-dark.svg
:class: only-dark
:alt: Samples of a line's terminal admittance, the rational model fitted through them, and its poles beneath the resonances
```
:::

:::{container} caption
The terminal admittance of a 100 km three-phase line from 1 Hz to 10 kHz: samples, the rational
model fitted through them with 60 poles, and those poles beneath, complex pairs under the
resonances.
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

`z` and `y` come from any line constants program. For overhead lines,
[OpenLine](https://openline.readthedocs.io) computes them from its catalog of conductors and
structures.

## Examples

:::{container} examples
{doc}`gallery/plot_fit`
: Real poles fit a line's characteristic admittance, and their number sets the accuracy.

{doc}`gallery/plot_line`
: A line from OpenLine's catalog to a passive universal line model, saved as JSON.

{doc}`gallery/plot_modal_delays`
: Why delays are searched, and what phase delays cost.

{doc}`gallery/plot_passivity`
: Find and remove passivity violations of a cable model.

{doc}`gallery/plot_transients`
: Energize the fitted line and compare with the exact response.
:::

The {doc}`guide` covers conventions and methods, {doc}`benchmarks` the accuracy on six lines
and cables, and the {doc}`api` every function.

## Author

[Luke Lowery](https://lukelowry.github.io/), PhD student in the
[Birchfield Research Group](https://birchfield.engr.tamu.edu/) at Texas A&M University ·
[Google Scholar](https://scholar.google.com/citations?user=CTynuRMAAAAJ&hl=en) ·
[ORCID](https://orcid.org/0009-0000-0029-3403)

Also by the author: [OpenLine](https://openline.readthedocs.io), overhead transmission line
modeling · [latkit](https://latkit.readthedocs.io), GPU visualization of large networks.

<nav class="project-links" aria-label="Project links">
  <a href="https://github.com/BirchfieldResearchGroup/pyvectorfit">GitHub</a>
  <a href="https://pypi.org/project/pyvectorfit/">PyPI</a>
  <a href="https://github.com/BirchfieldResearchGroup/pyvectorfit/actions/workflows/ci.yml">CI</a>
  <a href="https://github.com/BirchfieldResearchGroup/pyvectorfit/blob/main/LICENSE">MIT license</a>
</nav>

```{toctree}
:hidden:

gallery/index
guide
benchmarks
api
```

```{toctree}
:hidden:
:caption: Related

OpenLine <https://openline.readthedocs.io>
```
