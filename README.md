<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/BirchfieldResearchGroup/pyvectorfit/main/docs/_static/banner-dark.svg">
    <img src="https://raw.githubusercontent.com/BirchfieldResearchGroup/pyvectorfit/main/docs/_static/banner.svg" width="100%" alt="Samples of a line's terminal admittance, the rational model fitted through them, and its poles beneath the resonances">
  </picture>
</p>

# pyvectorfit

[![CI](https://img.shields.io/github/actions/workflow/status/BirchfieldResearchGroup/pyvectorfit/ci.yml?branch=main&label=CI)](https://github.com/BirchfieldResearchGroup/pyvectorfit/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyvectorfit?label=PyPI)](https://pypi.org/project/pyvectorfit/)
[![Python](https://img.shields.io/pypi/pyversions/pyvectorfit)](https://pypi.org/project/pyvectorfit/)
[![Documentation](https://img.shields.io/readthedocs/pyvectorfit?label=docs)](https://pyvectorfit.readthedocs.io)
[![MIT license](https://img.shields.io/badge/license-MIT-blue)](https://github.com/BirchfieldResearchGroup/pyvectorfit/blob/main/LICENSE)

Vector fitting of delayed pole-residue models for electromagnetic transient (EMT) simulation:
rational fitting, universal line models with searched modal delays, and passivity assessment
and enforcement, on plain NumPy arrays.

[Documentation](https://pyvectorfit.readthedocs.io): examples, guide, benchmarks and API.

```bash
pip install pyvectorfit
```

## Quick Start

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

Line constants `z` and `y` come from any line constants program. For overhead lines,
[OpenLine](https://openline.readthedocs.io) computes them from conductor and structure catalogs.

## Author

[Luke Lowery](https://lukelowry.github.io/), PhD student in the
[Birchfield Research Group](https://birchfield.engr.tamu.edu/) at Texas A&M University ·
[Google Scholar](https://scholar.google.com/citations?user=CTynuRMAAAAJ&hl=en) ·
[ORCID](https://orcid.org/0009-0000-0029-3403)

Also by the author: [OpenLine](https://openline.readthedocs.io), overhead transmission line
modeling · [latkit](https://latkit.readthedocs.io), GPU visualization of large networks ·
[sgwt](https://github.com/lukelowry/sgwt), sparse graph convolution for power systems ·
[gridkit-studio](https://github.com/lukelowry/gridkit-studio), a VS Code extension for GridKit
