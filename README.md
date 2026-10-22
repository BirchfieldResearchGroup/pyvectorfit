# pyvectorfit

Vector fitting of delayed pole-residue models for electromagnetic transient (EMT) simulation:
rational fitting, universal line model fitting with modal delay identification, and passivity
assessment and enforcement.

```bash
pip install pyvectorfit
```

```python
import pyvectorfit as vf

model, report = vf.fit(f, samples, poles=12)
line, report = vf.fit_line(f, z, y, length)
line, report = vf.enforce_passivity(line, f)
```

## Author

[Luke Lowery](https://lukelowry.github.io/), PhD student in the
[Birchfield Research Group](https://birchfield.engr.tamu.edu/) at Texas A&M University ·
[Google Scholar](https://scholar.google.com/citations?user=CTynuRMAAAAJ&hl=en) ·
[ORCID](https://orcid.org/0009-0000-0029-3403)

Also by the author: [sgwt](https://github.com/lukelowry/sgwt), sparse graph convolution for power
systems · [latkit](https://github.com/lukelowry/latkit), GPU power system visualization ·
[gridkit-studio](https://github.com/lukelowry/gridkit-studio), a VS Code extension for GridKit
