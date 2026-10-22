# Benchmarks

Six overhead lines and underground cables, fitted from 0.1 Hz to 1 MHz with 20 samples per
decade. Errors are the largest relative errors on a validation grid ten times denser that avoids
the fitting samples. The passivity fix is the largest relative change that
{func}`~pyvectorfit.enforce_passivity` made to $Y_c$, and the EMT error the largest difference
from the exact line during energization, relative to the peak voltage.
`python -m benchmarks` regenerates every number and figure here.

```{csv-table}
:file: _static/benchmarks/summary.csv
:header-rows: 1
:class: numbers
```

## Accuracy

The characteristic admittance, propagation and terminal admittance of every case, across the
band. $H$ is weighted by its influence on the terminal admittance, so its relative error can
grow where the propagation has decayed and no longer matters.

:::{container} plot
```{image} _static/benchmarks/accuracy.svg
:class: only-light
:alt: Relative errors of Yc, H and the terminal admittance for six lines and cables
```
```{image} _static/benchmarks/accuracy-dark.svg
:class: only-dark
:alt: Relative errors of Yc, H and the terminal admittance for six lines and cables
```
:::

## Model order

Errors fall steadily with the number of poles until the terminal admittance reaches the
accuracy of $Y_c$. The terminal admittance is plotted against the poles per delay group of $H$.

:::{container} plot
```{image} _static/benchmarks/orders.svg
:class: only-light
:alt: Maximum errors of Yc and the terminal admittance against the number of poles
```
```{image} _static/benchmarks/orders-dark.svg
:class: only-dark
:alt: Maximum errors of Yc and the terminal admittance against the number of poles
```
:::

## Robustness

The three-phase line with its length, earth resistivity and sampling density varied one at a
time.

:::{container} plot
```{image} _static/benchmarks/sweeps.svg
:class: only-light
:alt: Maximum errors against line length, earth resistivity and samples per decade
```
```{image} _static/benchmarks/sweeps-dark.svg
:class: only-dark
:alt: Maximum errors against line length, earth resistivity and samples per decade
```
:::

## Data

Line constants come from `benchmarks/cases.py`: overhead earth return by the complex depth of
Deri et al. [1], conductor internal impedance by Schelkunoff's tubular formulas [2] and buried
earth return by the closed form of Saad et al. [3]. Shunt conductance is $10^{-11}$ S/m on
overhead lines, and cable insulation has a loss tangent of $4 \times 10^{-4}$. The exact
transients invert the exact line's two-port with the numerical Laplace transform [4].

1. A. Deri, G. Tevan, A. Semlyen, A. Castanheira, "The complex ground return plane", *IEEE
   Trans. Power Apparatus and Systems* 100(8), 1981.
2. S. A. Schelkunoff, "The electromagnetic theory of coaxial transmission lines and cylindrical
   shields", *Bell System Technical Journal* 13, 1934.
3. O. Saad, G. Gaba, M. Giroux, "A closed-form approximation for ground return impedance of
   underground cables", *IEEE Trans. Power Delivery* 11(3), 1996.
4. P. Moreno, A. Ramirez, "Implementation of the numerical Laplace transform: a review", *IEEE
   Trans. Power Delivery* 23(4), 2008.
