# Guide

Conventions, the model, and the methods behind each step.

## Conventions

- Frequencies `f` are in Hz; `s`, poles and residues in rad/s; delays in s.
- Samples are arrays of shape `(len(f), *shape)`, with a scalar, vector or square matrix
  `shape`.
- Lines take per-metre series impedance `z` (Ω/m) and shunt admittance `y` (S/m).
- Fits return a model and a report, `model, report = vf.fit(...)`.

## Model

A {class}`~pyvectorfit.PoleResidue` is a delayed pole-residue model

$$
F(s) = s\,e + \sum_g e^{-s\tau_g} \Big( d_g + \sum_{k \in g} \frac{r_k}{s - p_k} \Big)
$$

with stable poles $p_k$, residues $r_k$ and, per delay group $g$, a delay $\tau_g$ and a
constant $d_g$. A complex pole is stored once, with positive imaginary part, and implies its
conjugate pair. An undelayed model has one group with $\tau = 0$. State-space realizations are
an internal detail; the model is its poles, residues and delays.

## Lines

Lines use current waves, with propagation constant $\Gamma = \sqrt{YZ}$:

$$
Y_c = \Gamma Z^{-1}, \qquad H = e^{-\ell \Gamma}.
$$

Their terminal admittance, with currents into both ends, is

$$
\begin{bmatrix} I_1 \\ I_2 \end{bmatrix} =
\begin{bmatrix} A & B \\ B & A \end{bmatrix}
\begin{bmatrix} V_1 \\ V_2 \end{bmatrix}, \qquad
A = (I - H^2)^{-1} (I + H^2) Y_c, \qquad
B = -2 (I - H^2)^{-1} H Y_c.
$$

{func}`~pyvectorfit.fit_line` builds the universal line model [4]:

1. $Y_c$ is fitted in the phase domain with symmetric residues.
2. Modes are tracked across frequency by eigenvector continuity, and modes whose delays agree
   within `group_tol` form one delay group.
3. Each group's delay is searched, and its poles are fitted to its delay-removed modal
   propagation functions.
4. $H$ is fitted in the phase domain in one linear solve, with all poles and delays fixed.

## Methods

**Rational fitting.** Relaxed vector fitting [1, 2] on a real basis, so conjugate symmetry
holds by construction. Pole relocation uses the QR-compressed fast formulation [3] with column
equilibration and rank-revealing QR, and the relaxation is normalized by an exact constraint.
Relocation stops once the residual stagnates, and the best iterate is returned, not the last.

**Delays.** The delay of each group minimizes the residual of its delay-removed modal fit [5].
The optimum lies in a narrow valley just below a causality cliff: near the phase delay at the
highest frequency for lightly damped modes, and up to 15 % above it for heavily attenuated
ones. Two nested scans locate the valley and Brent's method refines it.

**Weighting.** $H$ is weighted by $1 / \lVert I - H^2 \rVert$, the relative sensitivity of the
terminal admittance to $H$. At low frequency $I - H \approx \ell\Gamma$, so absolute errors in
$H$ are amplified by $1/\lvert \ell \Gamma \rvert$; plain relative weights leave terminal
errors of 1 to 20 % below 10 Hz, where this weighting gives about $10^{-3}$.

**Passivity assessment.** An undelayed model is checked at all frequencies: the imaginary zeros
of $F(s) + F(-s)^T$ come from its Rosenbrock pencil, a generalized eigenproblem that tolerates a
singular $d$ (compare [7]), and the sign of the Hermitian part is tested between them. Delayed
models and terminal admittances are checked on samples.

**Passivity enforcement.** Residues and $d$ are perturbed [6] with accumulated linear cuts, each
least-distance problem solved by reduction to nonnegative least squares [9]; line constraints
follow [8]. Margins are relative, $10^{-10}$ of the admittance, because a line's natural margin
at low frequency is only about $G\ell/2$. Cuts lift the whole cluster of low eigenvalues, so
nearly degenerate eigenvectors cannot rotate between iterations.

**Time domain.** {func}`~pyvectorfit.recursive_convolution` gives each pole state the recursion
$x_n = a\,x_{n-1} + b_0 u_{n-1} + b_1 u_n$, exact for inputs that are linear over each step
[10]. A real pole adds $r\,x$ to the output and a stored complex pole $2\,\mathrm{Re}(r\,x)$.

## Practical notes

- Sample the band the simulation needs at about 20 points per decade; validate and enforce
  passivity on a grid that is ten times denser.
- With $G = 0$ a line is only marginally passive near DC. Give line constants a small shunt
  conductance, as EMT programs do, for example $10^{-11}$ S/m.
- Nearly lossless data, such as a constant series resistance, leaves the terminal admittance
  passive only to rounding; realistic skin effect and earth return avoid this.

## References

1. B. Gustavsen, A. Semlyen, "Rational approximation of frequency domain responses by vector
   fitting", *IEEE Trans. Power Delivery* 14(3), 1999.
2. B. Gustavsen, "Improving the pole relocating properties of vector fitting", *IEEE Trans.
   Power Delivery* 21(3), 2006.
3. D. Deschrijver, M. Mrozowski, T. Dhaene, D. De Zutter, "Macromodeling of multiport systems
   using a fast implementation of the vector fitting method", *IEEE Microw. Wireless Compon.
   Lett.* 18(6), 2008.
4. A. Morched, B. Gustavsen, M. Tartibi, "A universal model for accurate calculation of
   electromagnetic transients on overhead lines and underground cables", *IEEE Trans. Power
   Delivery* 14(3), 1999.
5. B. Gustavsen, "Optimal time delay extraction for transmission line modeling", *IEEE Trans.
   Power Delivery* 32(1), 2017.
6. B. Gustavsen, A. Semlyen, "Enforcing passivity for admittance matrices approximated by
   rational functions", *IEEE Trans. Power Systems* 16(1), 2001.
7. A. Semlyen, B. Gustavsen, "A half-size singularity test matrix for fast and reliable
   passivity assessment of rational models", *IEEE Trans. Power Delivery* 24(1), 2009.
8. B. Gustavsen, "Passivity enforcement for transmission line models based on the method of
   characteristics", *IEEE Trans. Power Delivery* 23(4), 2008.
9. C. L. Lawson, R. J. Hanson, *Solving Least Squares Problems*, Prentice-Hall, 1974.
10. A. Semlyen, A. Dabuleanu, "Fast and accurate switching transient calculations on
    transmission lines with ground return using recursive convolutions", *IEEE Trans. Power
    Apparatus and Systems* 94(2), 1975.
