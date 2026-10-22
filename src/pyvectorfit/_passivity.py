from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Literal, TypeVar

import numpy as np
from numpy.typing import ArrayLike
from scipy import linalg as la
from scipy.optimize import minimize_scalar, nnls

from ._fit import (
    Weights,
    _assemble,
    _basis,
    _pack,
    _stack,
    _symmetric_weights,
    _unpack,
    _weights,
    relative_error,
)
from ._line import LineModel, _waves, terminal_admittance
from ._model import PoleResidue, Result, _frequencies

Admittance = TypeVar("Admittance", PoleResidue, LineModel)

_RTOL = 1e-12
_MARGIN = 1e-10
_CLUSTER = 10.0
_WITNESSES = 10


@dataclass(frozen=True)
class PassivityReport:
    """``margin`` is the smallest eigenvalue found of the Hermitian part, at ``frequency`` in Hz."""

    passive: bool
    scope: Literal["global", "sampled"]
    margin: float
    frequency: float
    bands: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class CorrectionReport:
    """``change`` is the largest relative change of the admittance on the correction frequencies."""

    iterations: int
    before: PassivityReport
    after: PassivityReport
    change: float


def check_passivity(model: PoleResidue | LineModel, f: ArrayLike | None = None) -> PassivityReport:
    """Passivity of an admittance or impedance.

    An undelayed PoleResidue is assessed at all frequencies from the imaginary zeros of
    ``F(s) + F(-s)^T``. A delayed PoleResidue and the terminal admittance of a LineModel are
    assessed on the frequencies ``f``.
    """
    return _assess(model, f)[0]


def enforce_passivity(
    model: Admittance,
    f: ArrayLike,
    *,
    preserve_dc: bool = False,
    iterations: int = 20,
    weights: Weights = "relative_norm",
) -> Result[Admittance, CorrectionReport]:
    """Passive model from the smallest weighted change of the residues and ``d`` of an admittance.

    The change is measured on ``f``. A LineModel keeps ``h`` and corrects ``yc`` until ``yc`` is
    passive and the terminal admittance is passive on ``f``. Poles, delays and ``e`` are kept.
    Raises if not achieved.
    """
    f = _frequencies(f)
    y = model.yc if isinstance(model, LineModel) else model
    h = model.h if isinstance(model, LineModel) else None
    if np.any(y.delays):
        raise ValueError("only undelayed admittances can be corrected")
    space = _Correction(y, f, weights, preserve_dc)
    if not _psd(y.e.reshape(space.n, space.n)):
        raise ValueError("e must be symmetric positive semidefinite")
    scale = _scale(y, space.n)
    rows: list[np.ndarray] = []
    bounds: list[float] = []
    corrected = y
    for iteration in range(iterations + 1):
        report, witnesses = _global(corrected, space.n)
        cuts: list[tuple[float, PoleResidue | None]] = [(witness, None) for witness in witnesses]
        passive = report.passive
        if h is not None:
            report, witnesses = _assess(LineModel(corrected, h), f)
            passive &= report.passive
            cuts += [(witness, h) for witness in witnesses]
        if passive:
            break
        if iteration == iterations:
            raise RuntimeError(f"not passive after {iterations} iterations")
        for witness, propagation in cuts:
            new_rows, new_bounds = _cuts(space, witness, y, corrected, propagation, scale)
            rows += new_rows
            bounds += new_bounds
        corrected = space.model(_ldp(np.array(rows), np.array(bounds)))
    result = replace(model, yc=corrected) if isinstance(model, LineModel) else corrected
    change = float(relative_error(corrected.response(f), y.response(f)).max())
    before, after = _assess(model, f)[0], _assess(result, f)[0]
    return Result(result, CorrectionReport(iteration, before, after, change))


def _assess(model, f):
    if isinstance(model, LineModel):
        f = _required(f)
        return _sampled(terminal_admittance(model.yc.response(f), model.h.response(f)), f)
    n = _ports(model)
    if np.any(model.delays):
        f = _required(f)
        return _sampled(model.response(f).reshape(-1, n, n), f)
    return _global(model, n)


def _required(f):
    if f is None:
        raise ValueError("f is required for delayed and line models")
    return _frequencies(f)


def _sampled(values, f):
    """Report and witnesses (local minima and spread samples of each violating run)."""
    herm = np.linalg.eigvalsh((values + values.conj().swapaxes(1, 2)) / 2)[:, 0]
    bad = np.flatnonzero(herm < -_RTOL * np.linalg.norm(values, axis=(1, 2)))
    runs = np.split(bad, np.flatnonzero(np.diff(bad) > 1) + 1) if bad.size else []
    padded = np.r_[np.inf, herm, np.inf]
    minima = bad[(herm[bad] <= padded[bad]) & (herm[bad] <= padded[bad + 2])]
    spread = [r[np.linspace(0, len(r) - 1, min(len(r), _WITNESSES)).astype(int)] for r in runs]
    worst = int(np.argmin(herm))
    report = PassivityReport(
        not bad.size,
        "sampled",
        float(herm[worst]),
        float(f[worst]),
        tuple((float(f[r[0]]), float(f[r[-1]])) for r in runs),
    )
    return report, f[np.unique(np.concatenate([minima, *spread]))]


def _global(model, n):
    """Report and witness frequencies from the sign of the Hermitian part between crossings."""
    tol = _RTOL * _scale(model, n)
    if not _psd(model.e.reshape(n, n)):
        return PassivityReport(False, "global", -np.inf, np.inf, ()), np.zeros(0)

    def herm(f):
        values = model.response(np.atleast_1d(f)).reshape(-1, n, n)
        return np.linalg.eigvalsh((values + values.conj().swapaxes(1, 2)) / 2)[:, 0]

    d = model.d.sum(axis=0).reshape(n, n)
    infinity = float(np.linalg.eigvalsh((d + d.T) / 2)[0])
    crossings = _crossings(model, n)
    reference = np.exp(np.mean(np.log(np.abs(model.poles)))) / (2 * np.pi) if model.order else 1.0
    edges = np.r_[0.0, crossings, np.inf]
    points, values, witnesses = [0.0, np.inf], [herm(0.0)[0], infinity], []
    bands: list[tuple[float, float]] = []
    for lo, hi in pairwise(edges):
        mid = _inside(lo, hi, reference)
        value = herm(mid)[0]
        points.append(mid)
        values.append(value)
        if value >= -tol:
            continue
        upper = hi if np.isfinite(hi) else 1e6 * mid
        lower = lo if lo else 1e-6 * mid
        search = minimize_scalar(
            lambda x: herm(np.exp(x))[0], bounds=(np.log(lower), np.log(upper)), method="bounded"
        )
        worst = float(np.exp(search.x))
        points.append(worst)
        values.append(search.fun)
        witnesses += [worst, *np.geomspace(lower, upper, _WITNESSES + 2)[1:-1]]
        if bands and bands[-1][1] == lo:
            bands[-1] = (bands[-1][0], float(hi))
        else:
            bands.append((float(lo), float(hi)))
    if infinity < -tol:
        witnesses.append(np.inf)
    worst = int(np.argmin(values))
    passive = bool(values[worst] >= -tol)
    report = PassivityReport(
        passive, "global", float(values[worst]), float(points[worst]), tuple(bands)
    )
    return report, np.array(witnesses)


def _inside(lo, hi, reference):
    """A point strictly inside (lo, hi), which may start at 0 or extend to infinity."""
    if np.isinf(hi):
        return 2 * lo if lo else reference
    return np.sqrt(lo * hi) if lo else hi / 2


def _crossings(model, n):
    """Frequencies in Hz of purely imaginary zeros of F(s) + F(-s)^T, from its Rosenbrock pencil."""
    if not model.order:
        return np.zeros(0)
    a, b, c = _realization(model, n)
    rate = np.exp(np.mean(np.log(np.abs(model.poles))))
    scale = _scale(model, n)
    d = model.d.sum(axis=0).reshape(n, n)
    a, c, q = a / rate, c / (rate * scale), (d + d.T) / scale
    m = len(a)
    zero = np.zeros((m, m))
    pencil = np.block([[a, zero, b], [zero, -a.T, -c.T], [c, b.T, q]])
    mass = la.block_diag(np.eye(2 * m), np.zeros((n, n)))
    alpha, beta = la.eigvals(pencil, mass, homogeneous_eigvals=True)
    if np.any(np.maximum(np.abs(alpha), np.abs(beta)) < 1e-13 * np.linalg.norm(pencil)):
        raise np.linalg.LinAlgError("F(s) + F(-s)^T is singular for all s")
    finite = np.abs(beta) > 1e-14 * np.abs(alpha)
    zeros = alpha[finite] / beta[finite]
    imaginary = (np.abs(zeros.real) <= 1e-4 * np.abs(zeros)) & (zeros.imag > 0)
    return np.unique(zeros[imaginary].imag) * rate / (2 * np.pi)


def _realization(model, n):
    """Real (A, B, C) with B = I per pole, so that F(s) - d - s e = C (sI - A)^-1 B."""
    eye = np.eye(n)
    a, b, c = [], [], []
    for p, r in zip(model.poles, model.residues.reshape(-1, n, n), strict=True):
        if p.imag:
            a.append(np.kron([[p.real, -p.imag], [p.imag, p.real]], eye))
            b.append(np.vstack((eye, 0 * eye)))
            c.append(np.hstack((2 * r.real, -2 * r.imag)))
        else:
            a += [p.real * eye]
            b += [eye]
            c += [r.real]
    return la.block_diag(*a), np.vstack(b), np.hstack(c)


def _ports(model):
    if model.shape == ():
        return 1
    if len(model.shape) != 2 or model.shape[0] != model.shape[1]:
        raise ValueError("passivity applies to scalar or square matrix models")
    return model.shape[0]


def _scale(model, n):
    """Upper bound of |F(j omega) - j omega e| over all omega."""
    residues = np.linalg.norm(model.residues.reshape(-1, n, n), 2, axis=(1, 2))
    weight = np.where(model.poles.imag > 0, 2.0, 1.0) / -model.poles.real
    return float(np.linalg.norm(model.d.sum(axis=0).reshape(n, n), 2) + weight @ residues) or 1.0


def _psd(e):
    return np.array_equal(e, e.T) and np.linalg.eigvalsh(e)[0] >= -_RTOL * np.abs(e).max()


class _Correction:
    """Coefficients of an admittance as x0 + T u, where |u| is the weighted change on f."""

    def __init__(self, y, f, weights, preserve_dc):
        self.y, self.n = y, _ports(y)
        n = self.n
        positive = f[f > 0]
        if not positive.size:
            raise ValueError("f needs a positive frequency")
        self.rate = 2 * np.pi * np.sqrt(positive[0] * positive[-1])
        self.poles = y.poles / self.rate
        r = y.residues.reshape(-1, n, n) / self.rate
        real = self.poles.imag == 0
        d = y.d.sum(axis=0).reshape(1, n, n)
        x = np.concatenate((r[real].real, r[~real].real, r[~real].imag, d))
        self.symmetric = n > 1 and np.allclose(x, x.swapaxes(1, 2), rtol=1e-12, atol=0)
        self.x0 = _pack(x, self.symmetric)
        phi = _basis(2j * np.pi * f / self.rate, self.poles, "constant")
        if phi.shape[1] > 2 * len(f):
            raise ValueError("too few frequencies for the number of coefficients")
        w = _pack(_weights(y.response(f).reshape(-1, n, n), weights), self.symmetric)
        if self.symmetric:
            w = w * _symmetric_weights(n)
        null = np.eye(phi.shape[1])
        if preserve_dc:
            null = la.null_space(_basis(np.zeros(1), self.poles, "constant").real)
        self.t, profile = [], None
        for column in w.T:
            peak = column.max()
            if profile is None or not np.allclose(column / peak, profile, rtol=1e-12, atol=0):
                profile, reference = column / peak, peak
                r = np.linalg.qr(_stack(column[:, None] * phi) @ null, mode="r")
                t = null @ la.solve_triangular(r, np.eye(len(r)))
            self.t.append(t * (reference / peak))

    def model(self, u):
        steps = np.split(u, len(self.t))
        x = self.x0 + np.column_stack([t @ step for t, step in zip(self.t, steps, strict=True)])
        x = _unpack(x, (self.n, self.n), self.symmetric)
        residues, d, _ = _assemble(x, self.poles, "constant", 1)
        shape = self.y.shape
        return PoleResidue(
            self.y.poles,
            (residues * self.rate).reshape(len(self.poles), *shape),
            d.reshape(1, *shape),
            self.y.e,
        )

    def gradient(self, row, p):
        """Gradient in u of sum_c Re(row_c sum_ij p_ij x_cij) for the basis row at one s."""
        if self.symmetric:
            p = p + p.T - np.diag(np.diag(p))
        g = np.real(row[:, None] * _pack(p[None], self.symmetric))
        return np.concatenate([g[:, j] @ t for j, t in enumerate(self.t)])


def _cuts(space, witness, original, corrected, h, scale):
    """Linear constraints lifting the low eigenvalue cluster at witness above the margin.

    The margin is relative to the admittance, so thin natural margins such as the shunt
    conductance of a line stay attainable; cutting the whole cluster keeps near-degenerate
    eigenvectors from rotating between iterations.
    """
    n = space.n
    if np.isinf(witness):
        row = np.zeros(len(space.x0))
        row[-1] = 1.0
        y0, y1 = (m.d.sum(axis=0).reshape(n, n) for m in (original, corrected))
        k = np.eye(n)
    else:
        s = 2j * np.pi * witness
        row = _basis(np.array([s / space.rate]), space.poles, "constant")[0]
        y0, y1 = (m(s).reshape(n, n) for m in (original, corrected))
        k = np.eye(n) if h is None else _waves(h(s).reshape(n, n))
    blocks = np.eye(len(k) // n)
    terminal, original = k @ np.kron(blocks, y1), k @ np.kron(blocks, y0)
    margin = _MARGIN * (scale if h is None else np.linalg.norm(terminal))
    values, vectors = np.linalg.eigh((terminal + terminal.conj().T) / 2)
    cluster = margin + _CLUSTER * max(margin - values[0], 0.0)
    rows, bounds = [], []
    for v in vectors[:, values < cluster].T:
        a = k.conj().T @ v
        p = sum(np.outer(a[i : i + n].conj(), v[i : i + n]) for i in range(0, len(v), n))
        rows.append(space.gradient(row, p))
        bounds.append(margin - np.real(v.conj() @ original @ v))
    return rows, bounds


def _ldp(g, h):
    """Minimum-norm u with g u >= h, by Lawson-Hanson reduction to nonnegative least squares."""
    norm = np.linalg.norm(g, axis=1)
    if np.any((norm == 0) & (h > 0)):
        raise RuntimeError("passivity constraints are infeasible")
    keep = norm > 0
    g, h = g[keep] / norm[keep, None], h[keep] / norm[keep]
    e = np.vstack((g.T, h))
    target = np.zeros(len(e))
    target[-1] = 1.0
    residual = e @ nnls(e, target, maxiter=50 * e.shape[1])[0] - target
    if residual[-1] > -1e-12:
        raise RuntimeError("passivity constraints are infeasible")
    return -residual[:-1] / residual[-1]
