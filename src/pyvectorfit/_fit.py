from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg as la

from ._model import PoleResidue, Result, _frequencies, _layout, _real

Asymptote = Literal["none", "constant", "linear"]
Weights = Literal["relative", "relative_norm"] | ArrayLike | None

_FLOOR = 1e-3
_RCOND = 1e-12
_TINY = np.finfo(float).tiny


@dataclass(frozen=True)
class FitReport:
    """``history`` holds the weighted relative residual at the start and after each relocation."""

    status: Literal["converged", "max_iterations", "fixed"]
    history: tuple[float, ...]
    rms_error: float
    max_error: float
    asymmetry: float


def relative_error(
    values: ArrayLike, reference: ArrayLike, *, floor: float = _FLOOR
) -> NDArray[np.float64]:
    """Per-sample Frobenius error relative to ``max(||reference||, floor * max ||reference||)``."""
    values, reference = np.asarray(values), np.asarray(reference)
    if values.shape != reference.shape or not values.ndim:
        raise ValueError("values and reference must share a shape with samples on axis 0")
    norm = np.linalg.norm(reference.reshape(len(reference), -1), axis=1)
    error = np.linalg.norm((values - reference).reshape(len(reference), -1), axis=1)
    return error / np.maximum(norm, max(floor * norm.max(), _TINY))


def fit(
    f: ArrayLike,
    values: ArrayLike,
    *,
    poles: int | ArrayLike = 12,
    asymptote: Asymptote = "constant",
    weights: Weights = "relative_norm",
    dc: ArrayLike | None = None,
    symmetric: bool = False,
    delay: float = 0.0,
    iterations: int = 20,
    tol: float = 1e-3,
) -> Result[PoleResidue, FitReport]:
    """Relaxed vector fitting of samples at frequencies ``f`` in Hz with one common set of poles.

    ``poles`` is the order (complex pairs count twice) or the starting poles. ``delay`` is removed
    from the samples and stored in the model. Relocation stops once the weighted residual
    improves by less than a fraction ``tol`` twice in a row; the best iterate is returned.
    """
    if not np.isfinite(delay) or delay < 0:
        raise ValueError("delay must be nonnegative and finite")
    if iterations < 0 or not 0 <= tol < 1:
        raise ValueError("iterations must be nonnegative and tol in [0, 1)")
    problem = _Problem(f, values, asymptote, weights, dc, symmetric, delay)
    if isinstance(poles, int | np.integer):
        p = problem.initial(int(poles))
    else:
        p = _layout(poles, (0.0,), None)[0] / problem.rate
    x, residual = problem.solve(p)
    best, history, status = (residual, x, p), [residual], "fixed"
    if len(p) and iterations and np.any(problem.y):
        status, stalled = "max_iterations", 0
        for _ in range(iterations):
            p = problem.relocate(p)
            x, residual = problem.solve(p)
            history.append(residual)
            stalled = 0 if residual < best[0] * (1 - tol) else stalled + 1
            if residual < best[0]:
                best = (residual, x, p)
            if stalled == 2:
                status = "converged"
                break
    model = problem.model(best[1], best[2])
    return Result(model, problem.report(model, status, history))


def fit_residues(
    f: ArrayLike,
    values: ArrayLike,
    *,
    poles: ArrayLike,
    delays: ArrayLike = (0.0,),
    groups: ArrayLike | None = None,
    asymptote: Asymptote = "constant",
    weights: Weights = "relative_norm",
    dc: ArrayLike | None = None,
    symmetric: bool = False,
) -> Result[PoleResidue, FitReport]:
    """Residues, delayed constants and s-term for fixed poles and delay groups, in one solve."""
    p, t, g = _layout(poles, delays, groups)
    problem = _Problem(f, values, asymptote, weights, dc, symmetric)
    p, t = p / problem.rate, t * problem.rate
    x, residual = problem.solve(p, g, t)
    model = problem.model(x, p, g, t)
    return Result(model, problem.report(model, "fixed", [residual]))


class _Problem:
    """Weighted samples packed as (samples, channels), with s normalized by rate."""

    def __init__(self, f, values, asymptote, weights, dc, symmetric, delay=0.0):
        f = _frequencies(f)
        y = np.asarray(values, complex)
        if y.ndim not in (1, 2, 3) or len(y) != len(f) or 0 in y.shape or not np.isfinite(y).all():
            raise ValueError("values must be finite with shape (len(f), *shape)")
        if asymptote not in ("none", "constant", "linear"):
            raise ValueError("asymptote must be 'none', 'constant' or 'linear'")
        if f[-1] == 0:
            raise ValueError("f needs a positive frequency")
        self.f, self.raw, self.shape = f, y, y.shape[1:]
        self.asymptote, self.symmetric, self.delay = asymptote, symmetric, delay
        positive = f[f > 0]
        self.rate = 2 * np.pi * np.sqrt(positive[0] * positive[-1])
        self.s = 2j * np.pi * f / self.rate
        self.asymmetry = 0.0
        if symmetric:
            if len(self.shape) != 2 or self.shape[0] != self.shape[1]:
                raise ValueError("symmetric=True requires square matrix values")
            transposed = y.swapaxes(1, 2)
            self.asymmetry = float(np.linalg.norm(y - transposed) / max(np.linalg.norm(y), _TINY))
            y = (y + transposed) / 2
        w = _weights(y, weights)
        if symmetric and not np.allclose(w, w.swapaxes(1, 2)):
            raise ValueError("symmetric=True requires symmetric weights")
        shift = np.exp(2j * np.pi * f * delay).reshape(-1, *(1,) * len(self.shape))
        self.y, self.w = _pack(y * shift, symmetric), _pack(w, symmetric)
        if symmetric:
            self.w = self.w * _symmetric_weights(self.shape[0])
        self.dc = None
        if dc is not None:
            dc = _real(dc, "dc")
            if dc.shape != self.shape or (symmetric and not np.allclose(dc, dc.T)):
                raise ValueError("dc must match the value shape and requested symmetry")
            self.dc = _pack(dc[None], symmetric)[0]

    def initial(self, order):
        """Lightly damped pairs spread logarithmically over the band, plus a real pole if odd."""
        if order < 0:
            raise ValueError("the model order must be nonnegative")
        positive = self.f[self.f > 0]
        beta = 2 * np.pi * np.geomspace(positive[0], positive[-1], max(order // 2, 1)) / self.rate
        return np.r_[np.full(order % 2, -1.0 + 0j), beta[: order // 2] * (-0.01 + 1j)]

    def solve(self, poles, groups=None, delays=None):
        """Weighted least-squares coefficients for fixed poles and their relative residual."""
        phi = _basis(self.s, poles, self.asymptote, groups, delays)
        if phi.shape[1] > 2 * len(self.s):
            raise ValueError("too few samples for the number of coefficients")
        row = None
        if self.dc is not None:
            row = _basis(np.zeros(1), poles, self.asymptote, groups, delays).real[0]
        w = self.w / self.w.max(axis=0)
        if np.allclose(w, w[:, :1], rtol=1e-12, atol=0):
            x = _lstsq(_stack(w[:, :1] * phi), _stack(w[:, :1] * self.y), row, self.dc)
        else:
            x = np.hstack(
                [
                    _lstsq(
                        _stack(w[:, [j]] * phi),
                        _stack(w[:, [j]] * self.y[:, [j]]),
                        row,
                        None if self.dc is None else self.dc[[j]],
                    )
                    for j in range(self.y.shape[1])
                ]
            )
        residual = np.linalg.norm(self.w * (phi @ x - self.y))
        return x, float(residual / max(np.linalg.norm(self.w * self.y), _TINY))

    def relocate(self, poles):
        """Zeros of the relaxed weighting function sigma, from the compressed fast-VF system."""
        phi = _basis(self.s, poles, self.asymptote)
        psi = _basis(self.s, poles, "constant")
        r, q, profile = np.zeros((0, psi.shape[1])), np.zeros((2 * len(phi), 0)), None
        for w, y in zip(self.w.T, self.y.T, strict=True):
            if profile is None or not np.allclose(w / w.max(), profile, rtol=1e-12, atol=0):
                profile, q = w / w.max(), _range(_stack(w[:, None] * phi))
            g = _stack((w * y)[:, None] * psi)
            r = np.linalg.qr(np.vstack((r, g - q @ (q.T @ g))), mode="r")
        z = _lstsq(r, np.zeros((len(r), 1)), psi.real.mean(axis=0), np.ones(1))[:, 0]
        if abs(z[-1]) < 1e-8 * np.linalg.norm(z):
            z = np.r_[_lstsq(r[:, :-1], -r[:, -1:])[:, 0], 1.0]
        roots = la.eigvals(_sigma_zeros(poles, z))
        roots = -np.maximum(np.abs(roots.real), np.finfo(float).eps) + 1j * roots.imag
        roots = roots[roots.imag >= 0]
        return roots[np.argsort(np.abs(roots))]

    def model(self, x, poles, groups=None, delays=None):
        delays = np.zeros(1) if delays is None else delays
        x = _unpack(x, self.shape, self.symmetric)
        residues, d, e = _assemble(x, poles, self.asymptote, len(delays))
        return PoleResidue(
            poles * self.rate,
            residues * self.rate,
            d,
            e / self.rate,
            delays / self.rate + self.delay,
            groups,
        )

    def report(self, model, status, history):
        error = relative_error(model.response(self.f), self.raw)
        return FitReport(
            status,
            tuple(history),
            float(np.sqrt(np.mean(error**2))),
            float(error.max()),
            self.asymmetry,
        )


def _weights(y, weights):
    """Residual weights with the shape of y, scaled to a maximum of one."""
    if weights is None:
        w = np.ones(y.shape)
    elif isinstance(weights, str):
        if weights == "relative":
            magnitude = np.abs(y)
            peak = magnitude.max(axis=0)
            w = 1 / np.maximum(magnitude, _FLOOR * np.where(peak > 0, peak, 1.0))
        elif weights == "relative_norm":
            norm = np.linalg.norm(y.reshape(len(y), -1), axis=1)
            w = 1 / np.maximum(norm, _FLOOR * norm.max() or 1.0)
            w = np.broadcast_to(w.reshape(-1, *(1,) * (y.ndim - 1)), y.shape)
        else:
            raise ValueError("weights must be None, 'relative', 'relative_norm' or an array")
    else:
        w = _real(weights, "weights")
        if w.shape == y.shape[:1]:
            w = w.reshape(-1, *(1,) * (y.ndim - 1))
        w = np.broadcast_to(w, y.shape)
    if np.any(w < 0) or not np.all(w.max(axis=0) > 0):
        raise ValueError("weights must be nonnegative and positive somewhere in every channel")
    return w / w.max()


def _pack(a, symmetric):
    """(K, *shape) as (K, channels); symmetric keeps the upper triangle."""
    if symmetric:
        i, j = np.triu_indices(a.shape[-1])
        return a[:, i, j]
    return a.reshape(len(a), -1)


def _unpack(x, shape, symmetric):
    if not symmetric:
        return x.reshape(len(x), *shape)
    i, j = np.triu_indices(shape[0])
    out = np.empty((len(x), *shape), x.dtype)
    out[:, i, j] = out[:, j, i] = x
    return out


def _symmetric_weights(n):
    """Weights of packed upper-triangle entries that preserve the Frobenius norm."""
    i, j = np.triu_indices(n)
    return np.where(i == j, 1.0, np.sqrt(2.0))


def _basis(s, poles, asymptote, groups=None, delays=None):
    """Real-coefficient basis: real poles, pair real parts, pair imaginary parts, constants, s."""
    z = s[:, None]
    phase = np.exp(-z * (np.zeros(1) if delays is None else delays))
    groups = np.zeros(len(poles), np.intp) if groups is None else groups
    real = poles.imag == 0
    a = phase[:, groups] / (z - poles)
    b = phase[:, groups[~real]] / (z - poles[~real].conj())
    columns = [a[:, real], a[:, ~real] + b, 1j * (a[:, ~real] - b)]
    if asymptote != "none":
        columns.append(phase)
    if asymptote == "linear":
        columns.append(z)
    return np.hstack(columns)


def _assemble(x, poles, asymptote, count):
    """Residues, the count delayed constants and the s-term from coefficients x of _basis."""
    real = poles.imag == 0
    nr, nc = int(real.sum()), int((~real).sum())
    residues = np.empty((len(poles), *x.shape[1:]), complex)
    residues[real] = x[:nr]
    residues[~real] = x[nr : nr + nc] + 1j * x[nr + nc : nr + 2 * nc]
    k = nr + 2 * nc
    d = x[k : k + count] if asymptote != "none" else np.zeros((count, *x.shape[1:]))
    e = x[-1] if asymptote == "linear" else np.zeros(x.shape[1:])
    return residues, d, e


def _stack(a):
    return np.vstack((a.real, a.imag))


def _lstsq(a, b, row=None, value=None):
    """Column-equilibrated minimum-norm least squares, optionally subject to row @ x = value."""
    if row is not None:
        norm = row @ row
        if norm:
            x0 = np.outer(row / norm, value)
            null = la.null_space(row[None])
            return x0 + null @ _lstsq(a @ null, b - a @ x0)
        if np.any(value):
            raise ValueError("dc cannot be matched without a constant or pole term")
    if not a.shape[1]:
        return np.zeros((0, b.shape[1]))
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1
    return la.lstsq(a / scale, b, cond=_RCOND, lapack_driver="gelsy")[0] / scale[:, None]


def _range(a):
    """Orthonormal basis for the numerical range of a."""
    scale = np.linalg.norm(a, axis=0)
    scale[scale == 0] = 1
    q, r, _ = la.qr(a / scale, mode="economic", pivoting=True)
    diag = np.abs(np.diag(r))
    return q[:, : np.count_nonzero(diag > _RCOND * diag[0])]


def _sigma_zeros(poles, z):
    """Real matrix whose eigenvalues are the zeros of sigma = z[-1] + sum_k z_k basis_k."""
    real = poles.imag == 0
    pairs = poles[~real]
    nr, nc = int(real.sum()), len(pairs)
    n = nr + 2 * nc
    a, b, c = np.zeros((n, n)), np.zeros(n), np.zeros(n)
    a[:nr, :nr] = np.diag(poles[real].real)
    i = nr + 2 * np.arange(nc)
    j = i + 1
    a[i, i] = a[j, j] = pairs.real
    a[i, j], a[j, i] = -pairs.imag, pairs.imag
    b[:nr], b[i] = 1, 1
    c[:nr], c[i], c[j] = z[:nr], 2 * z[nr : nr + nc], -2 * z[nr + nc : n]
    return a - np.outer(b, c) / z[-1]
