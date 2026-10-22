from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, NamedTuple, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import linalg as la
from scipy.optimize import linear_sum_assignment, minimize_scalar

from ._fit import _FLOOR, FitReport, fit, fit_residues
from ._model import PoleResidue, Result, _frequencies, _real


class LineFunctions(NamedTuple):
    """Characteristic admittance, current-wave propagation and modal propagation constants."""

    yc: NDArray[np.complex128]
    h: NDArray[np.complex128]
    gamma: NDArray[np.complex128]


@dataclass(frozen=True)
class LineModel:
    """Characteristic admittance yc and current-wave propagation h of a uniform line."""

    yc: PoleResidue
    h: PoleResidue

    def __post_init__(self) -> None:
        shape = self.yc.shape
        if len(shape) != 2 or shape[0] != shape[1] or self.h.shape != shape:
            raise ValueError("yc and h must be square with equal shapes")
        if np.any(self.yc.delays) or np.any(self.h.e):
            raise ValueError("yc must be undelayed and h proper")

    def to_dict(self) -> dict[str, Any]:
        return {"yc": self.yc.to_dict(), "h": self.h.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(PoleResidue.from_dict(data["yc"]), PoleResidue.from_dict(data["h"]))


@dataclass(frozen=True)
class LineReport:
    """groups lists the modes sharing each delay of h."""

    yc: FitReport
    h: FitReport
    groups: tuple[tuple[int, ...], ...]


def line_functions(z: ArrayLike, y: ArrayLike, length: float) -> LineFunctions:
    """Yc = Gamma Z^-1 and H = exp(-length Gamma) with Gamma = sqrt(Y Z), from per-metre Z and Y.

    Samples are taken at s = j omega with omega >= 0, or with positive damping; modes are ordered
    by eigenvector continuity between consecutive samples.
    """
    z, y = np.asarray(z, complex), np.asarray(y, complex)
    if z.ndim != 3 or z.shape[1] != z.shape[2] or y.shape != z.shape:
        raise ValueError("z and y must be square matrix samples of equal shape")
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be positive and finite")
    # sqrt with its branch cut on the negative imaginary axis: Gamma stays in the first quadrant
    rotation = np.exp(0.25j * np.pi)
    yc, h, gamma = np.empty_like(z), np.empty_like(z), np.empty(z.shape[:2], complex)
    previous = None
    for k, (zk, yk) in enumerate(zip(z, y, strict=True)):
        root = rotation * la.sqrtm(-1j * (yk @ zk))
        yc[k] = la.solve(zk.T, root.T).T
        h[k] = la.expm(-length * root)
        values, vectors = la.eig(yk @ zk)
        modes = rotation * np.sqrt(-1j * values)
        vectors /= np.linalg.norm(vectors, axis=0)
        if previous is None:
            order = np.argsort(modes.imag)
        else:
            order = linear_sum_assignment(-np.abs(previous.conj().T @ vectors))[1]
        previous, gamma[k] = vectors[:, order], modes[order]
    return LineFunctions(yc, h, gamma)


def terminal_admittance(yc: ArrayLike, h: ArrayLike) -> NDArray[np.complex128]:
    """Nodal admittance of a line from Yc and current-wave H samples, currents into both ends."""
    yc, h = np.asarray(yc, complex), np.asarray(h, complex)
    if yc.shape != h.shape or yc.ndim < 2 or yc.shape[-1] != yc.shape[-2]:
        raise ValueError("yc and h must be square matrix samples of equal shape")
    zero = np.zeros_like(yc)
    return _waves(h) @ np.block([[yc, zero], [zero, yc]])


def _waves(h):
    """K such that the terminal admittance is K blockdiag(Yc, Yc) for propagation h."""
    eye = np.eye(h.shape[-1])
    h2 = h @ h
    near = np.linalg.solve(eye - h2, eye + h2)
    far = np.linalg.solve(eye - h2, -2 * h)
    return np.block([[near, far], [far, near]])


def fit_line(
    f: ArrayLike,
    z: ArrayLike,
    y: ArrayLike,
    length: float,
    *,
    yc_poles: int | ArrayLike = 12,
    h_poles: int = 12,
    delays: Literal["search", "phase"] | ArrayLike = "search",
    group_tol: float = 0.01,
    h_asymptote: Literal["none", "constant"] = "none",
    iterations: int = 20,
    tol: float = 1e-3,
) -> Result[LineModel, LineReport]:
    """Universal-line-model fit from per-metre Z and Y sampled at frequencies f in Hz.

    Yc is fitted in the phase domain with relative weights. Modes whose delays agree within
    group_tol share one delay and h_poles poles, identified from their delay-removed modal
    propagation functions; H is then fitted in the phase domain with all poles and delays fixed.
    H residuals are weighted by 1 / ||I - H^2||, the relative sensitivity of the terminal
    admittance to H. delays is "search" (minimize the modal residual), "phase" (phase delay at
    the highest frequency) or one delay per mode.
    """
    f = _frequencies(f)
    yc, h, gamma = line_functions(z, y, length)
    if len(gamma) != len(f):
        raise ValueError("z and y must have one sample per frequency")
    if isinstance(delays, str) and delays not in ("search", "phase"):
        raise ValueError("delays must be 'search', 'phase' or one delay per mode")
    phase = length * gamma[-1].imag / (2 * np.pi * f[-1])
    estimate = phase if isinstance(delays, str) else _real(delays, "delays")
    if estimate.shape != phase.shape or np.any(estimate < 0):
        raise ValueError("delays must hold one nonnegative delay per mode")
    yc_fit = fit(f, yc, poles=yc_poles, symmetric=True, iterations=iterations, tol=tol)
    modal = np.exp(-length * gamma)
    groups = _groups(estimate, group_tol)
    poles, assignment, group_delays = [], [], []
    for index, modes in enumerate(map(list, groups)):
        modal_fit = partial(
            fit,
            f,
            modal[:, modes],
            poles=h_poles,
            asymptote=h_asymptote,
            weights=_sensitivity(modal[:, modes]),
            iterations=iterations,
            tol=tol,
        )
        tau = float(estimate[modes].min())
        if isinstance(delays, str) and delays == "search":
            tau = _search_delay(modal_fit, _delay_bounds(f, gamma[:, modes], length))
        p = modal_fit(delay=tau).model.poles
        poles.append(p)
        assignment += [index] * len(p)
        group_delays.append(tau)
    h_fit = fit_residues(
        f,
        h,
        poles=np.concatenate(poles),
        delays=group_delays,
        groups=assignment,
        asymptote=h_asymptote,
        weights=_sensitivity(h),
    )
    model = LineModel(yc_fit.model, h_fit.model)
    return Result(model, LineReport(yc_fit.report, h_fit.report, groups))


def _sensitivity(h):
    """Per-frequency 1 / ||I - H^2|| for matrix samples, or 1 / ||1 - h^2|| over modal samples."""
    square = np.eye(h.shape[-1]) - h @ h if h.ndim == 3 else 1 - h**2
    norm = np.linalg.norm(square.reshape(len(h), -1), axis=1)
    return 1 / np.maximum(norm, _FLOOR * norm.max())


def _groups(delays, tol):
    """Modes ordered by delay, split where a delay exceeds its group's smallest by more than tol."""
    if not np.isfinite(tol) or tol < 0:
        raise ValueError("group_tol must be nonnegative and finite")
    groups: list[list[int]] = []
    for mode in np.argsort(delays, kind="stable"):
        if groups and delays[mode] <= delays[groups[-1][0]] * (1 + tol):
            groups[-1].append(int(mode))
        else:
            groups.append([int(mode)])
    return tuple(tuple(g) for g in groups)


def _delay_bounds(f, gamma, length):
    """From 0.9 x the phase delay at the highest frequency to the phase delay at the highest
    frequency where each modal function still exceeds the weighting floor."""
    positive = f > 0
    phase = length * gamma[positive].imag / (2 * np.pi * f[positive, None])
    magnitude = np.exp(-length * gamma[positive].real)
    last = [np.flatnonzero(m >= _FLOOR * m.max())[-1] for m in magnitude.T]
    return 0.9 * phase[-1].min(), max(phase[last, range(len(last))].max(), phase[-1].max())


def _search_delay(modal_fit, bounds):
    """Delay minimizing the weighted residual of the delay-removed modal fit.

    Two nested scans locate the narrow valley below the causality limit; Brent refines it.
    """

    def residual(tau):
        return min(modal_fit(delay=tau).report.history)

    for _ in range(2):
        grid = np.linspace(bounds[0], bounds[1], 25)
        scores = [residual(tau) for tau in grid]
        k = int(np.argmin(scores))
        bounds = (grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)])
    refined = minimize_scalar(
        residual, bounds=bounds, method="bounded", options={"xatol": 1e-9 * bounds[1]}
    )
    return float(refined.x if refined.fun < scores[k] else grid[k])
