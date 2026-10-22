from typing import Any, Generic, NamedTuple, Self, TypeVar, final

import numpy as np
from numpy.typing import ArrayLike, NDArray

M = TypeVar("M")
R = TypeVar("R")

_INVERSE_FACTORIALS = 1 / np.cumprod(np.arange(1.0, 20.0))


class Result(NamedTuple, Generic[M, R]):
    """A model and the report describing how it was obtained."""

    model: M
    report: R


class Convolution(NamedTuple):
    """Recursion x[n] = a x[n-1] + b0 u[n-1] + b1 u[n] with one entry per stored pole."""

    a: NDArray[np.complex128]
    b0: NDArray[np.complex128]
    b1: NDArray[np.complex128]


def _real(a: ArrayLike, name: str) -> NDArray[np.float64]:
    a = np.asarray(a)
    if np.iscomplexobj(a) and np.any(a.imag):
        raise ValueError(f"{name} must be real")
    a = a.real.astype(float)
    if not np.isfinite(a).all():
        raise ValueError(f"{name} must be finite")
    return a


def _frequencies(f: ArrayLike) -> NDArray[np.float64]:
    f = _real(f, "f")
    if f.ndim != 1 or not f.size or f[0] < 0 or np.any(np.diff(f) <= 0):
        raise ValueError("f must be increasing nonnegative frequencies in Hz")
    return f


def _layout(poles: ArrayLike, delays: ArrayLike, groups: ArrayLike | None):
    p = np.asarray(poles, complex)
    t = _real(delays, "delays")
    g = np.zeros(p.shape, np.intp) if groups is None else np.asarray(groups)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any(p.real >= 0) or np.any(p.imag < 0):
        raise ValueError("poles must be finite and stable, with each complex pair stored once")
    if t.ndim != 1 or not t.size or np.any(t < 0):
        raise ValueError("delays must be a nonempty vector of nonnegative seconds")
    if g.shape != p.shape or (
        g.size and (g.dtype.kind not in "iu" or g.min() < 0 or g.max() >= t.size)
    ):
        raise ValueError("groups must assign each pole to a delay index")
    return p, t, g.astype(np.intp)


@final
class PoleResidue:
    """Delayed pole-residue model.

    F(s) = s e + sum_g exp(-s delays[g]) (d[g] + sum_{k in g} residues[k] / (s - poles[k])).
    A complex pole is stored once, with positive imaginary part, and implies its conjugate pair.
    Shapes: poles (N,), residues (N, *shape), d (G, *shape), e shape, delays (G,), groups (N,).
    """

    __slots__ = ("d", "delays", "e", "groups", "poles", "residues")

    poles: NDArray[np.complex128]
    residues: NDArray[np.complex128]
    d: NDArray[np.float64]
    e: NDArray[np.float64]
    delays: NDArray[np.float64]
    groups: NDArray[np.intp]

    def __init__(
        self,
        poles: ArrayLike,
        residues: ArrayLike,
        d: ArrayLike = 0.0,
        e: ArrayLike = 0.0,
        delays: ArrayLike = (0.0,),
        groups: ArrayLike | None = None,
    ) -> None:
        p, t, g = _layout(poles, delays, groups)
        r = np.asarray(residues, complex)
        if (
            r.ndim not in (1, 2, 3)
            or len(r) != len(p)
            or 0 in r.shape[1:]
            or not np.isfinite(r).all()
        ):
            raise ValueError("residues must be finite with shape (len(poles), *shape)")
        if np.any(r[p.imag == 0].imag):
            raise ValueError("real poles require real residues")
        shape = r.shape[1:]
        for name, value in (
            ("poles", p),
            ("residues", r),
            ("d", np.broadcast_to(_real(d, "d"), (t.size, *shape))),
            ("e", np.broadcast_to(_real(e, "e"), shape)),
            ("delays", t),
            ("groups", g),
        ):
            value = np.array(value)
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("PoleResidue is immutable")

    @property
    def shape(self) -> tuple[int, ...]:
        return self.residues.shape[1:]

    @property
    def order(self) -> int:
        """Number of poles, counting implied conjugates."""
        return len(self.poles) + int(np.count_nonzero(self.poles.imag))

    def __call__(self, s: ArrayLike) -> NDArray[np.complex128]:
        s = np.asarray(s, complex)
        z = s.reshape(-1, 1)
        phase = np.exp(-z * self.delays)
        pair = self.poles.imag > 0
        r = self.residues.reshape(len(self.poles), int(np.prod(self.shape)))
        value = (
            (phase[:, self.groups] / (z - self.poles)) @ r
            + (phase[:, self.groups[pair]] / (z - self.poles[pair].conj())) @ r[pair].conj()
            + phase @ self.d.reshape(len(self.delays), -1)
            + z * self.e.reshape(1, -1)
        )
        return value.reshape(s.shape + self.shape)

    def response(self, f: ArrayLike) -> NDArray[np.complex128]:
        """Evaluate at s = j 2 pi f for frequencies f in Hz."""
        return self(2j * np.pi * _real(f, "f"))

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible representation."""
        return {
            "version": 1,
            "shape": list(self.shape),
            "poles": _split(self.poles),
            "residues": _split(self.residues),
            "d": self.d.tolist(),
            "e": self.e.tolist(),
            "delays": self.delays.tolist(),
            "groups": self.groups.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        if data.get("version") != 1:
            raise ValueError("unsupported PoleResidue format")
        poles = _join(data["poles"])
        residues = _join(data["residues"]).reshape(len(poles), *data["shape"])
        return cls(poles, residues, data["d"], data["e"], data["delays"], data["groups"])

    def __repr__(self) -> str:
        return f"PoleResidue(order={self.order}, shape={self.shape}, delays={self.delays.tolist()})"


def _split(a: NDArray[np.complex128]) -> dict[str, Any]:
    return {"real": a.real.tolist(), "imag": a.imag.tolist()}


def _join(data: dict[str, Any]) -> NDArray[np.complex128]:
    return np.asarray(data["real"], float) + 1j * np.asarray(data["imag"], float)


def recursive_convolution(model: PoleResidue, dt: float) -> Convolution:
    """Exact recursion for each pole state x' = p x + u, with u linear over each step dt.

    A real pole contributes residue @ x to the output and a stored complex pole 2 Re(residue @ x).
    """
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    z = model.poles * dt
    phi1, phi2 = np.empty_like(z), np.empty_like(z)
    small = np.abs(z) < 0.5
    phi1[small] = np.polyval(_INVERSE_FACTORIALS[16::-1], z[small])
    phi2[small] = np.polyval(_INVERSE_FACTORIALS[17:0:-1], z[small])
    q = z[~small]
    phi1[~small] = np.expm1(q) / q
    phi2[~small] = (np.expm1(q) - q) / q**2
    return Convolution(np.exp(z), dt * (phi1 - phi2), dt * phi2)
