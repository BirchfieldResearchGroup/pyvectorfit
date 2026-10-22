"""Per-metre Z(s) and Y(s) of overhead lines and buried single-core cables (benchmark data only).

Overhead earth return: Deri complex depth. Buried earth return: Saad, Gaba and Giroux closed form.
Conductors: Schelkunoff solid and tubular internal impedance. All functions accept complex s.
"""

from dataclasses import dataclass, replace

import numpy as np
from scipy import special
from scipy.constants import epsilon_0, mu_0


@dataclass(frozen=True)
class Wire:
    """Overhead conductor in m and ohm/m; phase -1 is a continuously grounded shield wire."""

    x: float
    height: float
    radius: float
    inner: float
    rdc: float
    phase: int


@dataclass(frozen=True)
class Overhead:
    """Wires over earth of resistivity earth (ohm m), with shunt conductance per phase (S/m)."""

    wires: tuple[Wire, ...]
    earth: float
    conductance: float = 1e-11

    @property
    def conductors(self):
        return [f"Phase {k + 1}" for k in range(max(w.phase for w in self.wires) + 1)]

    def zy(self, s):
        s = np.asarray(s, complex)[:, None, None]
        x, h, r = (np.array([getattr(w, k) for w in self.wires]) for k in ("x", "height", "radius"))
        dx = x[:, None] - x[None, :]
        distance = np.hypot(dx, h[:, None] - h[None, :]) + np.diag(r)
        potential = np.log(np.hypot(dx, h[:, None] + h[None, :]) / distance) / (
            2 * np.pi * epsilon_0
        )
        depth = np.sqrt(self.earth / (s * mu_0))
        image = np.sqrt(dx**2 + (h[:, None] + h[None, :] + 2 * depth) ** 2)
        z = s * mu_0 / (2 * np.pi) * np.log(image / distance)
        for k, w in enumerate(self.wires):
            rho = w.rdc * np.pi * (w.radius**2 - w.inner**2)
            z[:, k, k] += _tube(s[:, 0, 0], rho, w.inner, w.radius)[1]
        phases = np.array([w.phase for w in self.wires])
        a = (phases[:, None] == np.arange(phases.max() + 1)).astype(float)
        y = s * np.linalg.inv(_merge(potential, a)) + self.conductance * np.eye(a.shape[1])
        return _merge(z, a), y


@dataclass(frozen=True)
class Cable:
    """Single-core cable radii (m), resistivities (ohm m) and permittivities.

    Dielectric loss is a constant conductance equal to tan_delta at 50 Hz.
    """

    core: float
    sheath_inner: float
    sheath_outer: float
    jacket: float
    core_rho: float
    sheath_rho: float
    insulation: float
    jacket_eps: float
    tan_delta: float = 4e-4


@dataclass(frozen=True)
class Buried:
    """Identical cables at (x, depth) in m; conductors ordered core, sheath of each cable."""

    cable: Cable
    positions: tuple[tuple[float, float], ...]
    earth: float

    @property
    def conductors(self):
        return [
            f"{part} {k + 1}" for k in range(len(self.positions)) for part in ("Core", "Sheath")
        ]

    def zy(self, s):
        c, s = self.cable, np.asarray(s, complex)
        inductance = s * mu_0 / (2 * np.pi)
        inner, outer, transfer = _tube(s, c.sheath_rho, c.sheath_inner, c.sheath_outer)
        core = _tube(s, c.core_rho, 0.0, c.core)[1] + inductance * np.log(c.sheath_inner / c.core)
        sheath = outer + inductance * np.log(c.jacket / c.sheath_outer)
        cable = np.empty((len(s), 2, 2), complex)
        cable[:, 0, 0] = core + inner - 2 * transfer + sheath
        cable[:, 0, 1] = cable[:, 1, 0] = sheath - transfer
        cable[:, 1, 1] = sheath
        n = len(self.positions)
        earth = _buried_earth(s, self.earth, np.array(self.positions), c.jacket)
        z = np.kron(earth, np.ones((2, 2))) + _blocks(cable, n)
        insulation = 2 * np.pi * epsilon_0 * c.insulation / np.log(c.sheath_inner / c.core)
        jacket = 2 * np.pi * epsilon_0 * c.jacket_eps / np.log(c.jacket / c.sheath_outer)
        potential = np.array([[1 / insulation + 1 / jacket, 1 / jacket], [1 / jacket, 1 / jacket]])
        y = (s[:, None, None] + 2 * np.pi * 50 * c.tan_delta) * np.linalg.inv(potential)
        return z, _blocks(y, n)


def _blocks(a, n):
    """Block-diagonal samples with n copies of the (S, 2, 2) samples a."""
    return np.einsum("ij,skl->sikjl", np.eye(n), a).reshape(len(a), 2 * n, 2 * n)


def _tube(s, rho, inner, outer):
    """Inner-surface, outer-surface and transfer impedance per metre; inner = 0 is solid."""
    m = np.sqrt(s * mu_0 / rho)
    b = m * outer
    if not inner:
        surface = rho * m * special.ive(0, b) / (2 * np.pi * outer * special.ive(1, b))
        return np.inf, surface, 0.0
    a = m * inner
    c = np.exp(a - b + (a - b).real)
    i0a, i1a, i0b, i1b = (special.ive(v, x) for v, x in ((0, a), (1, a), (0, b), (1, b)))
    k0a, k1a, k0b, k1b = (special.kve(v, x) for v, x in ((0, a), (1, a), (0, b), (1, b)))
    det = i1b * k1a - i1a * k1b * c
    z_inner = rho * m * (i0a * k1b * c + k0a * i1b) / (2 * np.pi * inner * det)
    z_outer = rho * m * (i0b * k1a + k0b * i1a * c) / (2 * np.pi * outer * det)
    z_transfer = rho * np.exp(a - b.real) / (2 * np.pi * inner * outer * det)
    return z_inner, z_outer, z_transfer


def _buried_earth(s, rho, xy, radius):
    """Self and mutual earth-return impedance per metre of conductors buried at xy."""
    m = np.sqrt(s * mu_0 / rho)[:, None, None]
    x = np.abs(xy[:, None, 0] - xy[None, :, 0])
    depth = xy[:, None, 1] + xy[None, :, 1]
    d = np.hypot(x, xy[:, None, 1] - xy[None, :, 1])
    np.fill_diagonal(d, radius)
    np.fill_diagonal(x, radius)
    bessel = special.kve(0, m * d) * np.exp(-m * d)
    return rho * m**2 / (2 * np.pi) * (bessel + 2 * np.exp(-depth * m) / (4 + (m * x) ** 2))


def _merge(m, a):
    """(a^T m^-1 a)^-1: bundles conductors of one phase and eliminates grounded wires."""
    return np.linalg.inv(a.T @ np.linalg.inv(m) @ a)


def _bundle(x, height, count, spacing, wire, phase):
    radius = spacing / (2 * np.sin(np.pi / count))
    angles = 2 * np.pi * np.arange(count) / count + np.pi / 4
    return tuple(
        replace(wire, x=x + radius * np.cos(t), height=height + radius * np.sin(t), phase=phase)
        for t in angles
    )


def _at(wire, x, height, phase):
    return replace(wire, x=x, height=height, phase=phase)


DRAKE = Wire(0.0, 0.0, 0.01407, 0.00520, 7.17e-5, 0)
RAIL = Wire(0.0, 0.0, 0.01480, 0.00370, 5.97e-5, 0)
ALUMOWELD = Wire(0.0, 0.0, 0.00489, 0.0, 1.55e-3, -1)
XLPE = Cable(0.0195, 0.0455, 0.0480, 0.0525, 2.0e-8, 2.83e-8, 2.85, 2.3)


@dataclass(frozen=True)
class Case:
    title: str
    system: Overhead | Buried
    length: float
    yc_poles: int = 12
    h_poles: int = 12
    band: tuple[float, float] = (0.1, 1e6)


HORIZONTAL = Overhead(
    (
        _at(DRAKE, -10.0, 20.0, 0),
        _at(DRAKE, 0.0, 20.0, 1),
        _at(DRAKE, 10.0, 20.0, 2),
        _at(ALUMOWELD, -6.0, 30.0, -1),
        _at(ALUMOWELD, 6.0, 30.0, -1),
    ),
    earth=100.0,
)
BUNDLED = Overhead(
    (
        *_bundle(-12.0, 25.0, 4, 0.457, RAIL, 0),
        *_bundle(0.0, 25.0, 4, 0.457, RAIL, 1),
        *_bundle(12.0, 25.0, 4, 0.457, RAIL, 2),
        _at(ALUMOWELD, -8.0, 37.0, -1),
        _at(ALUMOWELD, 8.0, 37.0, -1),
    ),
    earth=1000.0,
)
DOUBLE = Overhead(
    (
        *(_at(DRAKE, -6.0, h, p) for h, p in ((20.0, 0), (28.0, 1), (36.0, 2))),
        *(_at(DRAKE, 6.0, h, p) for h, p in ((20.0, 5), (28.0, 4), (36.0, 3))),
        _at(ALUMOWELD, 0.0, 44.0, -1),
    ),
    earth=100.0,
)

CASES = {
    "single": Case(
        "Single conductor, 50 km", Overhead((_at(DRAKE, 0.0, 20.0, 0),), 100.0), 50e3, 10, 10
    ),
    "horizontal": Case("Three-phase, 100 km", HORIZONTAL, 100e3),
    "bundled": Case("Bundled 500 kV, 300 km", BUNDLED, 300e3, 12, 16),
    "double": Case("Double circuit, 150 km", DOUBLE, 150e3, 14),
    "cable_flat": Case(
        "Cables in flat, 10 km",
        Buried(XLPE, ((-0.3, 1.2), (0.0, 1.2), (0.3, 1.2)), 100.0),
        10e3,
        16,
        14,
    ),
    "cable_trefoil": Case(
        "Cables in trefoil, 5 km",
        Buried(XLPE, ((-0.0525, 1.15), (0.0525, 1.15), (0.0, 1.2409)), 20.0),
        5e3,
        16,
        14,
    ),
}
