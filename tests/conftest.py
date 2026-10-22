import numpy as np
import pytest

import pyvectorfit as vf


@pytest.fixture
def rng():
    return np.random.default_rng(20261021)


@pytest.fixture
def f():
    return np.geomspace(1e-2, 1e4, 161)


@pytest.fixture
def model(rng):
    """Factory of stable models with random residues and poles inside the band of f."""

    def make(shape=(), real=2, pairs=2, *, d=True, e=False, symmetric=False, delays=(0.0,)):
        poles = np.r_[-np.geomspace(1.0, 1e3, real), np.geomspace(10.0, 1e4, pairs) * (-0.1 + 1j)]
        size = (len(poles), *shape)
        pair = (poles.imag > 0).reshape(-1, *(1,) * len(shape))
        residues = rng.normal(size=size) + 1j * pair * rng.normal(size=size)
        constant = rng.normal(size=(len(delays), *shape)) * d
        slope = 1e-3 * rng.normal(size=shape) * e
        if symmetric:
            residues, constant, slope = (
                a + np.swapaxes(a, -1, -2) for a in (residues, constant, slope)
            )
        groups = np.arange(len(poles)) % len(delays)
        return vf.PoleResidue(poles, residues, constant, slope, delays, groups)

    return make


@pytest.fixture
def rlgc():
    """Per-metre Z(s) and Y(s) of a noncommuting three-conductor line with constant parameters."""
    r = 1e-5 * np.array([[3.0, 0.5, 0.3], [0.5, 2.5, 0.4], [0.3, 0.4, 2.8]])
    inductance = 1e-6 * np.array([[1.3, 0.32, 0.17], [0.32, 1.1, 0.29], [0.17, 0.29, 1.2]])
    g = 1e-11 * np.eye(3)
    capacitance = 1e-11 * np.array([[1.8, -0.32, -0.18], [-0.32, 1.6, -0.22], [-0.18, -0.22, 1.7]])

    def zy(s):
        s = np.asarray(s, complex)[:, None, None]
        return r + s * inductance, g + s * capacitance

    return zy
