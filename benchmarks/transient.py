"""Line energization: method-of-characteristics simulation and numerical Laplace inversion.

The source 1 - exp(-t / rise) drives conductor 0 at the sending end through its termination;
terminations are conductances to earth at each end, 0 meaning open.
"""

import numpy as np
from scipy import linalg as la

import pyvectorfit as vf


def terminations(conductors):
    """Phases and cores: 10 ohm at the sending end, open at the receiving end. Sheaths: 0.1 ohm."""
    sheath = np.array([c.startswith("Sheath") for c in conductors])
    return np.where(sheath, 10.0, 0.1), np.where(sheath, 10.0, 0.0)


def frequencies(dt, steps):
    """Damped, half-shifted frequencies of the inversion over steps samples of dt."""
    window = dt * steps
    return np.log(1e6) / (2 * window) + 1j * np.pi / window * (np.arange(steps) + 0.5)


def response(yc, h, s, rise, sending, receiving):
    """Receiving-end voltage transforms from Yc and H samples at s."""
    nodal = vf.terminal_admittance(yc, h) + np.diag(np.r_[sending, receiving])
    drive = np.zeros((len(s), len(nodal[0]), 1), complex)
    drive[:, 0, 0] = sending[0] * (1 / s - 1 / (s + 1 / rise))
    return np.linalg.solve(nodal, drive)[:, len(sending) :, 0]


def invert(values, s, dt):
    """Time samples from transforms at frequencies(dt, len(values)), Hanning-windowed."""
    steps = len(values)
    t = dt * np.arange(steps)
    step = s[1].imag - s[0].imag
    window = 0.5 * (1 + np.cos(np.pi * (np.arange(steps) + 0.5) / steps))
    total = 2 * steps * np.fft.ifft(values * window[:, None], n=2 * steps, axis=0)[:steps]
    return (np.exp(s[0].real * t) * step / np.pi)[:, None] * (
        np.exp(0.5j * step * t)[:, None] * total
    ).real


def simulate(line, dt, steps, rise, sending, receiving):
    """Receiving-end voltages by the method of characteristics with recursive convolution."""
    lag = np.floor(line.h.delays / dt).astype(int)
    if lag.min() < 1:
        raise ValueError("dt must not exceed the smallest delay")
    fraction = line.h.delays / dt - lag
    yc = [_Convolution(line.yc, dt) for _ in range(2)]
    h = [_Convolution(line.h, dt) for _ in range(2)]
    gain = yc[0].gain
    solvers = [la.lu_factor(gain + np.diag(g)) for g in (sending, receiving)]
    n = len(gain)
    reflected = np.zeros((2, steps + 1, n))
    voltage = np.empty((steps, n))
    source = sending[0] * (1 - np.exp(-dt * np.arange(steps) / rise))
    for k in range(steps):
        newer, older = np.maximum(k - lag + 1, 0), np.maximum(k - lag, 0)
        for end, drive in ((0, source[k]), (1, 0.0)):
            far = reflected[1 - end]
            incident = h[end].output(
                (1 - fraction)[:, None] * far[newer] + fraction[:, None] * far[older]
            )
            injection = np.zeros(n)
            injection[0] = drive
            history = yc[end].history()
            v = la.lu_solve(solvers[end], injection - history + 2 * incident)
            yc[end].advance(v[None])
            reflected[end, k + 1] = gain @ v + history - incident
            if end:
                voltage[k] = v
    return voltage


class _Convolution:
    """Recursive convolution of a PoleResidue with one input vector per delay group."""

    def __init__(self, model, dt):
        a, b0, b1 = vf.recursive_convolution(model, dt)
        weight = np.where(model.poles.imag > 0, 2.0, 1.0)[:, None, None]
        self.a, self.b0, self.b1 = a[:, None], b0[:, None], b1[:, None]
        self.r, self.d, self.groups = weight * model.residues, model.d, model.groups
        self.x = np.zeros((len(a), model.shape[0]), complex)
        self.u = np.zeros((len(model.delays), model.shape[0]))

    @property
    def gain(self):
        """Gain from the present input of an undelayed model."""
        return self.d[0] + np.einsum("kij,k->ij", self.r, self.b1[:, 0]).real

    def history(self):
        """Output from the states and the previous input."""
        return np.einsum("kij,kj->i", self.r, self.a * self.x + self.b0 * self.u[self.groups]).real

    def output(self, u):
        """Output for the present inputs u, after which the states advance."""
        present = np.einsum("kij,kj->i", self.r, self.b1 * u[self.groups]).real
        y = self.history() + present + np.einsum("gij,gj->i", self.d, u)
        self.advance(u)
        return y

    def advance(self, u):
        self.x = self.a * self.x + self.b0 * self.u[self.groups] + self.b1 * u[self.groups]
        self.u = u
