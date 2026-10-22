import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg as la

import pyvectorfit as vf

SHAPES = [(), (3,), (2, 2)]
S = np.array([0.3j, 2.0 + 5.0j, 40j, 1e3 + 1e4j])


@pytest.mark.parametrize("shape", SHAPES)
def test_evaluation_is_the_partial_fraction_sum(model, shape):
    m = model(shape, e=True, delays=(0.0, 1e-3))
    expected = np.zeros((len(S), *shape), complex)
    for k, s in enumerate(S):
        expected[k] = s * m.e
        for g, tau in enumerate(m.delays):
            value = m.d[g].astype(complex)
            for p, r in zip(m.poles[m.groups == g], m.residues[m.groups == g], strict=True):
                value = value + r / (s - p) + (r.conj() / (s - p.conj()) if p.imag else 0)
            expected[k] += np.exp(-s * tau) * value
    assert_allclose(m(S), expected, rtol=1e-13)
    assert m(S[1:3].reshape(2, 1)).shape == (2, 1, *shape)


@pytest.mark.parametrize("shape", SHAPES)
def test_response_is_real_rational(model, shape, f):
    m = model(shape, delays=(0.0, 2e-3))
    assert_allclose(m(S.conj()), m(S).conj(), rtol=1e-13)
    assert_allclose(m.response(f), m(2j * np.pi * f))


def test_order_counts_implied_conjugates(model):
    assert model(real=3, pairs=2).order == 7


def test_model_is_immutable(model):
    m = model()
    with pytest.raises(ValueError, match="read-only"):
        m.poles[0] = -1.0
    with pytest.raises(AttributeError):
        m.d = np.zeros(1)


@pytest.mark.parametrize(
    ("poles", "residues", "options", "message"),
    [
        ([1.0], [1.0], {}, "stable"),
        ([-1 - 2j], [1.0], {}, "stored once"),
        ([-1.0], [1j], {}, "real residues"),
        ([np.nan], [1.0], {}, "finite"),
        ([-1.0], [1.0, 2.0], {}, "len"),
        ([-1.0], [1.0], {"delays": [-1e-3]}, "delays"),
        ([-1.0], [1.0], {"groups": [1]}, "groups"),
        ([-1.0], [1.0], {"groups": [0.5]}, "groups"),
        ([-1.0], [1.0], {"d": 1j}, "real"),
    ],
    ids=[
        "unstable",
        "lower-half-plane",
        "complex-residue-of-real-pole",
        "nonfinite",
        "residue-count",
        "negative-delay",
        "group-index",
        "fractional-group",
        "complex-d",
    ],
)
def test_rejects_invalid_coefficients(poles, residues, options, message):
    with pytest.raises(ValueError, match=message):
        vf.PoleResidue(poles, residues, **options)


@pytest.mark.parametrize("shape", SHAPES)
def test_dict_round_trip(model, shape, f):
    m = model(shape, e=True, delays=(0.0, 1e-3))
    copy = vf.PoleResidue.from_dict(json.loads(json.dumps(m.to_dict())))
    assert_allclose(copy.response(f), m.response(f), rtol=1e-15)
    assert_allclose(copy.groups, m.groups)


def test_round_trip_without_poles(f):
    m = vf.PoleResidue([], np.zeros((0, 2, 2)), d=np.eye(2), delays=[1e-3])
    copy = vf.PoleResidue.from_dict(json.loads(json.dumps(m.to_dict())))
    assert copy.shape == (2, 2)
    assert_allclose(copy.response(f), np.exp(-2j * np.pi * f * 1e-3)[:, None, None] * np.eye(2))


def test_rejects_unknown_format(model):
    with pytest.raises(ValueError, match="format"):
        vf.PoleResidue.from_dict({**model().to_dict(), "version": 2})


@pytest.mark.parametrize(
    "pole", [-1e-9, -3.0, -20 + 30j, -1e7], ids=["slow", "real", "pair", "stiff"]
)
def test_recursive_convolution_is_exact_for_linear_input(pole):
    dt = 1e-3
    a, b0, b1 = vf.recursive_convolution(vf.PoleResidue([pole], [1.0]), dt)
    # Augmented state [x, u, du/dt] with u linear over the step.
    step = la.expm(dt * np.array([[pole, 1, 0], [0, 0, 1], [0, 0, 0]], complex))
    assert_allclose(a[0], step[0, 0], rtol=1e-12, atol=1e-300)
    assert_allclose(b0[0], step[0, 1] - step[0, 2] / dt, rtol=1e-10, atol=1e-18)
    assert_allclose(b1[0], step[0, 2] / dt, rtol=1e-10, atol=1e-18)


def test_recursive_convolution_rejects_invalid_step(model):
    with pytest.raises(ValueError, match="dt"):
        vf.recursive_convolution(model(), 0.0)
