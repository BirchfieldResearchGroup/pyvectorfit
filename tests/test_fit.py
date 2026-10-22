import numpy as np
import pytest
from numpy.testing import assert_allclose

import pyvectorfit as vf

SHAPES = [(), (3,), (2, 2)]


def sorted_poles(m):
    return m.poles[np.lexsort((m.poles.real, m.poles.imag))]


@pytest.mark.parametrize("shape", SHAPES)
def test_recovers_rational_model(model, f, shape):
    m = model(shape)
    fitted, report = vf.fit(f, m.response(f), poles=m.order)
    assert report.status == "converged"
    assert report.max_error < 1e-10
    assert_allclose(sorted_poles(fitted), sorted_poles(m), rtol=1e-8)


def test_starting_poles_and_history(model, f):
    m = model(real=1, pairs=3)
    fitted, report = vf.fit(f, m.response(f), poles=[-5.0, -1 + 50j, -5 + 600j, -20 + 7e3j])
    assert report.history[0] > 1e-3
    assert min(report.history) < 1e-12
    assert fitted.order == m.order


@pytest.mark.parametrize("asymptote", ["none", "constant", "linear"])
def test_asymptotes(model, f, asymptote):
    m = model((2,), d=asymptote != "none", e=asymptote == "linear")
    fitted, report = vf.fit(f, m.response(f), poles=m.order, asymptote=asymptote)
    assert report.max_error < 1e-10
    assert_allclose(fitted.d, m.d, atol=1e-9)
    assert_allclose(fitted.e, m.e, atol=1e-12)


def test_symmetric_fit_packs_and_reports_asymmetry(model, f):
    m = model((3, 3), symmetric=True)
    values = m.response(f)
    values[:, 0, 1] *= 1.001
    fitted, report = vf.fit(f, values, poles=m.order, symmetric=True)
    assert report.asymmetry > 0
    assert_allclose(fitted.residues, fitted.residues.swapaxes(1, 2))
    reference, _ = vf.fit(f, (values + values.swapaxes(1, 2)) / 2, poles=fitted.poles, iterations=0)
    assert_allclose(fitted.response(f), reference.response(f), rtol=1e-10, atol=1e-12)


def test_dc_is_matched_exactly(model, f, rng):
    m = model((2, 2))
    noisy = m.response(f) * (1 + 1e-3 * rng.normal(size=(len(f), 2, 2)))
    fitted, _ = vf.fit(f, noisy, poles=m.order, dc=m(0.0).real)
    assert_allclose(fitted(0.0).real, m(0.0).real, rtol=1e-12, atol=1e-12)


def test_noisy_fit_returns_best_iterate(model, f, rng):
    m = model((2,))
    noise = 1e-3 * (rng.normal(size=(len(f), 2)) + 1j * rng.normal(size=(len(f), 2)))
    fitted, report = vf.fit(f, m.response(f) + noise, poles=m.order + 4, iterations=15)
    assert report.status in ("converged", "max_iterations")
    weighted = vf.fit(f, m.response(f) + noise, poles=fitted.poles, iterations=0)[1].history[0]
    assert weighted == pytest.approx(min(report.history), rel=1e-6)
    assert report.max_error < 2e-2


def test_delay_is_removed_and_stored(model, f):
    m = model((2,), delays=(2.5e-4,))
    fitted, report = vf.fit(f, m.response(f), poles=m.order, delay=2.5e-4)
    assert report.max_error < 1e-10
    assert_allclose(fitted.delays, [2.5e-4])


def test_joint_residues_for_delay_groups(model, f):
    m = model((2, 2), real=2, pairs=2, delays=(1e-4, 3e-4))
    fitted, report = vf.fit_residues(
        f, m.response(f), poles=m.poles, delays=m.delays, groups=m.groups
    )
    assert report.status == "fixed"
    assert_allclose(fitted.residues, m.residues, rtol=1e-8)
    assert_allclose(fitted.d, m.d, rtol=1e-8, atol=1e-10)


def test_duplicate_poles_give_minimum_norm_residues(f):
    m = vf.PoleResidue([-10.0], [5.0], d=1.0)
    fitted, report = vf.fit_residues(f, m.response(f), poles=[-10.0, -10.0])
    assert report.max_error < 1e-12
    assert_allclose(fitted.residues, [2.5, 2.5])


@pytest.mark.parametrize("weights", [None, "relative", "relative_norm", "frequency", "entries"])
def test_weight_forms(model, f, weights):
    m = model((2, 2))
    values = m.response(f)
    if weights == "frequency":
        weights = np.geomspace(1.0, 1e-2, len(f))
    elif weights == "entries":
        weights = np.abs(values) + 1.0
    _, report = vf.fit(f, values, poles=m.order, weights=weights)
    assert report.max_error < 1e-9


def test_scaling_frequency_and_values(model, f):
    m = model((2,), e=True)
    a, b = 1e3, 1e-4
    reference, _ = vf.fit(f, m.response(f), poles=m.order, asymptote="linear")
    scaled, _ = vf.fit(f * a, b * m.response(f), poles=m.order, asymptote="linear")
    assert_allclose(scaled.poles, a * reference.poles, rtol=1e-8)
    assert_allclose(scaled.response(f * a), b * reference.response(f), rtol=1e-9, atol=1e-14)


def test_zero_and_constant_data(f):
    zero, report = vf.fit(f, np.zeros(len(f)), poles=4)
    assert report.status == "fixed"
    assert_allclose(zero.response(f), 0)
    constant, _ = vf.fit(f, np.full(len(f), 3.0), poles=0)
    assert_allclose(constant.response(f), 3.0)


def test_relative_error_is_floored():
    reference = np.array([1.0, 1e-6, 0.0])
    error = vf.relative_error(reference + 1e-6, reference, floor=1e-3)
    assert_allclose(error, [1e-6, 1e-3, 1e-3])


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"f": [1.0, 1.0, 2.0]}, "increasing"),
        ({"f": [-1.0, 1.0, 2.0]}, "increasing"),
        ({"values": np.ones(4)}, "shape"),
        ({"weights": -np.ones(3)}, "nonnegative"),
        ({"weights": "inverse"}, "weights"),
        ({"asymptote": "quadratic"}, "asymptote"),
        ({"poles": -1}, "order"),
        ({"delay": -1.0}, "delay"),
        ({"symmetric": True}, "square"),
        ({"dc": [1.0, 2.0]}, "dc"),
    ],
    ids=lambda v: str(v)[:24],
)
def test_rejects_invalid_input(change, message):
    arguments = {"f": [1.0, 2.0, 4.0], "values": np.ones(3), "poles": 1, **change}
    with pytest.raises(ValueError, match=message):
        vf.fit(arguments.pop("f"), arguments.pop("values"), **arguments)


def test_rejects_underdetermined_fit():
    with pytest.raises(ValueError, match="too few samples"):
        vf.fit([1.0, 2.0], np.ones(2), poles=6)
