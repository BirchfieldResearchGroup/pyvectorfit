import json

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy import linalg as la

import pyvectorfit as vf

LENGTH = 50e3


@pytest.fixture
def band():
    return np.geomspace(1.0, 1e6, 141)


@pytest.fixture
def validation():
    return np.geomspace(1.1, 9e5, 997)


def test_line_functions_satisfy_the_telegrapher_identities(rlgc, band):
    z, y = rlgc(2j * np.pi * band)
    yc, h, gamma = vf.line_functions(z, y, LENGTH)
    assert_allclose(yc @ z @ yc, y, rtol=1e-9, atol=1e-18)
    assert_allclose(yc, yc.swapaxes(1, 2), rtol=1e-10, atol=1e-16)
    for k in (0, 70, 140):
        voltage_wave = la.expm(-LENGTH * la.sqrtm(z[k] @ y[k]))
        assert_allclose(h[k] @ yc[k], yc[k] @ voltage_wave, rtol=1e-8, atol=1e-14)
        assert_allclose(
            np.sort_complex(np.exp(-LENGTH * gamma[k])), np.sort_complex(la.eigvals(h[k]))
        )
    assert np.all(gamma.real > 0)
    assert np.all(gamma.imag > 0)
    steps = np.abs(np.diff(gamma, axis=0)) / np.abs(gamma[1:])
    assert steps.max() < 0.2


def test_lossless_branch_propagates_forward(band):
    s = 2j * np.pi * band[:, None, None]
    yc, h, gamma = vf.line_functions(s * 1e-6 * np.eye(2), s * 1e-11 * np.eye(2), LENGTH)
    assert_allclose(gamma.real, 0, atol=1e-12)
    assert_allclose(
        gamma.imag, np.broadcast_to(2 * np.pi * band[:, None] * 1e-17**0.5, gamma.shape)
    )
    assert_allclose(yc, np.broadcast_to(np.sqrt(1e-11 / 1e-6) * np.eye(2), yc.shape), atol=1e-15)
    assert_allclose(np.abs(np.linalg.eigvals(h)), 1, rtol=1e-12)


def test_damped_samples_follow_the_analytic_line(band):
    s = 50.0 + 2j * np.pi * band
    yc, h, _ = vf.line_functions(s[:, None, None] * 1e-6, s[:, None, None] * 1e-11, LENGTH)
    tau = LENGTH * np.sqrt(1e-17)
    assert_allclose(h[:, 0, 0], np.exp(-s * tau), rtol=1e-10)
    assert_allclose(yc[:, 0, 0], np.sqrt(1e-11 / 1e-6), rtol=1e-12)


def test_terminal_admittance_matches_coth_csch():
    yc, gamma_l = 0.02 + 0.001j, 0.3 + 2.0j
    expected = yc * np.array(
        [
            [1 / np.tanh(gamma_l), -1 / np.sinh(gamma_l)],
            [-1 / np.sinh(gamma_l), 1 / np.tanh(gamma_l)],
        ]
    )
    value = vf.terminal_admittance([[yc]], [[np.exp(-gamma_l)]])
    assert_allclose(value, expected, rtol=1e-13)


def test_lossless_line_is_fitted_exactly(band):
    s = 2j * np.pi * band[:, None, None]
    line, report = vf.fit_line(
        band,
        s * 1e-6,
        s * 1e-11,
        LENGTH,
        yc_poles=0,
        h_poles=0,
        delays="phase",
        h_asymptote="constant",
    )
    tau = LENGTH * np.sqrt(1e-17)
    assert_allclose(line.h.delays, [tau], rtol=1e-12)
    assert_allclose(line.yc.d, [[[np.sqrt(1e-11 / 1e-6)]]], rtol=1e-12)
    assert_allclose(line.h.response(band)[:, 0, 0], np.exp(-2j * np.pi * band * tau), atol=1e-11)
    assert report.groups == ((0,),)


def test_fit_line_is_accurate_and_passive(rlgc, band, validation):
    line, report = vf.fit_line(band, *rlgc(2j * np.pi * band), LENGTH, yc_poles=10, h_poles=10)
    exact = vf.line_functions(*rlgc(2j * np.pi * validation), LENGTH)
    terminal = vf.terminal_admittance(line.yc.response(validation), line.h.response(validation))
    assert vf.relative_error(line.yc.response(validation), exact.yc).max() < 1e-3
    assert vf.relative_error(terminal, vf.terminal_admittance(exact.yc, exact.h)).max() < 1e-2
    assert np.all(line.h.d == 0)
    assert sorted(sum(report.groups, ())) == [0, 1, 2]
    assert np.all(np.diff(line.h.delays) > 0)
    assert vf.check_passivity(line.yc).passive


def test_identical_modes_share_one_delay_group(band):
    s = 2j * np.pi * band[:, None, None]
    z, y = (2e-5 + s * 1e-6) * np.eye(3), (1e-11 + s * 1e-11) * np.eye(3)
    line, report = vf.fit_line(band, z, y, LENGTH, yc_poles=8, h_poles=8)
    assert report.groups == ((0, 1, 2),)
    assert len(line.h.delays) == 1


def test_given_delays_are_used(rlgc, band):
    z, y = rlgc(2j * np.pi * band)
    delays = np.array([1.70e-4, 1.68e-4, 1.80e-4])
    line, report = vf.fit_line(
        band, z, y, LENGTH, yc_poles=6, h_poles=6, delays=delays, group_tol=0.0
    )
    assert report.groups == ((1,), (0,), (2,))
    assert_allclose(line.h.delays, np.sort(delays))


def test_line_model_round_trip(rlgc, band):
    line, _ = vf.fit_line(
        band, *rlgc(2j * np.pi * band), LENGTH, yc_poles=6, h_poles=6, delays="phase"
    )
    copy = vf.LineModel.from_dict(json.loads(json.dumps(line.to_dict())))
    assert_allclose(copy.h.response(band), line.h.response(band), rtol=1e-15)
    assert_allclose(copy.yc.response(band), line.yc.response(band), rtol=1e-15)


@pytest.mark.parametrize(
    ("yc", "h", "message"),
    [
        (
            vf.PoleResidue([], np.zeros((0, 2)), d=1.0),
            vf.PoleResidue([], np.zeros((0, 2))),
            "square",
        ),
        (
            vf.PoleResidue([], np.zeros((0, 1, 1)), d=1.0, delays=[1e-3]),
            vf.PoleResidue([], np.zeros((0, 1, 1))),
            "undelayed",
        ),
        (
            vf.PoleResidue([], np.zeros((0, 1, 1)), d=1.0),
            vf.PoleResidue([], np.zeros((0, 1, 1)), e=1.0),
            "proper",
        ),
    ],
    ids=["vector", "delayed-yc", "improper-h"],
)
def test_line_model_rejects_inconsistent_parts(yc, h, message):
    with pytest.raises(ValueError, match=message):
        vf.LineModel(yc, h)


def test_fit_line_rejects_invalid_delays(rlgc, band):
    z, y = rlgc(2j * np.pi * band)
    with pytest.raises(ValueError, match="delays"):
        vf.fit_line(band, z, y, LENGTH, delays="optimal")
    with pytest.raises(ValueError, match="delays"):
        vf.fit_line(band, z, y, LENGTH, delays=[1e-4])
