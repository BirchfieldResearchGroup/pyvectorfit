import numpy as np
import pytest
from numpy.testing import assert_allclose

import pyvectorfit as vf

DENSE = np.r_[0.0, np.geomspace(1e-3, 1e6, 4001)]


def conductance(m, f=DENSE):
    values = m.response(f).reshape(len(f), *(m.shape or (1, 1)))
    return np.linalg.eigvalsh((values + values.conj().swapaxes(1, 2)) / 2)[:, 0]


NARROW = vf.PoleResidue([-1 + 100j], [-10.0], d=1.0)


@pytest.mark.parametrize(
    "model",
    [
        vf.PoleResidue([-10.0], [20.0], d=0.2),
        vf.PoleResidue([-10.0], [20.0]),
        vf.PoleResidue([-10.0, -1 + 50j], [[[2.0, 0.2], [0.2, 3.0]], np.eye(2) * 0.1], d=np.eye(2)),
        vf.PoleResidue([], [], d=1.0, e=0.2),
    ],
    ids=["rc", "strictly-proper", "matrix", "capacitive"],
)
def test_passive_models(model):
    report = vf.check_passivity(model)
    assert report.passive
    assert report.scope == "global"
    assert conductance(model).min() >= -1e-12


@pytest.mark.parametrize(
    "model",
    [
        vf.PoleResidue([-10.0], [20.0], d=-0.1),
        vf.PoleResidue([], np.zeros((0, 2, 2)), d=[[1.0, 2.0], [2.0, 1.0]]),
        vf.PoleResidue([], [], d=1.0, e=-1.0),
        vf.PoleResidue([], np.zeros((0, 2, 2)), d=np.eye(2), e=[[1.0, 1e-15], [0.0, 1.0]]),
    ],
    ids=["negative-feedthrough", "indefinite-matrix", "negative-capacitance", "skew-e"],
)
def test_nonpassive_models(model):
    assert not vf.check_passivity(model).passive


def test_global_check_finds_violation_between_samples():
    delayed = vf.PoleResidue(NARROW.poles, NARROW.residues, NARROW.d, delays=[1e-9])
    assert vf.check_passivity(delayed, [1.0, 1e3]).passive
    report = vf.check_passivity(NARROW)
    assert not report.passive
    assert any(lo < 100 / (2 * np.pi) < hi for lo, hi in report.bands)
    # Re Y(j100) = 1 - 10 - Re(10 / (1 + 200j))
    assert report.margin == pytest.approx(-9 - 10 / 40001, rel=1e-9)
    assert report.margin < conductance(NARROW).min()


def test_sampled_scope_requires_frequencies():
    with pytest.raises(ValueError, match="required"):
        vf.check_passivity(vf.PoleResidue([-1.0], [1.0], delays=[1e-3]))


def test_line_terminal_admittance_is_sampled(rlgc):
    f = np.geomspace(1.0, 1e6, 121)
    line, _ = vf.fit_line(f, *rlgc(2j * np.pi * f), 20e3, yc_poles=8, h_poles=8)
    report = vf.check_passivity(line, f)
    assert report.scope == "sampled"


def test_enforcement_preserves_dc():
    m = vf.PoleResidue([-10.0], [20.0], d=-0.1)
    fixed, report = vf.enforce_passivity(m, np.geomspace(1e-2, 1e4, 121), preserve_dc=True)
    assert not report.before.passive
    assert report.after.passive
    assert vf.check_passivity(fixed).passive
    assert_allclose(fixed(0.0), m(0.0), rtol=1e-12)
    assert_allclose(fixed.poles, m.poles)


def test_enforcement_repairs_narrow_violation():
    fixed, report = vf.enforce_passivity(NARROW, np.geomspace(1e-2, 1e4, 121))
    assert vf.check_passivity(fixed).passive
    assert conductance(fixed).min() >= -1e-12
    assert report.change < 2.0


def test_enforcement_of_static_matrix_is_the_psd_projection():
    m = vf.PoleResidue([], np.zeros((0, 2, 2)), d=[[1.0, 2.0], [2.0, 1.0]])
    fixed, _ = vf.enforce_passivity(m, np.geomspace(1.0, 1e3, 31))
    assert_allclose(fixed.d[0], np.full((2, 2), 1.5), rtol=1e-6)


def test_enforcement_keeps_symmetric_residues():
    rotation = np.array([[1.0, 1.0], [-1.0, 1.0]]) / np.sqrt(2)
    residue = rotation @ np.diag([-20.0, 5.0]) @ rotation.T
    m = vf.PoleResidue([-10.0], residue[None], d=np.eye(2))
    fixed, _ = vf.enforce_passivity(m, np.geomspace(1e-2, 1e4, 121))
    assert vf.check_passivity(fixed).passive
    assert_allclose(fixed.residues, fixed.residues.swapaxes(1, 2), atol=1e-14)
    assert conductance(fixed).min() >= -1e-12


def test_passive_model_is_returned_unchanged():
    m = vf.PoleResidue([-10.0], [20.0], d=0.2)
    fixed, report = vf.enforce_passivity(m, np.geomspace(1e-2, 1e4, 31))
    assert report.iterations == 0
    assert report.change == 0
    assert_allclose(fixed.residues, m.residues)


def test_infeasible_correction_raises():
    m = vf.PoleResidue([-10.0], [-20.0], d=1.0)
    with pytest.raises(RuntimeError, match="infeasible"):
        vf.enforce_passivity(m, np.geomspace(1e-2, 1e4, 121), preserve_dc=True)


def test_line_enforcement_corrects_yc_and_keeps_h():
    f = np.geomspace(0.1, 1e4, 121)
    yc = vf.PoleResidue([], np.zeros((0, 1, 1)), d=[[-1e-3]])
    h = vf.PoleResidue([], np.zeros((0, 1, 1)), d=[[0.8]], delays=[1e-3])
    fixed, report = vf.enforce_passivity(vf.LineModel(yc, h), f)
    assert report.after.scope == "sampled"
    assert report.after.passive
    assert fixed.h is h
    assert vf.check_passivity(fixed.yc).passive
