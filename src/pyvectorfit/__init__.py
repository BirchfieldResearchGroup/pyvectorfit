"""Vector fitting of delayed pole-residue models for electromagnetic transient simulation."""

from importlib.metadata import version

from ._fit import FitReport, fit, fit_residues, relative_error
from ._line import (
    LineFunctions,
    LineModel,
    LineReport,
    fit_line,
    line_functions,
    terminal_admittance,
)
from ._model import Convolution, PoleResidue, Result, recursive_convolution
from ._passivity import CorrectionReport, PassivityReport, check_passivity, enforce_passivity

__version__ = version(__name__)

__all__ = [
    "Convolution",
    "CorrectionReport",
    "FitReport",
    "LineFunctions",
    "LineModel",
    "LineReport",
    "PassivityReport",
    "PoleResidue",
    "Result",
    "check_passivity",
    "enforce_passivity",
    "fit",
    "fit_line",
    "fit_residues",
    "line_functions",
    "recursive_convolution",
    "relative_error",
    "terminal_admittance",
]
