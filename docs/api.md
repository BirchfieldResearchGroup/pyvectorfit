# API

Every name below is importable from `pyvectorfit`, which the examples import as `vf`.

## Fitting

```{eval-rst}
.. currentmodule:: pyvectorfit

.. autofunction:: fit
.. autofunction:: fit_residues
.. autofunction:: relative_error
.. autoclass:: FitReport
```

## Models

```{eval-rst}
.. currentmodule:: pyvectorfit

.. autoclass:: PoleResidue
   :members: shape, order, response, to_dict, from_dict
.. autoclass:: Result
```

## Lines

```{eval-rst}
.. currentmodule:: pyvectorfit

.. autofunction:: fit_line
.. autofunction:: line_functions
.. autofunction:: terminal_admittance
.. autoclass:: LineModel
   :members: to_dict, from_dict
.. autoclass:: LineReport
.. autoclass:: LineFunctions
```

## Passivity

```{eval-rst}
.. currentmodule:: pyvectorfit

.. autofunction:: check_passivity
.. autofunction:: enforce_passivity
.. autoclass:: PassivityReport
.. autoclass:: CorrectionReport
```

## Time domain

```{eval-rst}
.. currentmodule:: pyvectorfit

.. autofunction:: recursive_convolution
.. autoclass:: Convolution
```
