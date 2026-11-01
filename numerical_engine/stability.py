"""Regiones de estabilidad absoluta: |R(z)| en una malla del plano complejo"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from numerical_engine.numerical_engine import Array

Range = tuple[float, float]                      # intervalo (mín, máx)


def stability_grid(amplification: Callable[[Array], Array], re: Range = (-4.0, 2.0),
                   im: Range = (-3.5, 3.5), n: int = 301) -> tuple[Array, Array, Array]:
    """Devuelve (x, y, A) con A[i, j] = |R(x[i] + 1j*y[j])| en malla n x n; inf donde R no está definida"""
    x, y = np.linspace(*re, n), np.linspace(*im, n)
    with np.errstate(all="ignore"):
        A = np.abs(amplification(x[:, None] + 1j * y[None, :]))
    return x, y, np.nan_to_num(A, nan=np.inf)
