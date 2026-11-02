"""Orden y error por Richardson: orden observado, error estimado y extrapolación, sobre cualquier esquema de un paso"""

from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np
from numpy.linalg import norm

from numerical_engine.integration import cauchy_problem
from numerical_engine.numerical_engine import Array, Component, Field, Step, noise_floor


def _check_richardson(T: float, N: int, p: float) -> None:
    """ValueError si T no es positivo y finito, N no es entero >= 1 o p no es positivo"""
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"T debe ser positivo y finito (recibido {T!r})")
    if int(N) != N or N < 1:
        raise ValueError(f"N debe ser un entero ≥ 1 (recibido {N!r})")
    if not (np.isfinite(p) and p > 0):
        raise ValueError(f"el orden p debe ser positivo (recibido {p!r})")


def final_state(scheme: Step, f: Field, u0: Array, T: float, N: int, t0: float = 0.0) -> Array:
    """Estado en t0 + T tras N pasos iguales; ValueError si N < 1 o T no es positivo"""
    _check_richardson(T, N, 1.0)
    return cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)[-1, :]


def _rounding_floor(n_steps: int, u: Array) -> float:
    """Umbral de ruido de redondeo para diferencias entre mallas tras n_steps pasos"""
    return noise_floor(n_steps) * max(1.0, float(norm(u)))


def scheme_order(scheme: Step, f: Field, u0: Array, T: float, N: int,
                 components: Component = slice(None)) -> float:
    """Orden observado log2(|u_N - u_2N| / |u_2N - u_4N|) sobre components; NaN si hay ruido de redondeo o diverge"""
    _check_richardson(T, N, 1.0)
    u1, u2, u3 = (final_state(scheme, f, u0, T, k * N) for k in (1, 2, 4))
    with np.errstate(all="ignore"):
        d12, d23 = norm((u1 - u2)[components]), norm((u2 - u3)[components])
        floor = _rounding_floor(4 * N, u3[components])
        return float("nan") if not np.isfinite(d12 + d23) or d23 < floor else float(np.log2(d12 / d23))


def richardson_error(scheme: Step, f: Field, u0: Array, T: float, N: int, p: float, t0: float = 0.0) -> Array:
    """Error estimado de la solución fina (2N pasos) en los N+1 instantes de la malla gruesa"""
    _check_richardson(T, N, p)
    u_coarse = cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)
    u_fine = cauchy_problem(f, u0, np.linspace(t0, t0 + T, 2 * N + 1), scheme)
    return (u_fine[0::2, :] - u_coarse) / (2 ** p - 1)


def richardson_extrapolation(scheme: Step, f: Field, u0: Array, T: float, N: int, p: float) -> Array:
    """Estado en T extrapolado, w_fino + (w_fino - w_grueso)/(2^p - 1), de orden p + 1"""
    _check_richardson(T, N, p)
    w_coarse = final_state(scheme, f, u0, T, N)
    w_fine = final_state(scheme, f, u0, T, 2 * N)
    return w_fine + (w_fine - w_coarse) / (2 ** p - 1)


class RichardsonReport(NamedTuple):
    """Resultado de richardson_analysis"""
    n: int                 # pasos de la malla gruesa
    p: float
    err: float             # norma del error estimado en T de la solución fina
    u_extr: Array
    err_real: float        # NaN sin referencia
    noise: bool = False    # True: diferencia entre mallas al nivel del redondeo, err no fiable


def richardson_analysis(scheme: Step, f: Field, u0: Array, T: float, N: int, p: float,
                        components: Component = slice(0, 3),
                        reference: Callable[[float], Array] | None = None, t0: float = 0.0) -> RichardsonReport:
    """Richardson con N y 2N pasos y orden p; reference(T) opcional da el estado exacto para medir err_real"""
    _check_richardson(T, N, p)
    with np.errstate(all="ignore"):
        runs = {k: cauchy_problem(f, u0, np.linspace(t0, t0 + T, k * N + 1), scheme) for k in (1, 2)}
        coarse, fine = runs[1][-1, :], runs[2][-1, :]
        err = norm(((runs[2][0::2, :] - runs[1]) / (2 ** p - 1))[-1][components])
        u_extr = fine + (fine - coarse) / (2 ** p - 1)
        noise = bool(norm((fine - coarse)[components]) < _rounding_floor(2 * N, fine[components]))
        err_real = float(norm(fine[components] - reference(T))) if reference is not None else float("nan")
        return RichardsonReport(N, p, float(err), u_extr, err_real, noise)
