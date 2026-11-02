"""Integración: problema de Cauchy u' = f(u, t) con cualquier esquema de un paso, en una malla de tiempos o con paso fijo"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from numerical_engine.numerical_engine import Array, Field, Step


def _check_step(dt: float, n: int) -> None:
    """Valida los datos de la integración: ValueError si dt no es positivo y finito, o si n no es un entero >= 0."""
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"Delta_t debe ser positivo y finito (recibido {dt!r})")
    if int(n) != n or n < 0:
        raise ValueError(f"El número de pasos debe ser un entero igual o mayor que 0 (recibido {n!r})")


def _check_horizon(T: float, N: int) -> None:
    """Valida un horizonte de N pasos iguales: ValueError si T no es positivo y finito, o si N no es un entero >= 1."""
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"T debe ser positivo y finito (recibido {T!r})")
    if int(N) != N or N < 1:
        raise ValueError(f"N debe ser un entero ≥ 1 (recibido {N!r})")


def n_steps(tf: float, dt: float) -> int:
    """Número de pasos de tamaño dt (al menos 1) que cubren el intervalo [0, tf]: round(tf/dt)."""
    _check_step(dt, 1)                          # Solo valida dt (n = 1 es un valor de relleno siempre válido).
    return max(1, int(round(tf / dt)))


def cauchy_problem(f: Field, u0: Array, t: Array, scheme: Step,
                   stop: Callable[[Array], bool] | None = None) -> Array:
    """Problema de Cauchy u' = f(u, t), u(t[0]) = u0: aplica el esquema paso a paso en los instantes t (malla
    cualquiera, no necesariamente uniforme). Devuelve un array (len(t), dim), una fila por instante. Si stop(u) es
    True tras un paso, rellena el resto con NaN y para. Los errores de Newton se propagan a quien llama."""
    u0 = np.asarray(u0, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(t) - 1 # Número de pasos
    u = np.zeros((n + 1, u0.size))
    u[0] = u0
    for i in range(n):
        u[i + 1] = scheme(f, u[i], t[i + 1] - t[i], t[i])
        if stop is not None and stop(u[i + 1]): # “si hay criterio de parada, y dice que el estado ha divergido…”.
            u[i + 1:] = np.nan
            break
    return u


def integrate(scheme: Step, f: Field, u0: Array, dt: float, n: int, t0: float = 0.0,
              stop: Callable[[Array], bool] | None = None) -> Array:
    """Problema de Cauchy con paso fijo: n pasos de tamaño dt desde u0 en t0. Devuelve un array (n+1, dim).
    Si stop(u) es True tras un paso, rellena el resto con NaN y para."""
    _check_step(dt, n)
    t = np.asarray(t0 + dt * np.arange(n + 1), dtype=float) # Malla de tiempos uniformemente espaciada.
    with np.errstate(all="ignore"): # Sin warnings, todo se gestiona mediante el criterio de parada.
        return cauchy_problem(f, u0, t, scheme, stop=stop)


def final_state(scheme: Step, f: Field, u0: Array, T: float, N: int, t0: float = 0.0) -> Array:
    """Estado en t0 + T tras N pasos iguales. ValueError si N < 1 o T no es positivo."""
    _check_horizon(T, N)
    return cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)[-1, :]
