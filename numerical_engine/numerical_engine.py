"""Núcleo del motor: Newton, esquemas de un paso, integración y ruido de redondeo

Convenio: f(u, t) es el campo y scheme(f, u, dt, t) avanza un paso de tamaño dt"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple, Union

import numpy as np
from numpy.linalg import norm
from numpy.typing import NDArray

Array = NDArray[np.float64]
VectorMap = Callable[[Array], Array]             # función de vector a vector, F(Y)
Field = Callable[[Array, float], Array]          # campo f(u, t)
Step = Callable[..., Array]                      # esquema scheme(f, u, dt, t, ...); los implícitos admiten más
Amplification = Callable[[complex], complex]     # factor de amplificación R(z)
Component = Union[slice, list[int]]              # columnas del estado que se miran


# ============================================================== 1. esquemas de un paso

def euler(f: Field, u: Array, delta: float, t: float) -> Array:
    """Euler explícito, orden 1"""
    return u + delta * f(u, t)


def my_jacobian(F: VectorMap, Y: Array, eps: float = 6e-6) -> Array:
    """Jacobiana de F en Y por diferencias centradas, de orden 2 (error del orden de h^2)"""
    J = np.empty((Y.size, Y.size))
    for j in range(Y.size):
        h = eps * max(1.0, abs(Y[j]))      # paso relativo si |Y[j]| > 1, para que no se pierda al sumarlo
        Yp, Ym = Y.copy(), Y.copy()
        Yp[j] += h
        Ym[j] -= h
        J[:, j] = (F(Yp) - F(Ym)) / (2 * h)  # columna j: derivada de F respecto a Y[j]
    return J


def my_newton(F: Callable[[Array], Array], Y: Array, tol: float, max_iter: int,
              dF: Callable[[Array], Array] | None = None) -> Array:
    """Newton para F(Y) = 0 (dF exacta o por diferencias); ArithmeticError si diverge, RuntimeError si no converge"""
    for _ in range(max_iter):
        F0 = F(Y)
        J = my_jacobian(F, Y) if dF is None else dF(Y)
        dY = np.linalg.solve(J, -F0)
        Y = Y + dY
        if not np.all(np.isfinite(Y)):
            raise ArithmeticError("Newton diverge (NaN/inf)")
        if norm(dY) <= tol * max(1.0, norm(Y)):
            return Y
    raise RuntimeError(f"Newton sin converger en {max_iter} iteraciones")


def implicit_euler(f: Field, u: Array, delta: float, t: float, tol: float = 1e-12, max_iter: int = 20,
                   jac: Field | None = None) -> Array:
    """Euler implícito, orden 1, con Newton; jac(u, t) es df/du exacta opcional"""
    dF = None if jac is None else (lambda Y: np.eye(Y.size) - delta * jac(Y, t + delta))
    return my_newton(lambda Y: Y - u - delta * f(Y, t + delta), u + delta * f(u, t), tol, max_iter, dF)


def crank_nicolson(f: Field, u: Array, delta: float, t: float, tol: float = 1e-12, max_iter: int = 20,
                   jac: Field | None = None) -> Array:
    """Crank-Nicolson (trapecio), orden 2, con Newton; jac(u, t) es df/du exacta opcional"""
    fn = f(u, t)
    dF = None if jac is None else (lambda Y: np.eye(Y.size) - delta / 2 * jac(Y, t + delta))
    return my_newton(lambda Y: Y - u - delta * (fn + f(Y, t + delta)) / 2, u + delta * fn, tol, max_iter, dF)


def runge_kutta_4(f: Field, u: Array, delta: float, t: float) -> Array:
    """Runge-Kutta clásico, orden 4"""
    k1 = f(u, t)
    k2 = f(u + delta * k1 / 2, t + delta / 2)
    k3 = f(u + delta * k2 / 2, t + delta / 2)
    k4 = f(u + delta * k3, t + delta)
    return u + delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6


# ============================================================== 2. amplificación y registro de esquemas

def amplification_euler(z: complex) -> complex:
    """R(z) = 1 + z"""
    return 1 + z


def amplification_implicit_euler(z: complex) -> complex:
    """R(z) = 1 / (1 - z)"""
    return 1 / (1 - z)


def amplification_crank_nicolson(z: complex) -> complex:
    """R(z) = (1 + z/2) / (1 - z/2)"""
    return (1 + z / 2) / (1 - z / 2)


def amplification_runge_kutta_4(z: complex) -> complex:
    """R(z) = 1 + z + z^2/2 + z^3/6 + z^4/24"""
    return 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24


class Scheme(NamedTuple):
    """Esquema de un paso con sus propiedades teóricas"""
    # module_order y phase_order: orden del error de |R| y de arg R en z = i*beta*dt; None en module_order es |R| = 1 exacto (solo caso lineal)
    step: Step
    amplification: Amplification
    order: int                            # orden global en dt
    module_order: int | None = None
    phase_order: int = 1
    label: str = ""                       # nombre para mostrar
    short: str = ""                       # nombre corto para tablas y leyendas
    implicit: bool = False                # usa Newton (admite jac=)


SCHEMES: dict[str, Scheme] = {
    "euler": Scheme(euler,          amplification_euler,          1, 1, 2, "Euler explícito", "Euler"),
    "be":    Scheme(implicit_euler, amplification_implicit_euler, 1, 1, 2, "Euler implícito (inverso)", "Euler imp.", True),
    "cn":    Scheme(crank_nicolson, amplification_crank_nicolson, 2, None, 2, "Crank-Nicolson (implícito)", "C-N", True),
    "rk4":   Scheme(runge_kutta_4,  amplification_runge_kutta_4,  4, 5, 4, "Runge-Kutta 4", "RK4"),
}


# ============================================================== 3. integración

def _check_step(dt: float, n: int) -> None:
    """ValueError si dt no es real positivo y finito o n no es entero >= 0"""
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"Δt debe ser positivo y finito (recibido {dt!r})")
    if int(n) != n or n < 0:
        raise ValueError(f"el número de pasos debe ser un entero ≥ 0 (recibido {n!r})")


def n_steps(T: float, dt: float) -> int:
    """Pasos de tamaño dt (al menos 1) que cubren T"""
    _check_step(dt, 1)
    return max(1, int(round(T / dt)))


def cauchy_problem(f: Field, u0: Array, t: Array, scheme: Step,
                   stop: Callable[[Array], bool] | None = None) -> Array:
    """Integra u' = f(u, t) en los instantes t y devuelve array (len(t), dim); si stop(u) es True rellena el resto con NaN"""
    # los errores de Newton se propagan a quien llama
    u0 = np.asarray(u0, dtype=float)
    t = np.asarray(t, dtype=float)
    N = len(t) - 1
    u = np.zeros((N + 1, u0.size))
    u[0, :] = u0
    for n in range(N):
        u[n + 1, :] = scheme(f, u[n, :], t[n + 1] - t[n], t[n])
        if stop is not None and stop(u[n + 1, :]):
            u[n + 1:, :] = np.nan
            break
    return u


def integrate(scheme: Step, f: Field, u0: Array, dt: float, n: int, t0: float = 0.0,
              stop: Callable[[Array], bool] | None = None) -> Array:
    """n pasos de tamaño dt desde u0 en t0; si stop(u) es True rellena el resto con NaN"""
    _check_step(dt, n)
    t = np.asarray(t0 + dt * np.arange(n + 1), dtype=float)
    with np.errstate(all="ignore"):
        return cauchy_problem(f, u0, t, scheme, stop=stop)


# ============================================================== 4. ruido de redondeo

def noise_floor(n_steps: int) -> float:
    """Error de módulo indistinguible del redondeo acumulado tras n_steps pasos (crece como √n)"""
    eps = np.finfo(float).eps                          # precisión relativa de un float (2.2e-16)
    return float(50.0 * eps * math.sqrt(max(n_steps, 1)))
