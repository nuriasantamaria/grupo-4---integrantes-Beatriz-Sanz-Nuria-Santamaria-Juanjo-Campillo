"""Núcleo del motor: tipos, Newton, esquemas de un paso y ruido de redondeo.
Convenio: f(u, t) es el campo y scheme(f, u, dt, t) avanza un paso de tamaño dt"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Union

import numpy as np
from numpy.linalg import norm, solve
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]
VectorMap = Callable[[Array], Array]             # Función de vector a vector, F(Y)
Field = Callable[[Array, float], Array]          # Campo f(u, t)
Step = Callable[..., Array]                      # Esquema scheme(f, u, dt, t, ...); los implícitos admiten más
Amplification = Callable[[ArrayLike], ComplexArray]     # Factor de amplificación R(z)
Component = Union[slice, list[int]]              # Columnas del estado que se miran
Range = tuple[float, float]                      # Intervalo (mín, máx)


# ================ [1] NEWTON Y JACOBIANA NUMÉRICA ================

NEWTON_TOL = 1e-12        # Umbral por debajo de la cual Newton da por resuelta la ecuación.
NEWTON_MAX_ITER = 20      # Iteraciones máximas permitidas para el método de Newton antes de rendirse.

def my_jacobian(F: VectorMap, Y: Array, eps: float = 6e-6) -> Array:
    """Jacobiana de F en el punto Y por diferencias finitas centradas, de orden 2 (error del orden de h^2). Su coste 
    es de 2n evaluaciones de F. eps = 6e-6 es el óptimo para equilibrar el error de truncamiento y el de redondeo."""
    n = Y.size
    J = np.empty((n, n))
    for j in range(n):                       # Cada paso del bucle es una columna de J.
        h = eps * max(1.0, abs(Y[j]))        # Paso relativo si |Y[j]| > 1, para que no se pierda al sumarlo.
        Yp, Ym = Y.copy(), Y.copy() 
        Yp[j] += h                           # Punto Y con un incremento +h en la j-ésima componente.
        Ym[j] -= h                           # Punto Y con un decremento -h en la j-ésima componente.
        J[:, j] = (F(Yp) - F(Ym)) / (2 * h)  # Columna j: derivada de cada componente de F respecto a Y_j.
    return J


def my_newton(F: VectorMap, Y: Array, tol: float, max_iter: int, dF: VectorMap | None = None) -> Array:
    """Newton para F(Y) = 0 desde la estimación Y. dF es la jacobiana exacta (si es None, por diferencias finitas).
    Generalización del Newton escalar: dx = -f(x)/f'(x) se convierte en dY = -J^(-1) F(Y) ; (F(Y+dY) ≈ F(Y) + J dY)
    ArithmeticError si diverge, RuntimeError si no converge en max_iter, LinAlgError si la jacobiana es singular."""
    for _ in range(max_iter):
        J = my_jacobian(F, Y) if dF is None else dF(Y)
        dY = solve(J, -F(Y)) # Solve(A, b) resuelve A*x = b por eliminación gaussiana, sin calcular la inversa de J.
        Y = Y + dY
        if not np.all(np.isfinite(Y)):
            raise ArithmeticError("Newton diverge (NaN/inf)")
        if norm(dY) <= tol * max(1.0, norm(Y)):
            return Y
    raise RuntimeError(f"Newton sin converger en {max_iter} iteraciones")

# ================ [2] ESQUEMAS UNIPASO ================

def euler(f: Field, u: Array, dt: float, t: float) -> Array:
    """Euler explícito, orden 1: u^{n+1} = u^n + dt*f(u^n, t_n)"""
    return u + dt * f(u, t)


def implicit_euler(f: Field, u: Array, dt: float, t: float, tol: float = NEWTON_TOL,
                    max_iter: int = NEWTON_MAX_ITER, jac: Field | None = None) -> Array:
    """Euler implícito, orden 1: u^{n+1} = u^n + dt*f(u^{n+1}, t_{n+1}), resuelto con Newton desde el paso explícito.
    jac(u, t) es df/du exacta (opcional); sin ella, Newton usa la jacobiana numérica."""
    def F(Y: Array) -> Array: # Ecuación implícita del paso: F(u^{n+1}) = 0, con u^{n+1} = Y.
        return Y - u - dt * f(Y, t + dt)
    dF = None if jac is None else (lambda Y: np.eye(Y.size) - dt * jac(Y, t + dt))
    return my_newton(F, u + dt * f(u, t), tol, max_iter, dF) 


def crank_nicolson(f: Field, u: Array, dt: float, t: float, tol: float = NEWTON_TOL, max_iter: int = NEWTON_MAX_ITER,
                   jac: Field | None = None) -> Array:
    """Crank-Nicolson, orden 2: u^{n+1} = u^n + dt/2*(f(u^n, t_n) + f(u^{n+1}, t_{n+1})), resuelto con
    Newton desde el paso explícito. jac(u, t) es df/du exacta (opcional); sin ella, Newton usa la jacobiana numérica."""
    fn = f(u, t) # Tasa de cambio en el paso actual, que no depende de Y.
    def F(Y: Array) -> Array: # Ecuación implícita del paso: F(u^{n+1}) = 0, con u^{n+1} = Y.
        return Y - u - dt * (fn + f(Y, t + dt)) / 2
    dF = None if jac is None else (lambda Y: np.eye(Y.size) - dt / 2 * jac(Y, t + dt))
    return my_newton(F, u + dt * fn, tol, max_iter, dF)


def runge_kutta_4(f: Field, u: Array, dt: float, t: float) -> Array:
    """Runge-Kutta clásico, orden 4: avanza con la media (k1 + 2*k2 + 2*k3 + k4)/6 de las velocidades al inicio (k1),
    en el punto medio (k2, k3) y al final (k4) del paso."""
    k1 = f(u, t)
    k2 = f(u + dt * k1 / 2, t + dt / 2)
    k3 = f(u + dt * k2 / 2, t + dt / 2)
    k4 = f(u + dt * k3, t + dt)
    return u + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

# ================ [3] RUIDO DE REDONDEO ================

NOISE_FACTOR = 50.0  # Margen de seguridad sobre eps*sqrt(n): cada paso son decenas de operaciones.

def noise_floor(n: int) -> float:
    """Error relativo indistinguible del redondeo acumulado tras n pasos: NOISE_FACTOR*eps*sqrt(n). Los errores de
    cada operación (hasta eps) tienen signos al azar y se acumulan como un paseo aleatorio, de ahí sqrt(n)."""
    eps = np.finfo(float).eps  # Precisión relativa de un float (2.2e-16).
    return float(NOISE_FACTOR * eps * math.sqrt(max(n, 1)))
