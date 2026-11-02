"""Núcleo del motor: Newton, esquemas de un paso, amplificación, regiones de estabilidad, integración y ruido de redondeo. 
Convenio: f(u, t) es el campo y scheme(f, u, dt, t) avanza un paso de tamaño dt"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import NamedTuple, Union

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

# ================ [3] AMPLIFICACIÓN, ESTABILIDAD Y REGISTRO DE ESQUEMAS ================
# Paradigma declarativo: R(z) no se escribe a mano, cada esquema la descubre dando un paso del problema test u' = z*u.
# En ese problema, un paso de tamaño dt multiplica u por R(z*dt). Partiendo de u = 1 con dt = 1, el paso
# devuelve directamente R(z). Así R sale del mismo código que integra, y un esquema nuevo no necesita fórmula.
# Región de estabilidad absoluta: los z con |R(z)| <= 1 (la raíz r = R(z) de pi(r, z) = 0 dentro del círculo
# unidad). Ahí un modo de autovalor lambda no crece por efecto del esquema con el que se integra.

def amplification_factor(scheme: Step, z: ArrayLike, implicit: bool = False, chunk: int = 32) -> ComplexArray:
    """R(z) de un esquema, obtenida de su propio paso. z puede ser un número, un vector o una malla 2D, y R sale con 
    su misma forma. Los explícitos procesan todos los puntos a la vez; los implícitos usan Newton por bloques de 
    chunk puntos (inf en el polo de R)."""
    z = np.asarray(z, dtype=complex)
    if not implicit:                           # Explícito: solo operaciones punto a punto, toda la malla de golpe
        return np.asarray(scheme(lambda U, t: z * U, np.ones_like(z), 1.0, 0.0), dtype=complex)
    flat = z.ravel()                           # Implícito: la malla aplanada, troceada en bloques
    blocks = [_implicit_block(scheme, flat[i:i + chunk]) for i in range(0, flat.size, chunk)]
    return np.concatenate(blocks).reshape(z.shape)


def _implicit_block(scheme: Step, z: ComplexArray) -> ComplexArray:
    """R en los puntos z (vector) de un esquema implícito, con un solo paso del problema test; inf en el polo de R.
    Los puntos son independientes, así que la jacobiana de u' = z*u es diagonal, diag(z), y se pasa exacta: la
    numérica (my_jacobian) trabaja con reales y perdería la parte imaginaria de z."""
    try:
        return np.asarray(scheme(lambda U, t: z * U, np.ones_like(z), 1.0, 0.0, jac=lambda U, t: np.diag(z)),
                          dtype=complex)
    except (np.linalg.LinAlgError, ArithmeticError, RuntimeError):   # algún punto es el polo: Newton no puede
        if z.size == 1:
            return np.array([np.inf])
        return np.concatenate([_implicit_block(scheme, z[j:j + 1]) for j in range(z.size)])   # punto a punto


def stability_grid(amplification: Amplification, re: Range = (-4.0, 2.0), im: Range = (-3.5, 3.5),
                   n: int = 301) -> tuple[Array, Array, Array]:
    """Evalúa |R(z)| en una malla n x n del plano z: devuelve (x, y, A) con A[i, j] = |R(x[i] + i*y[j])|.
    La frontera de la región de estabilidad es la curva de nivel A = 1. Donde R no está definida (NaN) se pone inf."""
    x, y = np.linspace(*re, n), np.linspace(*im, n)               # * para desempaquetar la tupla, linspace(i, f, n).
    with np.errstate(all="ignore"):                               # Cerca del polo R se dispara: no es un error.
        A = np.abs(amplification(x[:, None] + 1j * y[None, :]))   # Malla por broadcasting: x en filas, y en columnas.
    return x, y, np.nan_to_num(A, nan=np.inf)                     # Cambia los nan por inf para que cuenten como fuera
                                                                  # de la región y no estropeen la curva.


class Scheme(NamedTuple):
    """Esquema de un paso: su función de paso y sus propiedades teóricas. R(z) no se guarda: se descubre del paso."""
    step: Step                            # función de paso scheme(f, u, dt, t)
    order: int                            # orden global del error en dt
    module_order: int | None = None       # orden global del error de |R| en un centro; None si |R| = 1 exacto
    phase_order: int = 1                  # orden global del error de fase (arg R - beta*dt) en un centro
    label: str = ""                       # nombre para mostrar
    short: str = ""                       # nombre corto para tablas y leyendas
    implicit: bool = False                # usa Newton (admite jac=)

    def amplification(self, z: ArrayLike) -> ComplexArray:
        """R(z) de este esquema, sacada de su propio paso (ver amplification_factor)"""
        return amplification_factor(self.step, z, self.implicit)

#Registro de esquemas: clave → Scheme(paso, orden, orden de módulo, orden de fase, nombre, nombre corto, implícito)
SCHEMES: dict[str, Scheme] = {
    "euler": Scheme(euler,           1, 1, 2, "Euler explícito", "Euler"),
    "be":    Scheme(implicit_euler,  1, 1, 2, "Euler implícito (inverso)", "Euler imp.", True),
    "cn":    Scheme(crank_nicolson,  2, None, 2, "Crank-Nicolson (implícito)", "C-N", True),
    "rk4":   Scheme(runge_kutta_4,   4, 5, 4, "Runge-Kutta 4", "RK4"),
}

# ================ [4] INTEGRACIÓN ================

def _check_step(dt: float, n: int) -> None:
    """Valida los datos de la integración: ValueError si dt no es positivo y finito, o si n no es un entero >= 0."""
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"Delta_t debe ser positivo y finito (recibido {dt!r})")
    if int(n) != n or n < 0:
        raise ValueError(f"El número de pasos debe ser un entero igual o mayor que 0 (recibido {n!r})")


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

# ================ [5] RUIDO DE REDONDEO ================

NOISE_FACTOR = 50.0  # Margen de seguridad sobre eps*sqrt(n): cada paso son decenas de operaciones.

def noise_floor(n: int) -> float:
    """Error relativo indistinguible del redondeo acumulado tras n pasos: NOISE_FACTOR*eps*sqrt(n). Los errores de
    cada operación (hasta eps) tienen signos al azar y se acumulan como un paseo aleatorio, de ahí sqrt(n)."""
    eps = np.finfo(float).eps  # Precisión relativa de un float (2.2e-16).
    return float(NOISE_FACTOR * eps * math.sqrt(max(n, 1)))
