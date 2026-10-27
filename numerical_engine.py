"""Esquemas de un paso, Richardson y regiones de estabilidad
Convenio: f(u, t) es el campo y scheme(f, u, delta, t) avanza un paso de tamaño delta"""

from __future__ import annotations

from typing import Callable, NamedTuple, Union

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

Array = npt.NDArray[np.float64]
Field = Callable[[Array, float], Array]
Step = Callable[..., Array]
Amplification = Callable[[complex], complex]
Component = Union[slice, list[int]]               # columnas del estado que se miran
Range = tuple[float, float]


# ============================================================== 1. esquemas de un paso

def euler(f: Field, u: Array, delta: float, t: float) -> Array:
    """Euler explícito, orden 1"""
    return u + delta * f(u, t)


def my_jacobian(F: Callable[[Array], Array], Y: Array, eps: float = 1e-7, F0: Array | None = None) -> Array:
    """Jacobiana de F en Y por diferencias hacia delante; F0 = F(Y) opcional"""
    if F0 is None:
        F0 = F(Y)
    J = np.empty((Y.size, Y.size))
    for j in range(Y.size):
        h = eps * max(1.0, abs(Y[j]))
        Yh = Y.copy()
        Yh[j] += h
        J[:, j] = (F(Yh) - F0) / h
    return J


def my_newton(F: Callable[[Array], Array], Y: Array, tol: float, max_iter: int,
              dF: Callable[[Array], Array] | None = None) -> Array:
    """Newton para F(Y) = 0 (dF exacta o por diferencias); ArithmeticError si diverge, RuntimeError si no converge"""
    for _ in range(max_iter):
        F0 = F(Y)
        J = my_jacobian(F, Y, F0=F0) if dF is None else dF(Y)
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
              limit: float | None = None, components: Component = slice(0, 3)) -> Array:
    """n pasos de tamaño dt desde u0 en t0; con limit pone NaN al divergir"""
    _check_step(dt, n)

    def blown(u: Array) -> bool:
        # diverge si la suma de |u[components]| no es finita o llega a limit
        with np.errstate(all="ignore"):
            return not (np.abs(u[components]).sum() < limit)

    t = np.asarray(t0 + dt * np.arange(n + 1), dtype=float)
    with np.errstate(all="ignore"):
        return cauchy_problem(f, u0, t, scheme, stop=None if limit is None else blown)


def first_divergence(Y: Array, bound: float, components: Component = slice(0, 3)) -> int | None:
    """Índice de la primera fila de Y no finita o con norma de Y[components] > bound; None si no hay"""
    with np.errstate(all="ignore"):
        bad = ~np.isfinite(Y).all(axis=1) | (np.linalg.norm(Y[:, components], axis=1) > bound)
    return int(np.argmax(bad)) if bad.any() else None


def final_state(scheme: Step, f: Field, u0: Array, T: float, N: int, t0: float = 0.0) -> Array:
    """Estado en t0 + T tras N pasos iguales; ValueError si N < 1 o T no es positivo"""
    _check_richardson(T, N, 1.0)
    return cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)[-1, :]


# ============================================================== 4. orden y error: Richardson

def _check_richardson(T: float, N: int, p: float) -> None:
    """ValueError si T no es positivo y finito, N no es entero >= 1 o p no es positivo"""
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"T debe ser positivo y finito (recibido {T!r})")
    if int(N) != N or N < 1:
        raise ValueError(f"N debe ser un entero ≥ 1 (recibido {N!r})")
    if not (np.isfinite(p) and p > 0):
        raise ValueError(f"el orden p debe ser positivo (recibido {p!r})")


def _rounding_floor(n_steps: int, u: Array) -> float:
    """Umbral de ruido de redondeo para diferencias entre mallas tras n_steps pasos"""
    return float(50 * np.finfo(float).eps * np.sqrt(n_steps) * max(1.0, float(norm(u))))


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


# ============================================================== 5. regiones de estabilidad absoluta

def stability_grid(amplification: Callable[[Array], Array], re: Range = (-4.0, 2.0),
                   im: Range = (-3.5, 3.5), n: int = 301) -> tuple[Array, Array, Array]:
    """Devuelve (x, y, A) con A[i, j] = |R(x[i] + 1j*y[j])| en malla n x n; inf donde R no está definida"""
    x, y = np.linspace(*re, n), np.linspace(*im, n)
    with np.errstate(all="ignore"):
        A = np.abs(amplification(x[:, None] + 1j * y[None, :]))
    return x, y, np.nan_to_num(A, nan=np.inf)

