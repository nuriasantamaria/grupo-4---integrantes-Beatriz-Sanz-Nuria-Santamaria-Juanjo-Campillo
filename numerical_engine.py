"""Integración temporal con esquemas de paso fijo (Euler, Euler implícito, Crank-Nicolson, RK4),
factores de amplificación R(z) de cada esquema (para estudiar su estabilidad y su error en
el centro lineal) y estimación del error asintótico por extrapolación de Richardson.

Convenio: f(u, t) es el campo y scheme(f, u, delta, t) avanza un paso de tamaño delta.
Es genérico: no sabe nada del problema que integra (el campo lo pone quien llama)."""

from __future__ import annotations

from typing import Callable, List, NamedTuple, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.linalg import norm

Array = npt.NDArray[np.float64]
Field = Callable[[Array, float], Array]               # f(u, t)
Step = Callable[..., Array]                           # scheme(f, u, delta, t, **opciones) -> u siguiente
Amplification = Callable[[complex], complex]          # R(z)
Component = Union[slice, List[int]]                  # qué columnas del estado se miran


# ============================================================== esquemas de un paso
def euler(f: Field, u: Array, delta: float, t: float) -> Array:
    """Euler explícito, orden 1."""
    return u + delta * f(u, t)


def my_jacobian(F: Callable[[Array], Array], Y: Array, eps: float = 1e-7, F0: Array | None = None) -> Array:
    """Jacobiana de F en Y por diferencias finitas hacia delante (F0 = F(Y) si ya se conoce)."""
    F0, J = (F(Y) if F0 is None else F0), np.empty((Y.size, Y.size))
    for j in range(Y.size):
        h = eps * max(1.0, abs(Y[j]))
        Yh = Y.copy()
        Yh[j] += h
        J[:, j] = (F(Yh) - F0) / h
    return J


def my_newton(F: Callable[[Array], Array], Y: Array, tol: float, max_iter: int,
              dF: Callable[[Array], Array] | None = None) -> Array:
    """Newton para F(Y) = 0 con la Jacobiana exacta `dF(Y)` si se da y, si no, por diferencias finitas.
    Lanza ArithmeticError si diverge y RuntimeError si no converge en max_iter."""
    for _ in range(max_iter):
        F0 = F(Y)
        dY = np.linalg.solve(my_jacobian(F, Y, F0=F0) if dF is None else dF(Y), -F0)
        Y = Y + dY
        if not np.all(np.isfinite(Y)):
            raise ArithmeticError("Newton diverge (NaN/inf)")
        if norm(dY) <= tol * max(1.0, norm(Y)):
            return Y
    raise RuntimeError(f"Newton sin converger en {max_iter} iteraciones")


def implicit_euler(f: Field, u: Array, delta: float, t: float, tol: float = 1e-12, max_iter: int = 20,
                   jac: Field | None = None) -> Array:
    """Euler implícito (inverso), orden 1: resuelve Y = u + Δ f(Y, t+Δ) con Newton.
    `jac(u, t)` = ∂f/∂u exacta (opcional); la Jacobiana del residuo es entonces I − Δ·jac(Y, t+Δ)."""
    dF = None if jac is None else (lambda Y: np.eye(Y.size) - delta * jac(Y, t + delta))
    return my_newton(lambda Y: Y - u - delta * f(Y, t + delta), u + delta * f(u, t), tol, max_iter, dF)


def crank_nicolson(f: Field, u: Array, delta: float, t: float, tol: float = 1e-12, max_iter: int = 20,
                   jac: Field | None = None) -> Array:
    """Crank-Nicolson (regla del trapecio), orden 2: Y = u + Δ/2 (f(u, t) + f(Y, t+Δ)) con Newton.
    `jac(u, t)` = ∂f/∂u exacta (opcional); la Jacobiana del residuo es entonces I − Δ/2·jac(Y, t+Δ)."""
    fn = f(u, t)
    dF = None if jac is None else (lambda Y: np.eye(Y.size) - delta / 2 * jac(Y, t + delta))
    return my_newton(lambda Y: Y - u - delta * (fn + f(Y, t + delta)) / 2, u + delta * fn, tol, max_iter, dF)


def runge_kutta_4(f: Field, u: Array, delta: float, t: float) -> Array:
    """Runge-Kutta clásico de 4 etapas, orden 4."""
    k1 = f(u, t)
    k2 = f(u + delta * k1 / 2, t + delta / 2)
    k3 = f(u + delta * k2 / 2, t + delta / 2)
    k4 = f(u + delta * k3, t + delta)
    return u + delta * (k1 + 2 * k2 + 2 * k3 + k4) / 6


# ============================================================== factores de amplificación R(z)
def amplification_euler(z: complex) -> complex:
    """R(z) = 1 + z."""
    return 1 + z


def amplification_implicit_euler(z: complex) -> complex:
    """R(z) = 1 / (1 − z)."""
    return 1 / (1 - z)


def amplification_crank_nicolson(z: complex) -> complex:
    """R(z) = (1 + z/2) / (1 − z/2)."""
    return (1 + z / 2) / (1 - z / 2)


def amplification_runge_kutta_4(z: complex) -> complex:
    """R(z) = 1 + z + z²/2 + z³/6 + z⁴/24 (desarrollo de e^z hasta orden 4)."""
    return 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24


def stability_grid(amplification: Callable[[Array], Array], re: tuple[float, float] = (-4.0, 2.0),
                   im: tuple[float, float] = (-3.5, 3.5), n: int = 301) -> tuple[Array, Array, Array]:
    """(x, y, |R|) en una malla n×n del plano z; A[i, j] = |R(x[i] + i·y[j])| (∞ donde no está definida)."""
    x, y = np.linspace(*re, n), np.linspace(*im, n)
    with np.errstate(all="ignore"):
        A = np.abs(amplification(x[:, None] + 1j * y[None, :]))
    return x, y, np.nan_to_num(A, nan=np.inf)


def error_exponents(amplification: Amplification, x: float = 0.05) -> tuple[Optional[float], float]:
    """Potencia de z de los errores por paso del esquema en el eje imaginario z = ix, medida numéricamente:
    (|R| − 1 ∝ z^a, arg R − x ∝ z^b). a es None si |R| = 1 (esquema que conserva el módulo)."""
    z = np.array([x, 2 * x])
    R = np.array([amplification(1j * v) for v in z])
    dmod, dph = np.abs(R) - 1.0, np.angle(R) - z
    a = None if np.all(np.abs(dmod) < 1e-13) else float(np.log2(abs(dmod[1] / dmod[0])))
    return a, float(np.log2(abs(dph[1] / dph[0])))


class Scheme(NamedTuple):
    """Esquema de un paso y sus propiedades teóricas (registro `SCHEMES`).

    `module_order` y `phase_order` describen el error de |R| y de arg R en el centro lineal (z = iβΔt);
    `module_order = None` significa |R| = 1 exactamente (Crank-Nicolson). Eso es una propiedad del
    problema LINEAL: en un problema no lineal, Crank-Nicolson solo conserva los invariantes cuadráticos
    de forma aproximada (deriva O(Δt²))."""
    step: Step                            # función de un paso
    amplification: Amplification          # R(z)
    order: int                            # orden global en Δt
    module_order: Optional[int] = None    # el error global de módulo es O(Δt^module_order); None: se conserva
    phase_order: int = 1                  # el error global de fase es O(Δt^phase_order)
    label: str = ""                       # nombre completo (para mostrar)
    short: str = ""                       # nombre corto (tablas y leyendas)
    implicit: bool = False                # resuelve Newton en cada paso (admite `jac=` con la Jacobiana exacta)


SCHEMES: dict[str, Scheme] = {
    "euler": Scheme(euler,          amplification_euler,          1, 1, 2, "Euler explícito", "Euler"),
    "be":    Scheme(implicit_euler, amplification_implicit_euler, 1, 1, 2, "Euler implícito (inverso)", "Euler imp.", True),
    "cn":    Scheme(crank_nicolson, amplification_crank_nicolson, 2, None, 2, "Crank-Nicolson (implícito)", "C-N", True),
    "rk4":   Scheme(runge_kutta_4,  amplification_runge_kutta_4,  4, 5, 4, "Runge-Kutta 4", "RK4"),
}


# ============================================================== integración
def _check_step(dt: float, n: int) -> None:
    """ValueError si Δt no es un real positivo y finito o si n es negativo."""
    if not (np.isfinite(dt) and dt > 0):
        raise ValueError(f"Δt debe ser positivo y finito (recibido {dt!r})")
    if int(n) != n or n < 0:
        raise ValueError(f"el número de pasos debe ser un entero ≥ 0 (recibido {n!r})")


def n_steps(T: float, dt: float) -> int:
    """Número de pasos de tamaño dt (al menos 1) que cubren el intervalo T."""
    _check_step(dt, 1)
    return max(1, int(round(T / dt)))


def cauchy_problem(f: Field, u0: Array, t: Array, scheme: Step,
                   stop: Callable[[Array], bool] | None = None) -> Array:
    """Integra u' = f(u, t) con `scheme` en los instantes t (paso Δ = t[n+1] − t[n]); devuelve (len(t), dim).
    Si stop(u) es True tras un paso, esa fila y las siguientes quedan en NaN y no se sigue calculando.
    Los esquemas explícitos que se desbordan devuelven inf/NaN; los implícitos lanzan RuntimeError o
    ArithmeticError si Newton no converge (quien llame decide si lo trata como divergencia)."""
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
    """n pasos de tamaño dt desde u0 en t0 → (n+1, dim). Con `limit`, la trayectoria se corta (NaN) en la
    primera fila cuyo Σ|u[components]| no es finito o llega a `limit`: es divergencia, no física.
    ValueError si dt no es positivo y finito o n < 0."""
    _check_step(dt, n)

    def blown(u: Array) -> bool:
        with np.errstate(all="ignore"):
            return not (np.abs(u[components]).sum() < limit)

    with np.errstate(all="ignore"):
        return cauchy_problem(f, u0, np.asarray(t0 + dt * np.arange(n + 1), dtype=float), scheme, stop=None if limit is None else blown)


def first_divergence(Y: Array, bound: float, components: Component = slice(0, 3)) -> int | None:
    """Índice de la primera fila de Y no finita o con ‖Y[components]‖ > bound; None si no hay ninguna."""
    with np.errstate(all="ignore"):
        bad = ~np.isfinite(Y).all(axis=1) | (np.linalg.norm(Y[:, components], axis=1) > bound)
    return int(np.argmax(bad)) if bad.any() else None


def final_state(scheme: Step, f: Field, u0: Array, T: float, N: int, t0: float = 0) -> Array:
    """Estado en t0 + T tras N pasos iguales (ValueError si N < 1 o T no es positivo)."""
    _check_richardson(T, N, 1.0)
    return cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)[-1, :]


# ============================================================== orden y error: Richardson
def _check_richardson(T: float, N: int, p: float) -> None:
    if not (np.isfinite(T) and T > 0):
        raise ValueError(f"T debe ser positivo y finito (recibido {T!r})")
    if int(N) != N or N < 1:
        raise ValueError(f"N debe ser un entero ≥ 1 (recibido {N!r})")
    if not (np.isfinite(p) and p > 0):
        raise ValueError(f"el orden p debe ser positivo (recibido {p!r})")


def scheme_order(scheme: Step, f: Field, u0: Array, T: float, N: int,
                 components: Component = slice(None)) -> float:
    """Orden observado del esquema con tres integraciones (N, 2N y 4N pasos), sin conocer la solución:
    p ≈ log₂(‖u_N − u_2N‖ / ‖u_2N − u_4N‖) sobre `components`. NaN si las diferencias entre mallas están al
    nivel del ruido de redondeo (el cociente no tendría sentido) o la integración diverge."""
    _check_richardson(T, N, 1.0)
    u1, u2, u3 = (final_state(scheme, f, u0, T, k * N) for k in (1, 2, 4))
    with np.errstate(all="ignore"):
        d12, d23 = norm((u1 - u2)[components]), norm((u2 - u3)[components])
        floor = 50 * np.finfo(float).eps * np.sqrt(4 * N) * max(1.0, float(norm(u3[components])))
        return float("nan") if not np.isfinite(d12 + d23) or d23 < floor else float(np.log2(d12 / d23))


def richardson_error(scheme: Step, f: Field, u0: Array, T: float, N: int, p: float, t0: float = 0) -> Array:
    """Estimación del error de la solución fina (2N pasos) en los N+1 instantes de la malla gruesa."""
    _check_richardson(T, N, p)
    u_coarse = cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)
    u_fine = cauchy_problem(f, u0, np.linspace(t0, t0 + T, 2 * N + 1), scheme)
    return (u_fine[0::2, :] - u_coarse) / (2 ** p - 1)


# no lo pide el enunciado PEI1. Obtiene una estimación mejor a partir de la estimación del error
def richardson_extrapolation(scheme: Step, f: Field, u0: Array, T: float, N: int, p: float) -> Array:
    """Estado en T extrapolado: w_fino + (w_fino − w_grueso)/(2^p − 1), de orden p + 1."""
    _check_richardson(T, N, p)
    w_coarse = final_state(scheme, f, u0, T, N)
    w_fine = final_state(scheme, f, u0, T, 2 * N)
    return w_fine + (w_fine - w_coarse) / (2 ** p - 1)


class RichardsonReport(NamedTuple):
    """Resultado de `richardson_analysis` (cada campo se documenta a su derecha)."""
    n: int                 # pasos de la malla gruesa
    p: float               # orden teórico usado
    err: float             # ‖error estimado‖ en T de la solución fina (2N pasos)
    u_extr: Array          # estado extrapolado en T
    err_real: float        # ‖u_fino − referencia‖ en T (NaN sin referencia)
    noise: bool = False    # True: la diferencia entre mallas está al nivel del redondeo y `err` no es fiable


def richardson_analysis(scheme: Step, f: Field, u0: Array, T: float, N: int, p: float,
                        components: Component = slice(0, 3),
                        reference: Callable[[float], Array] | None = None, t0: float = 0.0) -> RichardsonReport:
    """Richardson con dos integraciones (N y 2N pasos) y el orden teórico p: el error de la solución fina es
    (u_2N − u_N) / (2ᵖ − 1). `reference(T)` (opcional) da el estado exacto de `components` para medir el
    error real. `err` y `err_real` miden la misma cosa (norma sobre `components` en t0 + T). Si la diferencia
    entre mallas no supera el ruido de redondeo, `noise` es True. El orden observado se mide aparte
    (`scheme_order`)."""
    _check_richardson(T, N, p)
    with np.errstate(all="ignore"):
        runs = {k: cauchy_problem(f, u0, np.linspace(t0, t0 + T, k * N + 1), scheme) for k in (1, 2)}
        fin = {k: r[-1, :] for k, r in runs.items()}
        err = norm(((runs[2][0::2, :] - runs[1]) / (2 ** p - 1))[-1][components])
        u_extr = fin[2] + (fin[2] - fin[1]) / (2 ** p - 1)
        floor = 50 * np.finfo(float).eps * np.sqrt(2 * N) * max(1.0, float(norm(fin[2][components])))
        noise = bool(norm((fin[2] - fin[1])[components]) < floor)
        err_real = float(norm(fin[2][components] - reference(T))) if reference is not None else float("nan")
        return RichardsonReport(N, p, float(err), u_extr, err_real, noise)


# ============================================================== utilidades numéricas genéricas
def log_safe(d: Array, floor: float = 1e-18) -> Array:
    """|d| + floor: para dibujar errores en escala logarítmica sin ceros."""
    return np.abs(d) + floor


def mask_below(d: Array, floor: float) -> Array:
    """|d| donde supera `floor` y NaN donde no (valores indistinguibles del ruido)."""
    a = np.abs(d)
    return np.where(a > floor, a, np.nan)


def arrow_segments(X: Array, Y: Array, U: Array, V: Array, length: float, head: float = 0.38,
                   angle: float = 26.0) -> Array:
    """Segmentos (tallo + dos aletas) de las flechas de un campo (U, V) en los puntos (X, Y), todas de
    longitud `length` y con la cabeza `head·length`; forma (6·m, 2) lista para dibujar por pares.
    Se omiten los puntos con campo prácticamente nulo."""
    X, Y, U, V = (np.ravel(z) for z in (X, Y, U, V))
    sp = np.hypot(U, V)
    ok = sp > 1e-9 * max(sp.max(initial=0.0), 1e-300)
    if not ok.any():
        return np.zeros((0, 2))
    d = np.column_stack([U[ok], V[ok]]) / sp[ok, None]
    P = np.column_stack([X[ok], Y[ok]])
    tail, tip = P - 0.5 * length * d, P + 0.5 * length * d
    ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    left = tip - head * length * np.column_stack([ca * d[:, 0] - sa * d[:, 1], sa * d[:, 0] + ca * d[:, 1]])
    right = tip - head * length * np.column_stack([ca * d[:, 0] + sa * d[:, 1], -sa * d[:, 0] + ca * d[:, 1]])
    return np.stack([tail, tip, tip, left, tip, right], axis=1).reshape(-1, 2)


def level_curve(x: Array, y: Array, A: Array, level: float = 1.0) -> tuple[Array, Array]:
    """Curva de nivel A = level sobre la malla (x, y) por 'marching squares'; devuelve (X, Y) con NaN entre
    segmentos (listo para dibujar con conexión 'finite'). A[i, j] corresponde a (x[i], y[j])."""
    a, b, c, d = A[:-1, :-1], A[1:, :-1], A[1:, 1:], A[:-1, 1:]
    idx = (a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8
    idx = np.where(idx > 7, 15 - idx, idx)                     # el complemento da los mismos segmentos
    pairs = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)]}
    ii, jj = np.indices(idx.shape)

    def edge(e: int, i: Array, j: Array) -> tuple[Array, Array]:
        # e0: (i,j)-(i+1,j)  e1: (i+1,j)-(i+1,j+1)  e2: (i,j+1)-(i+1,j+1)  e3: (i,j)-(i,j+1)
        (i0, j0), (i1, j1) = (((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 0), (0, 1)))[e]
        p, q = A[i + i0, j + j0], A[i + i1, j + j1]
        with np.errstate(all="ignore"):
            s = np.clip((level - p) / (q - p), 0.0, 1.0)
        return (x[i + i0] + s * (x[i + i1] - x[i + i0]), y[j + j0] + s * (y[j + j1] - y[j + j0]))

    out = []
    for case, segs in pairs.items():
        sel = idx == case
        if not sel.any():
            continue
        i, j = ii[sel], jj[sel]
        for e0, e1 in segs:
            (xa, ya), (xb, yb) = edge(e0, i, j), edge(e1, i, j)
            out.append(np.column_stack([xa, xb, np.full_like(xa, np.nan)]).ravel())
            out.append(np.column_stack([ya, yb, np.full_like(ya, np.nan)]).ravel())
    if not out:
        return np.array([]), np.array([])
    return np.concatenate(out[0::2]), np.concatenate(out[1::2])


def stability_contour(amplification: Callable[[Array], Array], re: tuple[float, float], im: tuple[float, float],
                      n: int = 301) -> tuple[Array, Array, npt.NDArray[np.bool_], tuple[float, float, float, float]]:
    """Frontera |R(z)| = 1, máscara |R| ≤ 1 y rectángulo (x0, y0, ancho, alto) de la ventana re × im."""
    x, y, A = stability_grid(amplification, re, im, n)
    X, Y = level_curve(x, y, np.minimum(A, 10.0), 1.0)
    return X, Y, A <= 1.0, (float(x[0]), float(y[0]), float(x[-1] - x[0]), float(y[-1] - y[0]))


def stability_window(z: Array, zoom: bool) -> tuple[tuple[tuple[float, float], tuple[float, float]],
                                                  tuple[float, float], tuple[float, float]]:
    """(vista, re, im) para dibujar las regiones de estabilidad. `vista` = ((x0, x1), (y0, y1)) es lo que se
    muestra; re e im son la malla calculada (mayor que la vista, para que el contorno llegue al borde).
    Con `zoom` la vista se centra en el origen y contiene los puntos z = λΔt dados (valor redondeado a 2 cifras
    para que la caché de contornos acierte más)."""
    if zoom and len(z):
        r = float(f"{max(3.0 * float(np.max(np.abs(z))), 0.02):.2g}")
        return ((-r, r), (-r, r)), (-3 * r, 3 * r), (-3 * r, 3 * r)
    return ((-4.0, 2.0), (-3.5, 3.5)), (-7.0, 5.0), (-4.5, 4.5)

