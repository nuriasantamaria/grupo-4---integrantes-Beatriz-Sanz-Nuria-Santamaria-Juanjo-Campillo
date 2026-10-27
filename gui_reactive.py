"""Interfaz gráfica (PySide6 + pyqtgraph) para simular y analizar la rotación libre de un sólido rígido"""

from __future__ import annotations

import html
import math
import sys
import time
from dataclasses import replace
from functools import partial, lru_cache
from typing import Any, Callable, NamedTuple, Sequence, TypedDict, Union

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_CULL_FACE
from pyqtgraph.opengl.GLGraphicsItem import GLOptions

import numerical_engine as ne
from simulation import (SimConfig, Simulation, run_preview, richardson_report, convergence_report, run_each,
                        linear_errors, MAX_LIVE_STEPS, MAX_PREVIEW_STEPS)
from space_physics import (kinetic_energy, angular_momentum_sq, equilibrium_analysis, EulerExact, SEP_M, SEPARATRIX_TOL,
                           separatrix_side,
                           relevant_eigenvalues, equilibrium_amplification, noise_floor, predicted_errors,
                           sorted_axes, is_degenerate, surrounded_axis, euler_equations,
                           Array, ArrayLike, LinearErrors, Vec)

IntArray = NDArray[np.intp]
Range = tuple[float, float]               # intervalo (mínimo, máximo)
Mesh = tuple[Array, IntArray]            # (vértices, caras triangulares)

# esquemas: claves del registro de numerical_engine (etiqueta, orden y R(z) se leen de ahí)
M_EULER, M_BE, M_CN, M_RK4 = "euler", "be", "cn", "rk4"
METHOD_LABEL = {k: s.label for k, s in ne.SCHEMES.items()}
METHOD_SHORT = {k: s.short for k, s in ne.SCHEMES.items()}
SCHEME_COLORS = {M_EULER: "#ff8a80", M_BE: "#b39ddb", M_CN: "#ffd740", M_RK4: "#80cbc4"}

GRADE_WORDS = ("≈ 0", "ínfimo", "relevante")
AMBER = "#FFD740"                # color de aviso y de la órbita exacta de referencia
GRADE_COLORS = ("#8a8f98", "#9fb4c9", AMBER)


def _power(n: int | None) -> str:
    """Texto del orden en dt: dt, dt2, dt3... (None si el esquema conserva el módulo)"""
    return "nulo (|R| = 1)" if n is None else "Δt" + ("" if n == 1 else "⁰¹²³⁴⁵⁶⁷⁸⁹"[n])


# orden global en dt del error de módulo y de fase de cada esquema en el centro lineal (registro de ne.SCHEMES)
SCHEME_ORDER = {k: (_power(s.module_order), _power(s.phase_order)) for k, s in ne.SCHEMES.items()}

STABILITY_COLORS = {"estable": "#69F0AE", "marginal": "#80D8FF", "inestable": "#FF6B6B", "neutro": "#80D8FF"}


@lru_cache(maxsize=64)
def stability_layer(method: str, re: tuple[float, float], im: tuple[float, float]):
    """Contorno |R(z)| = 1, máscara |R| <= 1 y ventana de un esquema, en caché para no recalcular al mover los sliders"""
    return stability_contour(ne.SCHEMES[method].amplification, re, im)


# ============================================================== Apoyo a la visualización
# No intervienen en la resolución del problema: space_physics y numerical_engine no los usan.

# ---- inercia y forma del cuerpo

def inertia_ranks(I: ArrayLike, tol: float = 1e-9) -> list[str]:
    """Etiqueta de cada momento ('mín', 'medio', 'máx'), o '=' si empata con otro"""
    I = np.asarray(I, dtype=float)
    tags = [""] * 3
    for rank, k in enumerate(np.argsort(I, kind="stable")):
        tags[k] = ("mín", "medio", "máx")[rank]
    for k in range(3):
        if any(j != k and abs(I[j] - I[k]) < tol for j in range(3)):
            tags[k] = "="
    return tags


def triangle_violation(I: ArrayLike, tol: float = 1e-9) -> int | None:
    """Índice k con I_k > I_i + I_j (viola la desigualdad triangular), o None"""
    I = np.asarray(I, dtype=float)
    return next((k for k in range(3) if I[k] > I[(k + 1) % 3] + I[(k + 2) % 3] + tol), None)


def body_half_sizes(I: Vec, min_rel: float = 0.15) -> Array:
    """Semiejes relativos (máx = 1) de una caja homogénea con esos momentos de inercia; los imposibles se acotan en min_rel"""
    I1, I2, I3 = (float(x) for x in I)
    s = np.sqrt(np.maximum([I2 + I3 - I1, I1 + I3 - I2, I1 + I2 - I3], 0.0))
    return np.asarray(np.maximum(s / s.max(), min_rel), dtype=float)

# ---- actitud 3D

def quat_to_matrix(q: ArrayLike) -> Array:
    """Matriz de rotación cuerpo -> inercial del cuaternión q = (w, x, y, z), que se normaliza"""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def attitude_vectors(q: ArrayLike, I: ArrayLike, w: ArrayLike, L0_norm: float,
                     w0_norm: float) -> tuple[Array, Array, Array]:
    """Devuelve (R, L, omega) en ejes inerciales para dibujar la actitud, normalizados con las normas iniciales"""
    R = quat_to_matrix(q)
    w = np.asarray(w, dtype=float)
    return R, R @ (np.asarray(I, dtype=float) * w) / max(L0_norm, 1e-12), R @ w / max(w0_norm, 1e-12)

# ---- curvas 3D: elipsoides, polhodias y separatriz

def poinsot_semiaxes(I: ArrayLike, w0: ArrayLike) -> tuple[Array, Array]:
    """Semiejes de los elipsoides de energía y de momento en el espacio de omega"""
    I = np.asarray(I, dtype=float)
    twoT = 2 * kinetic_energy(w0, I)
    L = np.sqrt(angular_momentum_sq(w0, I))
    return np.sqrt(twoT / I), L / I


def separatrix_curves(I: ArrayLike, twoT: float, n_pts: int = 300) -> list[Array]:
    """Las dos curvas separatriz (n, 3) sobre el elipsoide de energía; vacío si es simétrico o sin energía"""
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    k = math.sqrt(Ia * (Ib - Ia) / (Ic * (Ic - Ib)))
    # 4m+1 puntos: incluye s = pi/2 y 3pi/2 (los equilibrios)
    s = np.linspace(0, 2 * np.pi, 4 * max(n_pts // 4, 1) + 1)
    n_pts = len(s)
    A = math.sqrt(twoT / (Ia + Ic * k * k))
    B = math.sqrt(twoT / Ib)
    curves = []
    for sign in (+1, -1):
        P = np.zeros((n_pts, 3))
        P[:, a] = A * np.cos(s)
        P[:, b] = B * np.sin(s)
        P[:, c] = sign * k * A * np.cos(s)
        curves.append(P)
    return curves


def polhode_family(I: ArrayLike, twoT: float, n_levels: int = 5, n_pts: int = 200) -> list[Array]:
    """Familia de polhodias (órbitas de omega sobre el elipsoide de energía) alrededor de los ejes extremos"""
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    s = np.linspace(0, 2 * np.pi, n_pts)
    curves = []
    levels = np.linspace(0.15, 0.9, n_levels)

    for f in levels:
        L2 = twoT * (Ia + f * (Ib - Ia))
        wc = math.sqrt((L2 - twoT * Ia) / (Ic * (Ic - Ia))) * np.cos(s)
        wb = math.sqrt((L2 - twoT * Ia) / (Ib * (Ib - Ia))) * np.sin(s)
        wa = np.sqrt(np.maximum(twoT - Ib * wb ** 2 - Ic * wc ** 2, 0) / Ia)
        for sign in (+1, -1):
            P = np.zeros((n_pts, 3))
            P[:, a], P[:, b], P[:, c] = sign * wa, wb, wc
            curves.append(P)

    for f in levels:
        L2 = twoT * (Ic - f * (Ic - Ib))
        wa = math.sqrt((twoT * Ic - L2) / (Ia * (Ic - Ia))) * np.cos(s)
        wb = math.sqrt((twoT * Ic - L2) / (Ib * (Ic - Ib))) * np.sin(s)
        wc = np.sqrt(np.maximum(twoT - Ia * wa ** 2 - Ib * wb ** 2, 0) / Ic)
        for sign in (+1, -1):
            P = np.zeros((n_pts, 3))
            P[:, a], P[:, b], P[:, c] = wa, wb, sign * wc
            curves.append(P)
    return curves

# ---- plano de fases (carta de dos componentes)

def chart_axes(k: int) -> tuple[int, int]:
    """Ejes (i, j) de la carta omega_i-omega_j sobre el hemisferio de omega_k"""
    return (k + 1) % 3, (k + 2) % 3


def default_chart(I: ArrayLike, w0: Vec) -> tuple[int, int]:
    """(eje k, signo) de la carta natural: la del eje que rodea la órbita (o el intermedio) y el hemisferio de omega0"""
    k = surrounded_axis(I, w0)
    if k is None:
        k = sorted_axes(I)[1]
    return k, (1 if w0[k] >= 0 else -1)


def project_to_chart(W: Array, k: int, sign: int) -> Array:
    """Proyecta la trayectoria W (n, 3) sobre la carta; NaN donde omega_k tiene el signo contrario"""
    i, j = chart_axes(k)
    P = np.asarray(W)[:, [i, j]].astype(float)
    wrong_side = ~(sign * np.asarray(W)[:, k] > 0)     # también atrapa los NaN
    P[wrong_side] = np.nan
    return P


def chart_domain(I: ArrayLike, twoT: float, k: int, n_pts: int = 300) -> Array:
    """Borde (n, 2) de la región admisible de la carta: la elipse I_i w_i^2 + I_j w_j^2 = 2T"""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    s = np.linspace(0, 2 * np.pi, n_pts)
    return np.column_stack([math.sqrt(twoT / I[i]) * np.cos(s), math.sqrt(twoT / I[j]) * np.sin(s)])


def reduced_flow(I: ArrayLike, twoT: float, k: int, sign: int, X: ArrayLike, Y: ArrayLike) -> tuple[Array, Array]:
    """Derivadas de (omega_i, omega_j) del flujo en la carta, con omega_k despejado de la energía"""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    W = np.zeros((3,) + X.shape)
    W[i], W[j] = X, Y
    W[k] = sign * np.sqrt(np.maximum(twoT - I[i] * X ** 2 - I[j] * Y ** 2, 0) / I[k])
    dW = euler_equations(W, np.diag(I))
    return dW[i], dW[j]


def reduced_flow_field(I: ArrayLike, twoT: float, k: int, sign: int, n: int = 21,
                       margin: float = 0.96) -> tuple[Array, Array, Array, Array, float]:
    """Campo de direcciones del plano de fases: (X, Y, U, V, paso) en una malla n x n dentro del dominio de la carta"""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    D = chart_domain(I, twoT, k)
    xm, ym = float(D[:, 0].max()), float(D[:, 1].max())
    X, Y = np.meshgrid(np.linspace(-xm, xm, n), np.linspace(-ym, ym, n))
    inside = I[i] * X ** 2 + I[j] * Y ** 2 < margin * twoT
    U, V = reduced_flow(I, twoT, k, sign, X[inside], Y[inside])
    return X[inside], Y[inside], U, V, min(2 * xm, 2 * ym) / (n - 1)


def phase_plane_orbits(I: ArrayLike, twoT: float, k: int, n_levels: int = 9,
                       n_pts: int = 400) -> tuple[list[Array], list[Array]]:
    """Devuelve (órbitas, separatrices) de la carta k como curvas de nivel a w_i^2 + b w_j^2 = c"""
    if is_degenerate(I) or twoT <= 0:
        return [], []
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    fr = np.linspace(0.12, 1.0, n_levels) ** 2
    qa, qb = a * twoT / I[i], b * twoT / I[j]
    if a * b > 0:
        levels = fr * (qa if abs(qa) > abs(qb) else qb)
        seps: list[Array] = []
    else:
        levels = np.concatenate([fr * qa, fr * qb])
        seps = chart_level_curve(I, twoT, k, 0.0, n_pts)
    orbits = [P for c in levels for P in chart_level_curve(I, twoT, k, c, n_pts)]
    return orbits, seps


def _hyperbola_branches(A: float, B: float, c: float, XM: float, YM: float,
                        n_pts: int) -> list[tuple[Array, Array]]:
    """Ramas (x, y) de A x^2 + B y^2 = c con A > 0 > B, recortadas a |x| <= XM y |y| <= YM"""
    signs = (1, -1)
    if c > 0:       # ramas derecha e izquierda: x = +-r cosh s, y = q sinh s
        r, q = math.sqrt(c / A), math.sqrt(c / -B)
        s = np.linspace(-1, 1, n_pts) * math.asinh(YM / q)
        return [(sg * r * np.cosh(s), q * np.sinh(s)) for sg in signs]
    if c < 0:       # ramas superior e inferior: x = q sinh s, y = +-r cosh s
        r, q = math.sqrt(c / B), math.sqrt(-c / A)
        s = np.linspace(-1, 1, n_pts) * math.asinh(XM / q)
        return [(q * np.sinh(s), sg * r * np.cosh(s)) for sg in signs]
    slope = math.sqrt(A / -B)
    x = np.linspace(-XM, XM, n_pts)
    return [(x, slope * x), (x, -slope * x)]


def chart_level_curve(I: ArrayLike, twoT: float, k: int, c: float, n_pts: int = 400) -> list[Array]:
    """Curvas (n, 2) del nivel a w_i^2 + b w_j^2 = c recortadas al dominio de la carta"""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    xm, ym = math.sqrt(twoT / I[i]), math.sqrt(twoT / I[j])

    def clip(x: Array, y: Array) -> Array:
        """Pone a NaN los puntos fuera del dominio de la carta"""
        P = np.column_stack([x, y]).astype(float)
        P[I[i] * P[:, 0] ** 2 + I[j] * P[:, 1] ** 2 > twoT * (1 + 1e-9)] = np.nan
        return P

    if a * b > 0:
        if c / a <= 0:
            return []
        s = np.linspace(0, 2 * np.pi, n_pts)
        return [clip(math.sqrt(c / a) * np.cos(s), math.sqrt(c / b) * np.sin(s))]

    # hipérbola: se pone el coeficiente positivo en la primera variable y, si hace falta, se intercambian los ejes
    swapped = a < 0
    if swapped:
        A, B, XM, YM = b, a, ym, xm
    else:
        A, B, XM, YM = a, b, xm, ym
    branches = _hyperbola_branches(A, B, c, XM, YM, n_pts)
    if swapped:
        return [clip(second, first) for first, second in branches]
    return [clip(first, second) for first, second in branches]


def chart_reference_orbit(I: ArrayLike, w0: ArrayLike, k: int, sign: int, n_pts: int = 600) -> Array | None:
    """Órbita exacta de omega0 en la carta (k, sign) con NaN entre ramas; None si no está en ese hemisferio o es simétrico"""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    if not (sign * w0[k] > 0) or is_degenerate(I):
        return None
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    curves = chart_level_curve(I, 2 * float(kinetic_energy(w0, I)), k, a * w0[i] ** 2 + b * w0[j] ** 2, n_pts)
    return np.vstack([np.vstack([C, [[np.nan, np.nan]]]) for C in curves]) if curves else None

# ---- edicion de las condiciones iniciales y comparacion de problemas

def axis_decomposition(w: Sequence[float], keep: tuple[int, ...] | None = None) -> tuple[int, int, int]:
    """Terna (principal, perturbación 1, perturbación 2) ordenada por |omega|; con omega nulo mantiene el principal de keep"""
    size = [abs(x) for x in w]
    if keep is not None and max(size) < 1e-12:
        a = keep[0]
    else:
        a = int(np.argmax(size))
    others = [k for k in range(3) if k != a]
    # sorted es estable: en empate gana el índice menor
    b, c = sorted(others, key=lambda k: -size[k])
    return a, b, c


def omega_from_axis(axes: tuple[int, int, int], main: float, p1: float, p2: float = 0.0) -> list[float]:
    """Construye omega0 con el valor principal y las dos perturbaciones sobre sus ejes"""
    a, b, c = axes
    w = [0.0] * 3
    w[a], w[b], w[c] = main, p1, p2
    return w


def reassign_axes(axes: tuple[int, int, int], new_main: int | None = None,
                  new_p1: int | None = None) -> tuple[int, int, int]:
    """Nueva terna de ejes al elegir otro eje principal o otra primera perturbación"""
    a, b, c = axes
    if new_main is not None and new_main != a:
        rest = [k for k in range(3) if k != new_main]
        b2 = b if b in rest else a
        return new_main, b2, next(k for k in rest if k != b2)
    if new_p1 is not None and new_p1 != b:
        return a, new_p1, next(k for k in range(3) if k not in (a, new_p1))
    return a, b, c


def same_problem(I_a: ArrayLike, w_a: ArrayLike, I_b: ArrayLike, w_b: ArrayLike) -> bool:
    """Indica si coinciden I y omega0, es decir, si la solución exacta es la misma"""
    return bool(np.allclose(I_a, I_b, rtol=1e-12, atol=0) and np.allclose(w_a, w_b, rtol=1e-12, atol=1e-15))


SURFACE_TOL = 1e-3


class SurfaceMismatch(TypedDict):
    """Diferencia entre las superficies de dos problemas"""
    same_I: bool
    dT: float              # diferencia relativa de energía
    dL: float              # diferencia relativa del momento angular al cuadrado


def surface_mismatch(I_ref: ArrayLike, w_ref: ArrayLike, I: ArrayLike, w: ArrayLike,
                     tol: float = SURFACE_TOL) -> SurfaceMismatch | None:
    """None si la trayectoria de (I, w) está sobre las superficies de referencia (a tol); si no, las diferencias relativas"""
    Ir, Ic = np.asarray(I_ref, dtype=float), np.asarray(I, dtype=float)
    Tr, Tc = float(kinetic_energy(w_ref, Ir)), float(kinetic_energy(w, Ic))
    Lr, Lc = float(angular_momentum_sq(w_ref, Ir)), float(angular_momentum_sq(w, Ic))

    def rel(a: float, b: float) -> float:
        return (b / a - 1.0) if a > 0 else (0.0 if b == 0 else math.inf)

    same_I = bool(np.allclose(Ir, Ic, rtol=1e-9, atol=1e-12))
    dT, dL = rel(Tr, Tc), rel(Lr, Lc)
    if same_I and abs(dT) <= tol and abs(dL) <= tol:
        return None
    return {"same_I": same_I, "dT": dT, "dL": dL}

# ---- veredictos del analisis (texto y gravedad)

# por debajo de esto no se ve en ninguna gráfica (módulo adimensional, fase en grados)
TINY_R, TINY_PSI = 1e-6, 1e-3


def grade(x: float, floor: float, tiny: float) -> int:
    """0: ruido de redondeo, 1: ínfimo, 2: a tener en cuenta"""
    size = abs(x)
    if size <= floor:
        return 0
    if size < tiny:
        return 1
    return 2


def grade_pair(r: float, psi: float, n: int) -> tuple[int, int]:
    """Gravedad del error de módulo r y de fase psi (en grados) tras n pasos"""
    fl = noise_floor(n)
    return grade(r, fl, TINY_R), grade(psi, math.degrees(fl), TINY_PSI)


# dt*beta por encima de esto (unos 25 pasos por ciclo) el paso es grueso
COARSE_BETA_DT = 0.25


# |R| por encima de esto el error de módulo explota en pocos pasos
EXPLOSION_R = 1.5


# |R| - 1 por debajo de esto se considera que se conserva
TREND_TOL = 1e-10


def amplitude_trend(dmod: float, tol: float = TREND_TOL) -> str:
    """'crece', 'decae' o 'se conserva' según el signo de |R| - 1 por paso"""
    if dmod > tol:
        return "crece"
    if dmod < -tol:
        return "decae"
    return "se conserva"


class StepResolution(TypedDict):
    """Pasos por ciclo del esquema y si la teoría lineal lo da por perdido"""
    steps_per_cycle: float   # periodo de oscilación / dt
    coarse: bool             # dt*beta > COARSE_BETA_DT
    explodes: bool           # |R| > EXPLOSION_R


def step_resolution(lin: LinearErrors, dt: float) -> StepResolution:
    """Resolución temporal de dt frente a la oscilación del centro"""
    return {"steps_per_cycle": lin["period"] / dt, "coarse": lin["beta"] * dt > COARSE_BETA_DT,
            "explodes": abs(lin["R"]) > EXPLOSION_R}


def scheme_verdict(lin: LinearErrors, n: int) -> tuple[tuple[str, int], tuple[str, int]]:
    """Qué prevé la teoría lineal para módulo y fase tras n pasos, como (palabra, gravedad) de cada uno"""
    r, psi = predicted_errors(lin, n)
    gr, gp = grade_pair(r, psi, n)
    if gr == 0:
        mod = "se conserva"
    else:
        mod = "crece" if lin["dmod"] > 0 else "decae"
    if gp == 0:
        pha = "sin error"
    else:
        pha = "se retrasa" if lin["dphase"] < 0 else "se adelanta"
    return (mod, gr), (pha, gp)

# ---- escenarios predefinidos

class ScenarioSim(TypedDict, total=False):
    """Una simulación de un escenario; lo que falta se deja como está"""
    enabled: bool
    method: str
    dt: float
    I: tuple[float, float, float]
    w0: tuple[float, float, float]


class Scenario(NamedTuple):
    """Escenario didáctico: nota, simulaciones A, B y C, y si los errores van en escala logarítmica"""
    note: str
    sims: tuple[ScenarioSim, ...]
    log_errors: bool = False


_I123 = (1.0, 2.0, 3.0)


_JANI_W0 = (0.01, 2.0, 0.01)    # rotación casi pura sobre el eje intermedio


_MIN_W0 = (2.0, 0.3, 0.3)       # rotación sobre el eje de menor inercia


def _sim(method: str, dt: float, w0: tuple[float, float, float], I: tuple[float, float, float] = _I123,
         enabled: bool = True) -> ScenarioSim:
    return ScenarioSim(enabled=enabled, method=method, dt=dt, I=I, w0=w0)


SCENARIOS: dict[str, Scenario] = {
    "Comparar integradores (mismo Δt)": Scenario(
        "Mismo problema y mismo Δt con Euler, Crank-Nicolson y RK4: compara cómo derivan T y ‖L‖.",
        tuple(_sim(m, 0.02, _JANI_W0) for m in ("euler", "cn", "rk4")), log_errors=True),
    "Efecto Janibekov (eje intermedio)": Scenario(
        "Rotación casi pura sobre el eje intermedio con una pequeña perturbación: observa el vuelco.",
        (_sim("rk4", 0.01, _JANI_W0), _sim("cn", 0.01, _JANI_W0, enabled=False),
         _sim("euler", 0.01, _JANI_W0, enabled=False))),
    "Estable: eje de menor inercia": Scenario(
        "Rotación sobre el eje de menor inercia: las órbitas próximas lo rodean.",
        (_sim("rk4", 0.01, (2.0, 0.08, 0.08)), ScenarioSim(enabled=False), ScenarioSim(enabled=False))),
    "Estable: eje de mayor inercia": Scenario(
        "Rotación sobre el eje de mayor inercia: las órbitas próximas lo rodean.",
        (_sim("rk4", 0.01, (0.08, 0.08, 2.0)), ScenarioSim(enabled=False), ScenarioSim(enabled=False))),
    "Separatriz: un lado y el otro": Scenario(
        "Dos Ω₀ casi iguales a ambos lados de la separatriz: una órbita rodea el eje mayor y la otra el menor.",
        (_sim("rk4", 0.01, (0.85, 1.0, 0.5)), _sim("rk4", 0.01, (0.88, 1.0, 0.5)), ScenarioSim(enabled=False))),
    "Comparar inercias (mismo Ω₀)": Scenario(
        "Mismo Ω₀ con tres cuerpos distintos (B y C con física propia): cambia la órbita con la I₂.",
        (_sim("rk4", 0.01, _JANI_W0), _sim("rk4", 0.01, _JANI_W0, (1.0, 2.7, 3.0)),
         _sim("rk4", 0.01, _JANI_W0, (1.0, 1.3, 3.0)))),
    "Euler vs Crank-Nicolson vs RK4 (Δt grande)": Scenario(
        "Los tres esquemas con Δt = 0.08 sobre el mismo problema: observa cuál se desvía antes.",
        tuple(_sim(m, 0.08, _JANI_W0) for m in ("euler", "cn", "rk4")), log_errors=True),
    "Euler explícito vs implícito vs C-N (Δt grande)": Scenario(
        "Euler explícito, implícito y C-N con Δt = 0.08 girando sobre el eje de menor inercia: "
        "observa si la energía crece, decae o se conserva.",
        tuple(_sim(m, 0.08, _MIN_W0) for m in ("euler", "be", "cn")), log_errors=True),
}


DEFAULT_SCENARIO = "Comparar integradores (mismo Δt)"

# ---- escalas y flechas de los graficos

def stability_window(z: Array, zoom: bool) -> tuple[tuple[Range, Range], Range, Range]:
    """Devuelve (vista, re, im) de las regiones de estabilidad; la malla es mayor que la vista y con zoom se ajusta a los puntos z"""
    if zoom and len(z):
        # se redondea a 2 cifras para que la caché de contornos acierte más
        r = float(f"{max(3.0 * float(np.max(np.abs(z))), 0.02):.2g}")
        return ((-r, r), (-r, r)), (-3 * r, 3 * r), (-3 * r, 3 * r)
    return ((-4.0, 2.0), (-3.5, 3.5)), (-7.0, 5.0), (-4.5, 4.5)


def log_safe(d: Array, floor: float = 1e-18) -> Array:
    """Valor absoluto más un suelo, para dibujar errores en escala logarítmica sin ceros"""
    return np.abs(d) + floor


def mask_below(d: Array, floor: float) -> Array:
    """Valor absoluto donde supera floor y NaN donde no (indistinguible del ruido)"""
    a = np.abs(d)
    return np.where(a > floor, a, np.nan)


def arrow_segments(X: Array, Y: Array, U: Array, V: Array, length: float, head: float = 0.38,
                   angle: float = 26.0) -> Array:
    """Segmentos (tallo y dos aletas) de flechas del campo (U, V), todas de longitud length, listos para dibujar por pares"""
    X, Y, U, V = (np.ravel(z) for z in (X, Y, U, V))
    speed = np.hypot(U, V)
    ok = speed > 1e-9 * max(speed.max(initial=0.0), 1e-300)
    if not ok.any():
        return np.zeros((0, 2))
    d = np.column_stack([U[ok], V[ok]]) / speed[ok, None]     # dirección unitaria de cada flecha
    P = np.column_stack([X[ok], Y[ok]])
    tail, tip = P - 0.5 * length * d, P + 0.5 * length * d
    ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    left = tip - head * length * np.column_stack([ca * d[:, 0] - sa * d[:, 1], sa * d[:, 0] + ca * d[:, 1]])
    right = tip - head * length * np.column_stack([ca * d[:, 0] + sa * d[:, 1], -sa * d[:, 0] + ca * d[:, 1]])
    return np.stack([tail, tip, tip, left, tip, right], axis=1).reshape(-1, 2)


# ---- solución exacta: puntos para dibujar


def exact_sample(ex: EulerExact, t_end: float, max_points: int = 8000) -> tuple[Array, Array]:
    """Muestrea omega(t) en [0, t_end] con unos 100 puntos por periodo (1500 si no hay periodo)"""
    if t_end <= 0:
        return np.empty(0), np.empty((0, 3))
    if math.isfinite(ex.period):
        n = int(min(max_points, max(600, 100 * t_end / ex.period)))
    else:
        n = 1500
    t = np.linspace(0.0, t_end, n)
    return t, ex.omega(t)



def exact_orbit(ex: EulerExact, n: int = 720) -> Array | None:
    """Puntos (n, 3) de la órbita de omega en ejes del cuerpo; None en equilibrio"""
    if ex.kind == "equilibrio":
        return None
    # se parametriza por el ángulo de amplitud, no por el tiempo, para resolver bien cerca de la separatriz
    if ex.m >= SEP_M:
        phi = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n + 1)[1:-1]
    else:
        phi = np.linspace(0.0, 2 * math.pi, n + 1)
    out = np.empty((len(phi), 3))
    s = np.sin(phi)
    out[:, ex.axes] = ex.amp * np.stack([np.sqrt(1.0 - ex.m * s * s), np.cos(phi), s], axis=-1)
    return out



# ---- estabilidad de la órbita (etiquetas de la GUI)

def regime(I: ArrayLike, w0: ArrayLike) -> str:
    """Régimen geométrico de la órbita (reposo, esfera, simétrico, separatriz o precesión); la estabilidad la da orbit_stability"""
    twoT = 2 * kinetic_energy(w0, I)
    if twoT <= 0:
        return "reposo (ω = 0)"
    if np.ptp(np.asarray(I, dtype=float)) < 1e-6:
        return "esfera inercial"
    if is_degenerate(I):
        return "sólido simétrico"
    side = separatrix_side(I, w0)
    if abs(side) < SEPARATRIX_TOL:
        return "sobre la separatriz"
    if side > 0:
        return "precesa en torno al eje mayor"
    return "precesa en torno al eje menor"


# desviación máxima del eje de giro (grados) a partir de la cual se considera que la rotación vuelca
FLIP_DEG = 90.0


@lru_cache(maxsize=64)
def _max_deviation(I: tuple[float, ...], w0: tuple[float, ...]) -> float:
    """Versión con caché de max_deviation, con argumentos hashables"""
    orbit = exact_orbit(EulerExact(I, w0), 720)
    if orbit is None:
        return 0.0
    dot = np.sum(orbit * np.asarray(w0), axis=1)                 # producto escalar con w0, sin BLAS
    cos = dot / (np.linalg.norm(orbit, axis=1) * np.linalg.norm(w0))
    return float(np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0))))


def max_deviation(I: ArrayLike, w0: ArrayLike) -> float:
    """Ángulo máximo, en grados, entre omega(t) y omega0 a lo largo de la órbita exacta"""
    # distingue el eje intermedio: allí llega a ~180 grados por pequeña que sea la perturbación
    return _max_deviation(tuple(float(x) for x in np.ravel(I)), tuple(float(x) for x in np.ravel(w0)))


def orbit_stability(I: ArrayLike, w0: ArrayLike) -> tuple[str, str]:
    """Estabilidad física de la rotación: (texto, 'estable' | 'marginal' | 'inestable')"""
    r = regime(I, w0)
    if "reposo" in r:
        return "en reposo · sin rotación", "marginal"
    if "esfera" in r:
        return "esfera inercial · ω constante", "marginal"
    if "simétrico" in r:
        return "sólido simétrico · precesión uniforme", "marginal"
    dev = max_deviation(I, w0)
    ax = surrounded_axis(I, w0)
    if ax is None:
        donde = " · sobre la separatriz"
    else:
        lado = "mayor" if "mayor" in r else "menor"
        donde = f" · rodea el eje {lado} e{'₁₂₃'[ax]}"
    if dev >= FLIP_DEG:
        return f"inestable · vuelca: ω llega a {dev:.0f}° de su dirección inicial{donde}", "inestable"
    return f"estable · ω se desvía como máximo {dev:.0f}° de su dirección inicial{donde}", "estable"


# ---- región de estabilidad: contorno

# aristas de una celda como pares de esquinas (di, dj): 0 abajo, 1 derecha, 2 arriba, 3 izquierda
_CELL_EDGES = (((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 0), (0, 1)))


# Segmentos de la curva en una celda según el código de esquinas (1..7), como pares de aristas
_CELL_SEGMENTS = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)]}


def level_curve(x: Array, y: Array, A: Array, level: float = 1.0) -> tuple[Array, Array]:
    """Curva de nivel A = level por marching squares, con NaN entre segmentos"""
    # bits 1, 2, 4, 8 según supere el nivel la esquina (i,j), (i+1,j), (i+1,j+1) o (i,j+1)
    a, b, c, d = A[:-1, :-1], A[1:, :-1], A[1:, 1:], A[:-1, 1:]
    code = (a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8
    code = np.where(code > 7, 15 - code, code)     # el complemento da los mismos segmentos
    ii, jj = np.indices(code.shape)

    def crossing(edge: int, i: Array, j: Array) -> tuple[Array, Array]:
        """Punto donde la curva corta la arista edge de las celdas (i, j), por interpolación lineal"""
        (i0, j0), (i1, j1) = _CELL_EDGES[edge]
        p, q = A[i + i0, j + j0], A[i + i1, j + j1]
        with np.errstate(all="ignore"):
            s = np.clip((level - p) / (q - p), 0.0, 1.0)
        return x[i + i0] + s * (x[i + i1] - x[i + i0]), y[j + j0] + s * (y[j + j1] - y[j + j0])

    out = []
    for case, segments in _CELL_SEGMENTS.items():
        sel = code == case
        if not sel.any():
            continue
        i, j = ii[sel], jj[sel]
        for e0, e1 in segments:
            (xa, ya), (xb, yb) = crossing(e0, i, j), crossing(e1, i, j)
            out.append(np.column_stack([xa, xb, np.full_like(xa, np.nan)]).ravel())
            out.append(np.column_stack([ya, yb, np.full_like(ya, np.nan)]).ravel())
    if not out:
        return np.array([]), np.array([])
    return np.concatenate(out[0::2]), np.concatenate(out[1::2])


def stability_contour(amplification: Callable[[Array], Array], re: Range, im: Range,
                      n: int = 301) -> tuple[Array, Array, NDArray[np.bool_], tuple[float, float, float, float]]:
    """Frontera |R(z)| = 1, máscara |R| <= 1 y rectángulo (x0, y0, ancho, alto) de la ventana"""
    x, y, A = ne.stability_grid(amplification, re, im, n)
    X, Y = level_curve(x, y, np.minimum(A, 10.0), 1.0)
    return X, Y, A <= 1.0, (float(x[0]), float(y[0]), float(x[-1] - x[0]), float(y[-1] - y[0]))


class PreviewWorker(QtCore.QThread):
    """Calcula las previsualizaciones fuera del hilo de la interfaz"""

    done = QtCore.Signal(int, object, float)       # token, {k: preview}, segundos

    def __init__(self, token: int, jobs: list[tuple[int, SimConfig]], T: float) -> None:
        super().__init__()
        self.token, self.jobs, self.T = token, jobs, T
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        """Ejecuta las previsualizaciones y emite el resultado salvo que se cancele"""
        t0 = time.perf_counter()
        out = {}
        for k, cfg in self.jobs:
            try:
                r = run_preview(cfg, self.T, lambda: self._cancel)
            except Exception as exc:            # el hilo nunca debe morir en silencio
                r = {"t": np.zeros(1), "W": np.zeros((1, 3)), "dT": np.zeros(1), "dL": np.zeros(1),
                     "r": np.zeros(1), "psi": np.zeros(1), "lag": np.zeros(1), "diverged": True,
                     "error": f"{type(exc).__name__}: {exc}", "truncated": False}
            if r is None:
                return
            out[k] = r
        self.done.emit(self.token, out, time.perf_counter() - t0)


class AnalysisWorker(QtCore.QThread):
    """Calcula un análisis pesado de cada simulación fuera del hilo de la interfaz"""

    done = QtCore.Signal(str, object, float)       # tipo de análisis, {k: resultado o excepción}, segundos

    def __init__(self, kind: str, cfgs: list[SimConfig], job: Callable[[SimConfig], Any]) -> None:
        super().__init__()
        self.kind, self.cfgs, self.job = kind, cfgs, job
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        """Ejecuta el análisis y emite el resultado salvo que se cancele"""
        t0 = time.perf_counter()
        out = run_each(self.cfgs, self.job, lambda: self._cancel)
        if not self._cancel:
            self.done.emit(self.kind, out, time.perf_counter() - t0)


SIM_COLORS = ["#7FE7FF", "#FFA07A", "#C6FF7A"]
AXIS_COLORS = ["#FF6B6B", "#69F0AE", "#D1A8FF"]
BG = "#3a3d42"
PANEL = "#2f3236"
VIEW_BG = "#26282c"
FIELD = "#222427"
BORDER = "#5a5f66"
BUTTON = "#4a4e55"
ACCENT = "#3b82f6"
TEXT = "#f2f2f2"
MUTED = "#b8bcc2"

UI_FONT_CANDIDATES = ["Bahnschrift", "Segoe UI Variable Display", "Segoe UI", "Arial"]
MONO_FONT_CANDIDATES = ["Cascadia Mono", "Consolas", "Courier New"]
UI_FONT = "Segoe UI"
MONO = "Consolas, monospace"


def pick_font(candidates: Sequence[str]) -> str:
    """Primera fuente de la lista que está instalada"""
    families = set(QtGui.QFontDatabase.families())
    return next((f for f in candidates if f in families), QtGui.QFont().defaultFamily())


def axis_label_style() -> dict[str, str]:
    return {"font-family": UI_FONT, "font-size": "11pt", "color": TEXT}


def pill(text: str, color: str) -> str:
    """Etiqueta HTML con fondo de color"""
    text = text.replace(" ", "&nbsp;")           # una etiqueta no se parte en dos líneas
    return (f"<span style='background-color:{color}; color:#04123a; font-weight:700'>"
            f"&nbsp;{text}&nbsp;</span>")


def badge_css(color: str, size: int = 13) -> str:
    """Estilo CSS de una insignia con el color dado"""
    c = QtGui.QColor(color)
    return (f"QLabel {{ background: rgba({c.red()}, {c.green()}, {c.blue()}, 55); color: #ffffff;"
            f" border: 2px solid {color}; border-radius: 12px; padding: 4px 12px;"
            f" font-weight: 700; font-size: {size}px; }}")


def rgba(hex_color: str, alpha: float = 1.0, factor: float = 1.0) -> tuple[float, float, float, float]:
    """Pasa un color hexadecimal a RGBA con un factor de brillo"""
    c = QtGui.QColor(hex_color)
    return (min(c.redF() * factor, 1.0), min(c.greenF() * factor, 1.0),
            min(c.blueF() * factor, 1.0), alpha)


_BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                       [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                       [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
_BOX_EDGES = np.array([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                       (0, 4), (1, 5), (2, 6), (3, 7)]).ravel()
PANEL_COLOR = "#1f3f8f"
PANEL_LINES = (0.62, 0.72, 1.0, 0.75)
METAL_COLOR = "#cfd4de"


def box_items(center: ArrayLike, half: ArrayLike, color: str, shades: Any = (1.0, 1.0, 1.0),
              edge_color: Any = (1, 1, 1, 0.6)) -> tuple[Any, Any]:
    """Caja con caras sombreadas y aristas"""
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    V = center + V * half
    C = np.array([rgba(color, 1.0, shades[f // 4]) for f in range(12)])
    mesh = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=V, faces=_BOX_FACES, faceColors=C),
                         smooth=False, shader="shaded", glOptions="opaque")
    edges = gl.GLLinePlotItem(pos=V[_BOX_EDGES], mode="lines", color=edge_color,
                              width=1.5, antialias=True)
    return mesh, edges


def satellite_items(half: ArrayLike, color: str) -> tuple[list[Any], Array]:
    """Satélite con cuerpo, paneles solares y mástil, y su extensión"""
    h = np.asarray(half, dtype=float)
    c, b, a = np.argsort(h)

    def vec(va: float, vb: float, vc: float) -> Array:
        v = np.zeros(3)
        v[a], v[b], v[c] = va, vb, vc
        return v

    items = []
    bus = 0.2 + 0.2 * h
    items += box_items(np.zeros(3), bus, color, shades=(0.7, 1.1, 0.9))

    gap, pt = 0.12, 0.015
    Lp = max(1.25 * h[a] - bus[a] - gap, 0.45)
    pw = max(0.75 * h[b], 0.28)
    grid, struts = [], []
    for s in (-1, 1):
        inner = bus[a] + gap
        items += box_items(vec(s * (inner + Lp / 2), 0, 0), vec(Lp / 2, pw, pt),
                           PANEL_COLOR, shades=(0.9, 0.9, 1.2), edge_color=PANEL_LINES)
        for z in (-pt - 0.003, pt + 0.003):
            for i in range(1, 4):
                x = s * (inner + i * Lp / 4)
                grid += [vec(x, -pw, z), vec(x, pw, z)]
            grid += [vec(s * inner, 0, z), vec(s * (inner + Lp), 0, z)]
        struts += [vec(s * bus[a], 0, 0), vec(s * inner, 0, 0)]
    items.append(gl.GLLinePlotItem(pos=np.array(grid), mode="lines", color=PANEL_LINES,
                                   width=1, antialias=True))

    mast_top = bus[c] + 0.2
    struts += [vec(0, 0, bus[c]), vec(0, 0, mast_top)]
    items.append(gl.GLLinePlotItem(pos=np.array(struts), mode="lines",
                                   color=rgba(METAL_COLOR), width=3, antialias=True))
    md = gl.MeshData.cylinder(rows=1, cols=28, radius=[0.03, 0.26], length=0.12)
    v = md.vertexes()
    P = np.zeros_like(v)
    P[:, a], P[:, b], P[:, c] = v[:, 0], v[:, 1], v[:, 2] + mast_top - 0.04
    md.setVertexes(P)
    items.append(gl.GLMeshItem(meshdata=md, smooth=False, color=rgba(METAL_COLOR, 1.0, 0.8),
                               drawEdges=True, edgeColor=rgba(METAL_COLOR), glOptions="opaque"))

    extent = vec(bus[a] + gap + Lp, max(bus[b], pw), mast_top + 0.1)
    return items, extent

# ============================================================== skins 3d
# Cada skin se modela en un marco canónico y un _Rig lo estira sobre los ejes principales

# translúcido con descarte de caras traseras: una sola capa de vidrio, sin rayas
_GLASS = dict(GLOptions["translucent"])
_GLASS[GL_CULL_FACE] = True


def orient_out(V: Array, F: IntArray, center: ArrayLike) -> Mesh:
    """Superficies abiertas: gira las caras para que las normales miren fuera de center"""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    d = V[F].mean(1) - np.asarray(center, float)
    return (V, F[:, ::-1].copy()) if np.mean(np.einsum("ij,ij->i", n, d)) < 0 else (V, F)


def _volume(V: Array, F: IntArray) -> float:
    return np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6.0


def _closed(V: Array, F: IntArray) -> Mesh:
    """Orienta las caras hacia fuera (volumen con signo positivo)"""
    return (V, F[:, ::-1].copy()) if _volume(V, F) < 0 else (V, F)


def _grid(X: Array, wrap: bool = True) -> Mesh:
    """Triangula una rejilla de puntos"""
    nr, nc = X.shape[:2]
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc if wrap else nc - 1), indexing="ij")
    j1 = (j + 1) % nc
    a, b, c, d = i * nc + j, i * nc + j1, (i + 1) * nc + j1, (i + 1) * nc + j
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    return X.reshape(-1, 3), F


def _capped(X: Array, cap0: bool = True, cap1: bool = True) -> Mesh:
    """Malla a partir de una rejilla de anillos (nr, nc, 3), con tapas opcionales"""
    V, F = _grid(X)
    nr, nc = X.shape[:2]
    parts_v, parts_f = [V], [F]
    n = len(V)
    j = np.arange(nc)
    if cap0:
        parts_v.append(X[0].mean(0, keepdims=True))
        parts_f.append(np.stack([np.full(nc, n), (j + 1) % nc, j], -1))
        n += 1
    if cap1:
        parts_v.append(X[-1].mean(0, keepdims=True))
        parts_f.append(np.stack([np.full(nc, n), (nr - 1) * nc + j, (nr - 1) * nc + (j + 1) % nc], -1))
    return _closed(np.vstack(parts_v), np.vstack(parts_f))


def merge(*parts) -> Mesh:
    """Une varias mallas en una sola reindexando las caras"""
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def blob(center: ArrayLike, radii: ArrayLike, rows: int = 14, cols: int = 24, rot: Array | None = None,
         theta: tuple[float, float] = (0.0, math.pi)) -> Mesh:
    """Elipsoide, o casquete si theta no es completo"""
    th = np.linspace(theta[0], theta[1], rows + 1)
    ph = np.linspace(0, 2 * math.pi, cols, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    s = np.maximum(np.sin(T), 1e-3)
    X = np.stack([s * np.cos(P), s * np.sin(P), np.cos(T)], -1) * np.asarray(radii, float)
    if rot is not None:
        X = X @ np.asarray(rot, float).T
    X = X + np.asarray(center, float)
    V, F = _grid(X)
    if theta == (0.0, math.pi):
        return _closed(V, F)
    return orient_out(V, F, center)


def rot_axis(axis: ArrayLike, deg: float) -> Array:
    """Matriz de rotación de deg grados alrededor de un eje"""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def smooth_path(pts: ArrayLike, per: int = 8) -> Array:
    """Suaviza una polilínea con Catmull-Rom"""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return P
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1])
    return np.array(out)


def _interp_radii(r: ArrayLike, n: int) -> Array:
    r = np.asarray(r, float)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r)), r)


def tube(path: ArrayLike, rw: ArrayLike, rh: ArrayLike | None = None, n: int = 16,
         side: Sequence[int] | None = None, per: int = 8, caps: bool = True) -> Mesh:
    """Barrido de una elipse a lo largo de un camino, con dirección fija del semieje ancho si se da side"""
    P = smooth_path(path, per)
    m = len(P)
    rw = _interp_radii(np.atleast_1d(rw), m)
    rh = rw if rh is None else _interp_radii(np.atleast_1d(rh), m)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    A = np.zeros_like(P)
    if side is not None:
        s = np.asarray(side, float)
        for i in range(m):
            a = s - s.dot(T[i]) * T[i]
            A[i] = a / np.linalg.norm(a)
    else:
        a = np.eye(3)[np.argmin(np.abs(T[0]))]
        a = a - a.dot(T[0]) * T[0]
        A[0] = a / np.linalg.norm(a)
        for i in range(1, m):
            a = A[i - 1] - A[i - 1].dot(T[i]) * T[i]
            A[i] = a / np.linalg.norm(a)
    B = np.cross(T, A)
    ph = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s_ = np.cos(ph), np.sin(ph)
    X = (P[:, None, :] + (rw[:, None, None] * c[None, :, None]) * A[:, None, :]
         + (rh[:, None, None] * s_[None, :, None]) * B[:, None, :])
    return _capped(X, caps, caps)


def lathe(profile: ArrayLike, n: int = 32, axis: int = 2, center: ArrayLike = (0, 0, 0), caps: bool = False) -> Mesh:
    """Superficie de revolución de un perfil (t, r) con t a lo largo del eje"""
    prof = np.asarray(profile, float)
    u, v = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    ph = np.linspace(0, 2 * math.pi, n, endpoint=False)
    X = np.zeros((len(prof), n, 3))
    X[:, :, axis] = prof[:, 0:1]
    r = np.maximum(prof[:, 1:2], 1e-4)
    X[:, :, u] = r * np.cos(ph)
    X[:, :, v] = r * np.sin(ph)
    X += np.asarray(center, float)
    return _capped(X, caps, caps)


def slab(poly: Union[Array, Sequence[Sequence[float]]], lo: float, hi: float, axis: int = 0,
         center: ArrayLike = (0, 0)) -> Mesh:
    """Prisma convexo: polígono (u, v) extruido entre lo y hi a lo largo de axis"""
    poly = np.asarray(poly, float) + np.asarray(center, float)
    u, v = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    k = len(poly)

    def pts(z: float) -> Array:
        Q = np.zeros((k, 3))
        Q[:, axis], Q[:, u], Q[:, v] = z, poly[:, 0], poly[:, 1]
        return Q

    V = np.vstack([pts(lo), pts(hi), pts(lo).mean(0, keepdims=True), pts(hi).mean(0, keepdims=True)])
    j = np.arange(k)
    j1 = (j + 1) % k
    F = [np.stack([j, j1, k + j1], -1), np.stack([j, k + j1, k + j], -1),
         np.stack([np.full(k, 2 * k), j1, j], -1), np.stack([np.full(k, 2 * k + 1), k + j, k + j1], -1)]
    return _closed(V, np.vstack(F))


def ngon(n: int, R: float, phase_deg: float = 90.0) -> Array:
    """Vértices de un polígono regular de n lados y radio R"""
    a = np.radians(phase_deg + 360.0 / n * np.arange(n))
    return np.column_stack([R * np.cos(a), R * np.sin(a)])


def smoothstep(e0: float, e1: float, x: Any) -> Any:
    """Transición suave de 0 a 1 entre e0 y e1"""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class _Rig:
    """Lleva el modelo canónico a los ejes principales; las piezas se estiran y los detalles conservan su forma"""
    SIZE = 1.25

    def __init__(self, half: ArrayLike, roles: Sequence[int], ref: Sequence[float], floor: float) -> None:
        h = np.asarray(half, float)
        order = np.argsort(h)
        rank_axis = [order[2], order[1], order[0]]
        hh = np.maximum(h / h.max(), floor)
        self.axis = [rank_axis[r] for r in roles]
        self.s = np.array([self.SIZE * hh[self.axis[i]] / ref[i] for i in range(3)])
        self.k = float(np.prod(self.s) ** (1 / 3))
        self.kmin = float(self.s.min())
        P = np.zeros((3, 3))
        for i in range(3):
            P[self.axis[i], i] = 1.0
        self.P = P
        self.flip = np.linalg.det(P) < 0
        ext = np.zeros(3)
        for i in range(3):
            ext[self.axis[i]] = self.SIZE * hh[self.axis[i]]
        self.extent = ext
        self._groups, self._lines = {}, {}

    def _xf(self, V: Array, local: ArrayLike | None = None, stretch: Sequence[int] = (),
            k: Union[int, str, None] = None) -> Array:
        """Pasa vértices del marco canónico a los ejes del cuerpo, con escala local opcional"""
        V = np.asarray(V, float)
        if local is None:
            return (V * self.s) @ self.P.T
        c = np.asarray(local, float)
        kk = self.k if k is None else (self.kmin if k == "min" else k)
        sv = np.array([self.s[i] if i in stretch else kk for i in range(3)])
        return ((c * self.s) + (V - c) * sv) @ self.P.T

    # acumulación por material
    def mesh(self, vf: tuple[Array, IntArray], color: str = "#ffffff", alpha: float = 1.0, shade: float = 1.0,
             vc: Array | None = None, smooth: bool = True, emissive: bool = False, glass: bool = False,
             local: ArrayLike | None = None, stretch: Sequence[int] = (),
             k: Union[int, str, None] = None) -> None:
        """Acumula una malla en el grupo de su material"""
        V, F = vf
        Vb = self._xf(V, local, stretch, k)
        if self.flip:
            F = F[:, ::-1]
        if vc is None:
            c = QtGui.QColor(color)
            vc = np.tile([c.redF() * shade, c.greenF() * shade, c.blueF() * shade, alpha], (len(Vb), 1))
        key = ("glass", emissive) if glass else (("emis",) if emissive else (("smooth",) if smooth else ("flat",)))
        self._groups.setdefault(key, []).append((Vb, F, np.asarray(vc, float)))

    def line(self, pts: Array, color: str = "#ffffff", alpha: float = 1.0, width: float = 1.5, mode: str = "line_strip",
             local: ArrayLike | None = None, stretch: Sequence[int] = (), k: Union[int, str, None] = None) -> None:
        """Acumula una polilínea o segmentos con su color"""
        P = self._xf(pts, local, stretch, k)
        if mode == "line_strip":
            P = np.repeat(P, 2, axis=0)[1:-1]
        c = QtGui.QColor(color)
        self._lines.setdefault(float(width), []).append(
            (P, np.tile([c.redF(), c.greenF(), c.blueF(), alpha], (len(P), 1))))

    def finish(self) -> list[Any]:
        """Fusiona la geometría acumulada: opacos, líneas, emisivos y vidrio, en ese orden"""
        items = [self._merged(key) for key in (("smooth",), ("flat",)) if key in self._groups]
        for width, segs in self._lines.items():
            items.append(gl.GLLinePlotItem(pos=np.vstack([p for p, _ in segs]), mode="lines",
                                           color=np.vstack([c for _, c in segs]), width=width,
                                           antialias=True))
        for key in (("emis",), ("glass", False), ("glass", True)):
            if key in self._groups:
                items.append(self._merged(key))
        return items

    def _merged(self, key: Any) -> gl.GLMeshItem:
        """Funde en un solo GLMeshItem las mallas de un grupo"""
        Vs, Fs, Cs, off = [], [], [], 0
        for V, F, C in self._groups[key]:
            Vs.append(V)
            Fs.append(F + off)
            Cs.append(C)
            off += len(V)
        md = gl.MeshData(vertexes=np.vstack(Vs), faces=np.ascontiguousarray(np.vstack(Fs)),
                         vertexColors=np.vstack(Cs))
        kind = key[0]
        return gl.GLMeshItem(meshdata=md, smooth=(kind != "flat"), color=(1, 1, 1, 1),
                             shader=None if (kind == "emis" or (kind == "glass" and key[1])) else "shaded",
                             glOptions=_GLASS if kind == "glass" else "opaque")


# ============================================================== vaca Holstein
_COW_WHITE = np.array([0.96, 0.95, 0.92])
_COW_BLACK = np.array([0.07, 0.07, 0.08])
# (centro, semiejes) de las manchas, en el marco canónico de la vaca
_SPOTS = [
    ((0.20, 0.22, 0.30), (0.20, 0.13, 0.17)),
    ((-0.30, 0.24, 0.14), (0.24, 0.13, 0.22)),
    ((-0.08, -0.24, 0.30), (0.26, 0.13, 0.15)),
    ((0.30, -0.25, 0.04), (0.15, 0.11, 0.17)),
    ((-0.55, -0.12, 0.36), (0.15, 0.15, 0.13)),
    ((0.05, 0.02, 0.52), (0.16, 0.13, 0.08)),
    ((0.74, 0.17, 0.30), (0.11, 0.09, 0.11)),     # parche en la cabeza
    ((0.62, -0.10, 0.34), (0.13, 0.12, 0.10)),    # cuello
    ((-0.62, 0.13, 0.00), (0.10, 0.10, 0.16)),    # muslo
]


def _coat(V: Array) -> Array:
    """Color del pelaje por vértice: manchas negras sobre fondo blanco"""
    f = np.full(len(V), 9.0)
    for c, r in _SPOTS:
        d = np.linalg.norm((V - np.array(c)) / np.array(r), axis=1)
        f = np.minimum(f, d)
    wob = 0.13 * np.sin(9.0 * V[:, 0] + 4.0 * V[:, 1]) * np.sin(7.0 * V[:, 2] + 3.0 * V[:, 1] + 1.0)
    m = 1.0 - smoothstep(0.90, 1.06, f + wob)
    rgb = _COW_WHITE[None, :] * (1 - m[:, None]) + _COW_BLACK[None, :] * m[:, None]
    return np.column_stack([rgb * 2.0, np.ones(len(V))])


def cow_items(half: ArrayLike, sim_color: str) -> tuple[list[Any], Array]:
    """Vaca: el cuerpo largo sigue al eje de menor inercia; el collar lleva el color de la simulación"""
    rig = _Rig(half, roles=(0, 2, 1), ref=(1.0, 0.30, 0.78), floor=0.24)
    ALL = (0, 1, 2)

    def coat(vf: tuple[Array, IntArray], **kw) -> None:
        rig.mesh(vf, vc=_coat(vf[0]), **kw)

    # tronco: tubo elíptico con perfil de barril
    t = np.cos(np.linspace(math.pi, 0, 45))
    e = np.maximum(np.maximum(1 - np.abs(t) ** 2.6, 0) ** (1 / 2.6), 0.004)
    xs = -0.07 + 0.63 * t
    path = np.column_stack([xs, np.zeros_like(xs), 0.14 - 0.03 * t])
    coat(tube(path, rw=0.285 * e * (1 + 0.05 * t), rh=0.36 * e, n=44, side=(0, 1, 0), per=1,
              caps=False), stretch=ALL)

    # cuello y cabeza
    coat(tube([(0.34, 0, 0.23), (0.55, 0, 0.255), (0.71, 0, 0.23)], rw=[0.23, 0.2, 0.17],
              rh=[0.21, 0.2, 0.18], n=28, side=(0, 1, 0), per=6), stretch=ALL)
    coat(tube([(0.68, 0, 0.27), (0.79, 0, 0.19), (0.90, 0, -0.01)], rw=[0.165, 0.15, 0.11],
              rh=[0.15, 0.135, 0.095], n=28, side=(0, 1, 0), per=6), stretch=ALL)
    mz = (0.915, 0, -0.065)
    rig.mesh(blob(mz, (0.07, 0.115, 0.08), 12, 22), "#f4aab5", shade=1.5, local=mz, stretch=ALL)
    for s in (-1, 1):
        nc = (0.982, s * 0.045, -0.045)
        rig.mesh(blob(nc, (0.014, 0.02, 0.016), 8, 12), "#3a1e26", local=nc)
        ec = (0.775, s * 0.15, 0.245)                                    # ojo con brillo
        rig.mesh(blob(ec, (0.026, 0.024, 0.03), 10, 16), "#0c0c0e", local=ec)
        gc = (0.79, s * 0.165, 0.258)
        rig.mesh(blob(gc, (0.008, 0.008, 0.008), 6, 8), "#ffffff", emissive=True, local=gc)
        oc = (0.665, s * 0.19, 0.285)                                    # oreja caída
        R = rot_axis((1, 0, 0), -s * 28)
        rig.mesh(blob(oc, (0.035, 0.115, 0.05), 10, 18, rot=R), "#25252a", shade=1.5, local=oc)
        ic = (0.675, s * 0.185, 0.28)
        rig.mesh(blob(ic, (0.02, 0.08, 0.03), 8, 14, rot=R), "#f1a6b2", shade=1.5, local=ic)
        hb = (0.70, s * 0.10, 0.37)                                      # cuerno curvo
        horn = tube([hb, (0.70, s * 0.19, 0.43), (0.71, s * 0.26, 0.53), (0.74, s * 0.25, 0.63)],
                    rw=[0.042, 0.034, 0.022, 0.004], n=12, per=6)
        rig.mesh(horn, "#efe3bf", shade=1.6, local=hb)

    # patas (blancas, con corvejón en las traseras) y pezuñas
    legs = {
        "front": ([(0.34, 0, 0.05), (0.35, 0, -0.25), (0.355, 0, -0.48), (0.36, 0, -0.66)],
                  [0.115, 0.085, 0.055, 0.05], 0.34),
        "rear": ([(-0.46, 0, 0.05), (-0.50, 0, -0.22), (-0.56, 0, -0.42), (-0.50, 0, -0.58),
                  (-0.49, 0, -0.66)], [0.15, 0.11, 0.06, 0.05, 0.05], -0.49),
    }
    for kind, (pts, rad, hx) in legs.items():
        for s in (-1, 1):
            P = [(x, s * 0.15, z) for x, _, z in pts]
            coat(tube(P, rw=rad, n=16, per=6), stretch=ALL)
            hc = (hx + 0.012, s * 0.15, -0.735)
            rig.mesh(blob(hc, (0.062, 0.052, 0.04), 10, 16), "#16161a", local=hc)

    # ubre con pezones
    uc = (-0.30, 0, -0.22)
    rig.mesh(blob(uc, (0.15, 0.11, 0.09), 12, 20), "#f2a9b4", shade=1.5, local=uc, stretch=(0, 1))
    for dx in (-0.055, 0.055):
        for dy in (-0.04, 0.04):
            tc = (-0.30 + dx, dy, -0.29)
            rig.mesh(tube([tc, (tc[0], tc[1], tc[2] - 0.055)], rw=[0.024, 0.017], n=10, per=1),
                     "#e58b9c", shade=1.5, local=tc)

    # cola con borla negra
    rig.mesh(tube([(-0.66, 0, 0.30), (-0.77, 0, 0.25), (-0.80, 0, 0.05), (-0.77, 0, -0.22),
                   (-0.76, 0, -0.42)], rw=[0.036, 0.03, 0.025, 0.02, 0.018], n=10, per=6),
             "#f1eee6", shade=1.8, stretch=ALL)
    tc = (-0.76, 0, -0.50)
    rig.mesh(blob(tc, (0.04, 0.04, 0.095), 10, 14), "#16161a", local=tc)

    # collar con el color de la simulación y cencerro dorado
    ph = np.linspace(0, 2 * math.pi, 40)
    ring = np.column_stack([np.full_like(ph, 0.555), 0.215 * np.cos(ph), 0.255 + 0.235 * np.sin(ph)])
    rig.mesh(tube(ring, rw=0.03, n=10, per=1, caps=False), sim_color, shade=1.5, stretch=ALL)
    bc = (0.555, 0, 0.03)
    bell = lathe([(0.045, 0.022), (0.03, 0.045), (0.0, 0.058), (-0.035, 0.07), (-0.05, 0.072)],
                 n=20, axis=2, center=bc, caps=True)
    rig.mesh(bell, "#f0bc3a", shade=1.6, local=bc, stretch=(0, 1))
    cl = (0.555, 0, -0.02)
    rig.mesh(blob(cl, (0.016, 0.016, 0.016), 6, 10), "#5a4210", local=cl)

    return rig.finish(), rig.extent


# ============================================================== ovni
_METAL_HI = np.array([0.74, 0.78, 0.85])
_METAL_LO = np.array([0.40, 0.44, 0.52])


def _hull_profile() -> Array:
    """Perfil (z, radio) del casco lenticular"""
    pts = [(-0.205, 0.0), (-0.205, 0.22), (-0.19, 0.45), (-0.145, 0.68), (-0.075, 0.88),
           (-0.015, 0.985), (0.02, 1.0), (0.055, 0.965), (0.095, 0.86), (0.135, 0.66),
           (0.17, 0.46), (0.19, 0.38), (0.19, 0.0)]
    P = smooth_path([(z, r, 0.0) for z, r in pts], per=6)
    return P[:, :2]


def _hull_colors(V: Array, rim_rgb: Array) -> Array:
    """Colores del casco con surcos y una banda del color de la simulación"""
    rho = np.hypot(V[:, 0], V[:, 1])
    z = V[:, 2]
    base = np.where((z > 0.02)[:, None], _METAL_HI[None, :], _METAL_LO[None, :])
    groove = np.zeros(len(V))                                  # surcos concéntricos
    for r0 in (0.48, 0.74):
        groove = np.maximum(groove, 1 - smoothstep(0.004, 0.02, np.abs(rho - r0)))
    base = base * (1 - 0.42 * groove[:, None])
    band = (1 - smoothstep(0.03, 0.055, np.abs(z - 0.018))) * smoothstep(0.88, 0.95, rho)
    rgb = base * (1 - band[:, None]) + rim_rgb[None, :] * band[:, None]   # banda del color de la sim
    return np.column_stack([rgb * 2.1, np.ones(len(V))])


def _mini_cow(rig: _Rig, center: ArrayLike, tilt: float = -22, spin: float = 35) -> None:
    """Vaquita abducida flotando en el haz"""
    R = rot_axis((0, 1, 0), tilt) @ rot_axis((0, 0, 1), spin)
    ctr = np.asarray(center, float)

    def part(off: ArrayLike, radii: ArrayLike, color: str) -> None:
        c = ctr + R @ np.asarray(off)
        rig.mesh(blob(c, radii, 8, 12, rot=R), color, shade=2.0, local=center)

    def limb(a: ArrayLike, b: ArrayLike, r: Union[float, Sequence[float]], color: str = "#f4f1ea") -> None:
        rig.mesh(tube([ctr + R @ np.array(a), ctr + R @ np.array(b)], rw=r, n=8, per=1),
                 color, shade=2.0, local=center)

    part((0, 0, 0), (0.095, 0.05, 0.055), "#f4f1ea")
    part((-0.02, 0.022, 0.012), (0.04, 0.035, 0.04), "#18181c")
    part((0.12, 0, 0.02), (0.042, 0.03, 0.035), "#f4f1ea")
    part((0.158, 0, 0.012), (0.018, 0.026, 0.02), "#f4aab5")
    for s in (-1, 1):
        part((0.115, s * 0.03, 0.055), (0.012, 0.012, 0.025), "#efe3bf")
        for dx in (-0.06, 0.06):
            limb((dx, s * 0.028, -0.03), (dx, s * 0.03, -0.1), [0.016, 0.012])
    limb((-0.095, 0, 0.01), (-0.14, 0, -0.06), [0.01, 0.008])


def ufo_items(half: ArrayLike, sim_color: str) -> tuple[list[Any], Array]:
    """Ovni con alienígena, cúpula, haz tractor y vaca abducida"""
    rig = _Rig(half, roles=(0, 1, 2), ref=(1.0, 1.0, 0.58), floor=0.30)
    ALL = (0, 1, 2)
    qc = QtGui.QColor(sim_color)
    sim_rgb = np.array([qc.redF(), qc.greenF(), qc.blueF()])

    # casco lenticular
    V, F = lathe(_hull_profile(), n=96, axis=2)
    rig.mesh((V, F), vc=_hull_colors(V, sim_rgb), stretch=ALL)

    # luces del borde (alternan color de la simulación / ámbar)
    for i in range(24):
        a = 2 * math.pi * i / 24
        c = (0.985 * math.cos(a), 0.985 * math.sin(a), 0.022)
        rig.mesh(blob(c, (0.036, 0.036, 0.036), 6, 10), sim_color if i % 2 == 0 else "#fff0a8",
                 shade=1.3, emissive=True, local=c)
    for i in range(10):                                        # luces de la panza
        a = 2 * math.pi * (i + 0.5) / 10
        c = (0.56 * math.cos(a), 0.56 * math.sin(a), -0.195)
        rig.mesh(blob(c, (0.03, 0.03, 0.016), 6, 10), "#bff6ff", emissive=True, local=c)

    # emisor del haz
    ph = np.linspace(0, 2 * math.pi, 48)
    ring = np.column_stack([0.19 * np.cos(ph), 0.19 * np.sin(ph), np.full_like(ph, -0.205)])
    rig.mesh(tube(ring, rw=0.04, n=10, per=1, caps=False), "#30384a", shade=2.0, stretch=ALL)
    e = (0, 0, -0.225)
    rig.mesh(blob(e, (0.17, 0.17, 0.035), 8, 24), "#d8fbff", emissive=True, local=e, stretch=(0, 1))

    # tren de aterrizaje: tres patas con patín
    for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3):
        ca, sa = math.cos(a), math.sin(a)
        P = [(0.62 * ca, 0.62 * sa, -0.14), (0.73 * ca, 0.73 * sa, -0.32), (0.77 * ca, 0.77 * sa, -0.42)]
        rig.mesh(tube(P, rw=[0.036, 0.026, 0.022], n=10, per=4), "#6a7283", shade=2.0, stretch=ALL)
        fc = (0.78 * ca, 0.78 * sa, -0.455)
        foot = lathe([(0.02, 0.0), (0.02, 0.06), (0.0, 0.095), (-0.02, 0.095), (-0.02, 0.0)],
                     n=18, axis=2, center=fc)
        rig.mesh(foot, "#2d3340", shade=1.6, local=fc)

    # alienígena en la cabina (escala local isótropa: no se deforma con la inercia)
    AC = (0.0, 0.0, 0.34)
    A = dict(local=AC, k="min")

    def al(vf: tuple[Array, IntArray], color: str, **kw) -> None:
        kw.setdefault("shade", 2.0)
        rig.mesh(vf, color, **A, **kw)

    al(blob((0, 0, -0.06 + 0.34), (0.075, 0.07, 0.09), 10, 16), "#6fcf52")
    al(blob((0, 0, 0.055 + 0.34), (0.108, 0.098, 0.122), 14, 22), "#86e060")
    for s in (-1, 1):
        R = rot_axis((1, 0, 0), -s * 28)
        ec = (0.088, s * 0.052, 0.068 + 0.34 - 0.04)
        al(blob(ec, (0.02, 0.034, 0.06), 10, 14, rot=R), "#08090c", shade=1.0)
        gc = (0.104, s * 0.046, 0.082 + 0.34 - 0.04)
        al(blob(gc, (0.008, 0.008, 0.008), 6, 8), "#ffffff", emissive=True)
        base = (0.0, s * 0.045, 0.34 + 0.165)                      # antena con bulbo luminoso
        top = (0.0, s * 0.085, 0.34 + 0.255)
        al(tube([base, (0.0, s * 0.06, 0.34 + 0.21), top], rw=[0.009, 0.007], n=8, per=4), "#6fcf52")
        al(blob(top, (0.022, 0.022, 0.022), 8, 12), sim_color, emissive=True, shade=1.3)
        al(tube([(0.01, s * 0.07, 0.31), (0.075, s * 0.09, 0.285), (0.15, s * 0.07, 0.265)],
                rw=[0.017, 0.014, 0.013], n=8, per=4), "#6fcf52")   # brazo hacia la consola
    ys = np.linspace(-0.04, 0.04, 9)                               # sonrisa
    zs = 0.34 + 0.012 - 0.018 * (1 - (ys / 0.04) ** 2)
    xs = 0.098 * np.sqrt(np.maximum(1 - (ys / 0.098) ** 2 - ((zs - 0.395) / 0.122) ** 2, 0.01)) + 0.003
    rig.line(np.column_stack([xs, ys, zs]), "#14361a", width=2, **A)
    cb = slab([(-0.1, -0.035), (0.1, -0.035), (0.1, 0.035), (-0.1, 0.035)], 0.15, 0.21, axis=0,
              center=(0, 0.255))
    al(cb, "#2a2f3b", shade=1.6)                                   # consola de mandos
    for j, col in enumerate(("#ff4d4d", "#5dff8a", "#ffe45c")):
        al(blob((0.2125, (j - 1) * 0.05, 0.268), (0.008, 0.012, 0.012), 6, 8), col, emissive=True)

    # vaquita abducida
    _mini_cow(rig, (0.04, 0.0, -0.66))

    # translúcidos (se dibujan al final): haz tractor y cúpula
    beam = orient_out(*lathe([(-0.235, 0.15), (-0.52, 0.29), (-0.86, 0.45)], n=48, axis=2),
                      (0, 0, -0.55))
    rig.mesh(beam, "#9ff3ff", alpha=0.16, glass=True, emissive=True)
    dome = blob((0, 0, 0.19), (0.405, 0.405, 0.44), 20, 40, theta=(0.0, math.pi / 2))
    rig.mesh(dome, sim_color, alpha=0.42, glass=True, shade=1.3)
    dring = np.column_stack([0.41 * np.cos(ph), 0.41 * np.sin(ph), np.full_like(ph, 0.19)])
    rig.mesh(tube(dring, rw=0.026, n=10, per=1, caps=False), "#e0b84e", shade=1.6, stretch=ALL)

    return rig.finish(), rig.extent


# ============================================================== caza TIE
_CELL_BLUE = np.array([0.10, 0.17, 0.38])
_CELL_SHADE = (1.0, 0.78, 1.22, 0.9, 1.12, 0.84)


def _rect(p0: Array, p1: Array, w: ArrayLike) -> Array:
    """Rectángulo de ancho w alrededor del segmento de p0 a p1"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * w / 2
    return np.array([p0 - n, p1 - n, p1 + n, p0 + n])


def _tie_panel(rig: _Rig, sgn: int, accent: str) -> None:
    """Ala hexagonal a un lado de la cabina, en el plano y z del marco canónico"""
    xc = sgn * 0.66
    dt = 0.032 / rig.s[0]
    Ro, Ri = 1.0, 0.905
    out, inn = ngon(6, Ro, 90), ngon(6, Ri, 90)
    ALL = (0, 1, 2)

    frame = merge(*[slab([out[k], out[(k + 1) % 6], inn[(k + 1) % 6], inn[k]],
                         xc - dt, xc + dt, axis=0) for k in range(6)])
    rig.mesh(frame, "#4b5260", smooth=False, shade=2.0, stretch=ALL)

    parts, cols = [], []                                       # seis cuñas de celdas, cada una con su tono
    for k in range(6):
        V, F = slab([(0, 0), inn[k] * 0.985, inn[(k + 1) % 6] * 0.985], xc - dt * 0.55, xc + dt * 0.55, axis=0)
        parts.append((V, F))
        cols.append(np.tile(np.append(_CELL_BLUE * _CELL_SHADE[k] * 2.3, 1.0), (len(V), 1)))
    rig.mesh(merge(*parts), vc=np.vstack(cols), smooth=False, stretch=ALL)

    ribs = merge(*[slab(_rect(ngon(6, 0.1, 90)[k], ngon(6, Ro * 0.99, 90)[k], 0.045),
                        xc - dt * 1.35, xc + dt * 1.35, axis=0) for k in range(6)])
    rig.mesh(ribs, "#6d7586", smooth=False, shade=2.1, stretch=ALL)
    rig.mesh(slab(ngon(6, 0.17, 90), xc - dt * 2.2, xc + dt * 2.2, axis=0), "#59616f",
             smooth=False, shade=2.0, stretch=ALL)
    rig.mesh(slab(ngon(6, 0.105, 90), xc - dt * 2.7, xc + dt * 2.7, axis=0), accent,
             smooth=False, shade=1.3, stretch=ALL)

    for face in (-1, 1):                                       # contorno del color de la sim y malla
        x = xc + face * (dt * 1.02)
        hexo = np.vstack([out, out[:1]])
        rig.line(np.column_stack([np.full(7, x), hexo]), accent, width=3)
        for R in (0.36, 0.64):
            h = ngon(6, R, 90)
            h = np.vstack([h, h[:1]])
            rig.line(np.column_stack([np.full(7, x - face * dt * 0.45), h]), "#8aa0d6",
                     alpha=0.75, width=1)


def fighter_items(half: ArrayLike, sim_color: str) -> tuple[list[Any], Array]:
    """Caza TIE con cabina esférica, pilones y alas hexagonales"""
    rig = _Rig(half, roles=(1, 2, 0), ref=(0.74, 0.87, 1.0), floor=0.42)
    ALL = (0, 1, 2)
    k = rig.k
    O = (0.0, 0.0, 0.0)
    K = dict(local=O)

    # cabina esférica con escotillas
    Rb = 0.31
    rig.mesh(blob(O, (Rb, Rb, Rb), 22, 40), "#b4bccb", shade=2.0, **K)
    rig.mesh(blob((0, 0, Rb * 0.97), (0.095, 0.095, 0.022), 8, 20), "#4d5563", shade=2.0, **K)
    rig.mesh(blob((0, -Rb * 0.93, 0), (0.12, 0.03, 0.12), 8, 20), "#4d5563", shade=2.0, **K)
    rig.mesh(blob((0, -Rb, 0), (0.045, 0.02, 0.045), 8, 14), "#ff6a3d", emissive=True, **K)

    # ventanal frontal octogonal con ocho radios
    yw = Rb * 0.80
    rig.mesh(slab(ngon(8, 0.150, 22.5), yw - 0.012, yw + 0.012, axis=1), "#0c1320", smooth=False, **K)
    out, inn = ngon(8, 0.185, 22.5), ngon(8, 0.145, 22.5)
    rig.mesh(merge(*[slab([out[i], out[(i + 1) % 8], inn[(i + 1) % 8], inn[i]], yw - 0.035, yw + 0.03, axis=1)
                     for i in range(8)]), "#6b7484", smooth=False, shade=2.1, **K)
    yl = yw + 0.0135
    a8 = np.radians(22.5 + 45 * np.arange(8))

    def wpt(R: float, ang: float) -> tuple[float, float, float]:
        return (R * math.sin(ang), yl, R * math.cos(ang))

    spokes = []
    for a in a8:
        spokes += [wpt(0.0, a), wpt(0.15, a)]
    rig.line(np.array(spokes), "#c5ccd8", width=2, mode="lines", **K)
    for R in (0.055, 0.1):
        rig.line(np.array([wpt(R, a) for a in list(a8) + [a8[0]]]), "#c5ccd8", width=1.5, **K)

    # cañones láser bajo el ventanal
    for s in (-1, 1):
        P = [(s * 0.065, 0.2, -0.12), (s * 0.065, 0.33, -0.13), (s * 0.065, 0.47, -0.135)]
        rig.mesh(tube(P, rw=[0.026, 0.02, 0.016], n=10, per=4), "#3a404c", shade=2.0, **K)
        rig.mesh(blob((s * 0.065, 0.485, -0.135), (0.026, 0.03, 0.026), 8, 12), "#5dff7a",
                 emissive=True, **K)

    # collares octogonales y pilones
    x_col = 0.33 * k / rig.s[0]
    for s in (-1, 1):
        collar = lathe([(0.0, 0.0), (0.0, 0.2), (s * 0.075, 0.2), (s * 0.075, 0.0)], n=8, axis=0,
                       center=(s * 0.22, 0, 0))
        rig.mesh(collar, "#59616f", smooth=False, shade=2.0, **K)
        P = [(s * x_col * 0.9, 0, 0), (s * (x_col + 0.66 - x_col) * 0.5, 0, 0), (s * 0.63, 0, 0)]
        r = 0.095 * k
        pyl = tube(P, rw=[r / rig.s[1], 0.8 * r / rig.s[1], 0.65 * r / rig.s[1]],
                   rh=[r / rig.s[2], 0.8 * r / rig.s[2], 0.65 * r / rig.s[2]], n=14, side=(0, 1, 0), per=4)
        rig.mesh(pyl, "#7a8392", shade=2.0, stretch=ALL)

    for s in (-1, 1):
        _tie_panel(rig, s, sim_color)

    return rig.finish(), rig.extent


class FloatSlider(QtWidgets.QWidget):
    """Deslizador con caja numérica sincronizada"""

    valueChanged = QtCore.Signal(float)

    def __init__(self, label: str, vmin: float, vmax: float, step: float, value: float, decimals: int = 3,
                 parent: QtWidgets.QWidget | None = None,
                 spin_range: tuple[float, float] | None = None) -> None:
        super().__init__(parent)
        self.vmin, self.vmax, self.step = vmin, vmax, step
        lo, hi = spin_range if spin_range else (vmin, vmax)
        self._lo, self._hi = lo, hi
        self._n = int(round((vmax - vmin) / step))

        self.label = QtWidgets.QLabel(label)
        self.label.setMinimumWidth(34)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._n)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(lo, hi)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(78)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.setValue(value, silent=True)

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float, silent: bool = False) -> None:
        """Fija el valor dentro del rango y emite la señal salvo que silent sea verdadero"""
        v = min(max(float(v), self._lo), self._hi)
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(self._pos(v))
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        if not silent:
            self.valueChanged.emit(v)

    def _from_slider(self, i: int) -> None:
        v = self.vmin + i * self.step
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _pos(self, v: float) -> int:
        return min(max(int(round((v - self.vmin) / self.step)), 0), self._n)

    def _from_spin(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._pos(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


def _group_box(title: str, widgets: Sequence[QtWidgets.QWidget]) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    v = QtWidgets.QVBoxLayout(box)
    v.setSpacing(6)
    for w in widgets:
        v.addWidget(w)
    return box


class PhysicsSliders(QtWidgets.QWidget):
    """Momentos de inercia y velocidad angular inicial, que se puede dar por eje más perturbaciones o por componentes"""

    changed = QtCore.Signal()
    AXIS_NAMES = ("e₁", "e₂", "e₃")

    def __init__(self, I: ArrayLike = (1.0, 2.0, 3.0), w0: ArrayLike = (0.0, 0.0, 0.0),
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        def heading(text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            lbl.setObjectName("muted")
            return lbl

        inertia_head = heading("Momentos principales de inercia")
        inertia_head.setToolTip("En kg·m² (solo importan las razones entre ellos)")
        lay.addWidget(inertia_head)
        self.I_sliders = [FloatSlider(f"I{k + 1}", 0.1, 6.0, 0.01, 1.0, 2) for k in range(3)]
        for s in self.I_sliders:
            lay.addWidget(s)
        self.tri_warning = QtWidgets.QLabel("")
        self.tri_warning.setObjectName("warning")
        self.tri_warning.setWordWrap(True)
        self.tri_warning.hide()
        lay.addWidget(self.tri_warning)
        lay.addSpacing(4)

        head = QtWidgets.QHBoxLayout()
        w_head = heading("Velocidad angular Ω₀")
        w_head.setToolTip("En rad/s")
        head.addWidget(w_head, 1)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Eje + perturb.", "Componentes"])
        self.mode.setToolTip("Dos formas de dar el mismo Ω₀. Al cambiar de una a otra el vector no cambia.")
        head.addWidget(self.mode)
        lay.addLayout(head)

        # --- modo componentes
        self.w_sliders = [FloatSlider(f"ω{k + 1}", -3.0, 3.0, 0.005, 0.0, 3) for k in range(3)]
        self.comp_page = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(self.comp_page)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        for s in self.w_sliders:
            cl.addWidget(s)

        # --- modo eje + perturbaciones
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.setToolTip("Eje principal de la rotación inicial")
        self.w_main = FloatSlider("Ω", -3.0, 3.0, 0.005, 0.0, 3)
        self.p1_axis = QtWidgets.QComboBox()
        self.p1_axis.setToolTip("Eje sobre el que actúa la primera perturbación")
        self.p1 = FloatSlider("ε₁", -0.5, 0.5, 0.001, 0.0, 3, spin_range=(-3.0, 3.0))
        self.chk_p2 = QtWidgets.QCheckBox("Segunda perturbación")
        self.chk_p2.setToolTip("Añade una componente sobre el tercer eje (la primera actúa sobre otro distinto)")
        self.p2 = FloatSlider("ε₂", -0.5, 0.5, 0.001, 0.0, 3, spin_range=(-3.0, 3.0))
        self.readout = QtWidgets.QLabel()
        self.readout.setObjectName("muted")
        self.axis_page = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(self.axis_page)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(QtWidgets.QLabel("Eje principal"))
        r1.addWidget(self.axis_combo, 1)
        al.addLayout(r1)
        al.addWidget(self.w_main)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("Perturbar"))
        r2.addWidget(self.p1_axis, 1)
        al.addLayout(r2)
        al.addWidget(self.p1)
        al.addWidget(self.chk_p2)
        al.addWidget(self.p2)
        al.addWidget(self.readout)
        lay.addWidget(self.axis_page)
        lay.addWidget(self.comp_page)

        for c in (self.mode, self.axis_combo, self.p1_axis):
            c.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(8)
        for s in self.I_sliders + self.w_sliders + [self.w_main, self.p1, self.p2]:
            s.label.setMinimumWidth(74)         # columnas alineadas aunque la etiqueta cambie

        self._axes = (1, 0, 2)                  # (principal, perturbación 1, perturbación 2)
        self._busy = False
        self._tags = ["", "", ""]
        self.set(I, w0)

        for k, s in enumerate(self.I_sliders):
            s.valueChanged.connect(lambda v, k=k: self._on_inertia(k))
        for s in self.w_sliders:
            s.valueChanged.connect(self._on_components)
        for s in (self.w_main, self.p1, self.p2):
            s.valueChanged.connect(self._on_axis_values)
        self.chk_p2.toggled.connect(self._on_axis_values)
        self.axis_combo.currentIndexChanged.connect(self._on_main_axis)
        self.p1_axis.currentIndexChanged.connect(self._on_p1_axis)
        self.mode.currentIndexChanged.connect(self._on_mode)

    # ---- vector inicial y su descomposición en eje y perturbaciones
    def _vector_from_axis(self) -> Array:
        return omega_from_axis(self._axes, self.w_main.value(), self.p1.value(),
                               self.p2.value() if self.chk_p2.isChecked() else 0.0)

    def _fill_axis_widgets(self, w: ArrayLike) -> None:
        """Rellena el modo eje con el vector w sin emitir señales"""
        self._busy = True
        self._axes = axis_decomposition(w, self._axes)
        a, b, c = self._axes
        self.w_main.setValue(w[a], silent=True)
        self.p1.setValue(w[b], silent=True)
        self.p2.setValue(w[c], silent=True)
        self.chk_p2.setChecked(abs(w[c]) > 0)
        self._rebuild_axis_combos()
        self._busy = False
        self._update_readout()

    def _axis_text(self, k: int) -> str:
        return f"{self.AXIS_NAMES[k]} · {self._tags[k]}" if self._tags[k] else self.AXIS_NAMES[k]

    def _rebuild_axis_combos(self) -> None:
        a, b, c = self._axes
        was = self._busy
        self._busy = True
        self.axis_combo.clear()
        for k in range(3):
            self.axis_combo.addItem(self._axis_text(k), k)
        self.axis_combo.setCurrentIndex(a)
        self.p1_axis.clear()
        for k in (j for j in range(3) if j != a):
            self.p1_axis.addItem(self._axis_text(k), k)
        self.p1_axis.setCurrentIndex(self.p1_axis.findData(b))
        self.p1.label.setText(f"ε₁ ({self.AXIS_NAMES[b]})")
        self.p2.label.setText(f"ε₂ ({self.AXIS_NAMES[c]})")
        self.w_main.label.setText(f"Ω ({self.AXIS_NAMES[a]})")
        self.p2.setEnabled(self.chk_p2.isChecked())
        self._busy = was

    def _update_readout(self) -> None:
        w = self._vector_from_axis()
        self.readout.setText(f"Ω₀ = ({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})")

    def _sync_components(self, w: ArrayLike) -> None:
        for s, v in zip(self.w_sliders, w):
            s.setValue(v, silent=True)

    # ---- reacciones a la interfaz
    def _on_inertia(self, k: int) -> None:
        self._update_tags()
        self.changed.emit()

    def _on_components(self, *_) -> None:
        if not self._busy:
            self.changed.emit()

    def _on_axis_values(self, *_) -> None:
        if self._busy:
            return
        self.p2.setEnabled(self.chk_p2.isChecked())
        w = self._vector_from_axis()
        self._sync_components(w)
        self._update_readout()
        self.changed.emit()

    def _on_main_axis(self, _: Any = None) -> None:
        if self._busy:
            return
        a_new = self.axis_combo.currentData()
        a, b, c = self._axes
        if a_new == a:
            return
        self._axes = reassign_axes(self._axes, new_main=a_new)
        self._rebuild_axis_combos()
        self._on_axis_values()

    def _on_p1_axis(self, _: Any = None) -> None:
        if self._busy:
            return
        b_new = self.p1_axis.currentData()
        if b_new is None or b_new == self._axes[1]:
            return
        self._axes = reassign_axes(self._axes, new_p1=b_new)
        self._rebuild_axis_combos()
        self._on_axis_values()

    def _on_mode(self, i: int) -> None:
        axis_mode = (i == 0)
        if axis_mode:
            self._fill_axis_widgets([s.value() for s in self.w_sliders])
        self.axis_page.setVisible(axis_mode)
        self.comp_page.setVisible(not axis_mode)

    def _update_tags(self) -> None:
        """Rotula cada I como mínima, media o máxima"""
        I = [s.value() for s in self.I_sliders]
        tags = inertia_ranks(I)                 # '=' si empata: sólido simétrico
        self._tags = tags
        for k, s in enumerate(self.I_sliders):
            s.label.setText(f"I{k + 1} · {tags[k]}")
        self._rebuild_axis_combos()
        # solo se avisa, no se impide: las ecuaciones de Euler valen aunque el sólido no pueda existir
        k = triangle_violation(I)
        if k is not None:
            self.tri_warning.setText(
                f"I{k + 1} > I{(k + 1) % 3 + 1} + I{(k + 2) % 3 + 1}: ningún sólido real tiene estos "
                f"momentos de inercia (desigualdad triangular). Las ecuaciones de Euler siguen valiendo.")
            self.tri_warning.show()
        else:
            self.tri_warning.hide()

    def get(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Devuelve los momentos de inercia y las componentes de la velocidad angular inicial"""
        w = self._vector_from_axis() if self.mode.currentIndex() == 0 else [s.value() for s in self.w_sliders]
        return (tuple(s.value() for s in self.I_sliders), tuple(float(x) for x in w))

    def set(self, I: ArrayLike, w0: ArrayLike) -> None:
        """Fija los momentos de inercia y la velocidad angular inicial sin emitir señales"""
        for s, v in zip(self.I_sliders, I):
            s.setValue(v, silent=True)
        self._sync_components(w0)
        self._update_tags()
        self._fill_axis_widgets([s.value() for s in self.w_sliders])
        axis_mode = self.mode.currentIndex() == 0
        self.axis_page.setVisible(axis_mode)
        self.comp_page.setVisible(not axis_mode)


class SimConfigPanel(QtWidgets.QWidget):
    """Panel de una simulación: integrador, paso de tiempo y física propia opcional"""

    changed = QtCore.Signal()

    def __init__(self, index: int, cfg: SimConfig, shared: PhysicsSliders,
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.index = index
        self.name = cfg.name
        self.color = cfg.color
        self.shared = shared

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(8)

        self.enabled = QtWidgets.QCheckBox(f"Simulación {cfg.name} activa")
        self.enabled.setStyleSheet(f"QCheckBox {{ color: {cfg.color}; font-weight: 600; }}")
        lay.addWidget(self.enabled)

        self.method = QtWidgets.QComboBox()
        for key, label in METHOD_LABEL.items():
            self.method.addItem(label, key)
        self.method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.method.setMinimumContentsLength(16)
        self.dt = FloatSlider("Δt", 0.001, 0.25, 0.001, 0.02, 3)
        lay.addWidget(_group_box("Esquema numérico", [self.method, self.dt]))

        self.linked = QtWidgets.QCheckBox("Usar el problema físico compartido")
        self.linked.setChecked(True)
        self.linked.setToolTip("Desmárcalo para dar a esta simulación sus propios I y Ω₀ "
                               "(p. ej. para comparar cuerpos distintos).")
        self.phys_summary = QtWidgets.QLabel()
        self.phys_summary.setObjectName("muted")
        self.phys_summary.setWordWrap(True)
        self.own = PhysicsSliders(cfg.I, cfg.w0)
        self.own_box = _group_box(f"Física propia de {cfg.name}", [self.own])
        lay.addWidget(_group_box("Problema físico", [self.linked, self.phys_summary, self.own_box]))
        lay.addStretch(1)

        self.set_config(cfg)
        self._sync_link_ui()

        self.enabled.toggled.connect(self.changed)
        self.method.currentIndexChanged.connect(self.changed)
        self.dt.valueChanged.connect(self.changed)
        self.own.changed.connect(self.changed)
        self.linked.toggled.connect(self._on_link_toggled)
        shared.changed.connect(self._update_summary)

    def is_linked(self) -> bool:
        return self.linked.isChecked()

    def set_linked(self, flag: bool) -> None:
        self.linked.blockSignals(True)
        self.linked.setChecked(bool(flag))
        self.linked.blockSignals(False)
        self._sync_link_ui()

    def _on_link_toggled(self, on: bool) -> None:
        if not on:      # parte de los valores compartidos: al desvincular nada salta
            self.own.set(*self.shared.get())
        self._sync_link_ui()
        self.changed.emit()

    def _sync_link_ui(self) -> None:
        linked = self.linked.isChecked()
        self.own_box.setVisible(not linked)
        self.phys_summary.setVisible(linked)
        self._update_summary()

    def _update_summary(self) -> None:
        if not self.linked.isChecked():
            return
        I, w = self.shared.get()
        self.phys_summary.setText(
            f"I = ({I[0]:.2f}, {I[1]:.2f}, {I[2]:.2f})   "
            f"Ω₀ = ({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})")

    def get_config(self) -> SimConfig:
        """Construye la configuración con la física compartida o la propia según el vínculo"""
        I, w0 = self.shared.get() if self.linked.isChecked() else self.own.get()
        return SimConfig(
            name=self.name, color=self.color,
            enabled=self.enabled.isChecked(),
            method=self.method.currentData(),
            I=I, w0=w0, dt=self.dt.value())

    def set_config(self, cfg: SimConfig) -> None:
        widgets = [self.enabled, self.method]
        for w in widgets:
            w.blockSignals(True)
        self.enabled.setChecked(cfg.enabled)
        self.method.setCurrentIndex(self.method.findData(cfg.method))
        for w in widgets:
            w.blockSignals(False)
        self.own.set(cfg.I, cfg.w0)
        self.dt.setValue(cfg.dt, silent=True)


def mono_font(size: int = 11) -> QtGui.QFont:
    """Fuente monoespaciada a partir de la familia MONO"""
    fam = MONO.split('"')[1] if '"' in MONO else MONO.split(",")[0]
    f = QtGui.QFont(fam, size)
    f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return f


class DataTable(QtWidgets.QTableWidget):
    """Tabla de solo lectura, sin rejilla ni selección, que se ajusta a su contenido"""

    ALERT = "#FF8A80"

    def __init__(self, headers: Sequence[str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(28)
        self.setShowGrid(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setWordWrap(False)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hh = self.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(False)
        hh.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setFixedHeight(40)

    def set_rows(self, rows: Sequence[Sequence[Any]]) -> None:
        """Rellena la tabla; cada celda es un texto o un par (texto, opciones con color, mono, right, tip)"""
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                text, opt = (cell, {}) if isinstance(cell, str) else cell
                existing = self.item(r, c)
                it = existing or QtWidgets.QTableWidgetItem()
                it.setText(text)
                it.setForeground(QtGui.QColor(opt.get("color", TEXT)))
                if opt.get("mono"):
                    it.setFont(mono_font(11))
                align = (QtCore.Qt.AlignmentFlag.AlignRight if opt.get("right")
                         else QtCore.Qt.AlignmentFlag.AlignLeft)
                it.setTextAlignment(align | QtCore.Qt.AlignmentFlag.AlignVCenter)
                it.setToolTip(opt.get("tip", ""))
                if existing is None:
                    self.setItem(r, c, it)
        self.resizeColumnsToContents()
        self._fit_height()

    def _fit_height(self) -> None:
        """Alto justo para las filas, más la barra horizontal si las columnas no caben"""
        h = self.horizontalHeader().height() + sum(self.rowHeight(i) for i in range(self.rowCount())) + 6
        if sum(self.columnWidth(c) for c in range(self.columnCount())) > self.width():
            h += self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(max(h, 40))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_height()


def scroll_area(widget: QtWidgets.QWidget, transparent: bool = False, frame: bool = False) -> QtWidgets.QScrollArea:
    """Área con scroll vertical (nunca horizontal) que envuelve `widget`; `transparent` la funde con la tarjeta"""
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    if not frame:
        sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    if transparent:
        sa.setStyleSheet("QScrollArea { background: transparent; }")
        sa.viewport().setAutoFillBackground(False)
    sa.setWidget(widget)
    return sa


def noise_floor_line(plot: pg.PlotWidget) -> pg.InfiniteLine:
    """Línea horizontal discontinua de ruido de redondeo, oculta hasta que se le da una altura"""
    line = pg.InfiniteLine(angle=0, pen=pg.mkPen((255, 255, 255, 110), width=1, style=QtCore.Qt.PenStyle.DashLine),
                           label="ruido de redondeo", labelOpts=dict(position=0.9, color=MUTED))
    plot.addItem(line, ignoreBounds=True)
    line.hide()
    return line


def plain_body(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Contenedor transparente que apila `widgets` (cuerpo de una tarjeta)"""
    w = QtWidgets.QWidget()
    w.setObjectName("plain")
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(8)
    for x in widgets:
        v.addWidget(x)
    v.addStretch(1)
    return w


def muted_label(text: str) -> QtWidgets.QLabel:
    """Texto explicativo gris con ajuste de línea"""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("muted")
    lbl.setWordWrap(True)
    return lbl


def control_row(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Fila de controles alineados a la izquierda, sin márgenes"""
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    lay.addStretch(1)
    return w


DASH = ("—", {"right": True, "color": MUTED})      # celda sin dato


def num_or_dash(x: float, fmt: str = "{:.2f}") -> tuple[str, dict[str, Any]]:
    """Celda numérica a la derecha en tipografía fija, o un guion si el valor no es finito"""
    return (fmt.format(x), {"mono": True, "right": True}) if np.isfinite(x) else DASH


def dot(cfg: SimConfig) -> tuple[str, dict[str, Any]]:
    """Celda con un punto del color de la simulación"""
    return ("●", {"color": cfg.color})


def num(x: float, fmt: str = "{:+.3e}", alert: Any = False) -> tuple[str, dict[str, Any]]:
    """Celda numérica en tipografía fija, resaltada si alert"""
    return (fmt.format(x), {"mono": True, "right": True, "color": DataTable.ALERT if alert else TEXT})


class LinkedCursor(QtCore.QObject):
    """Línea vertical que acompaña al ratón en varias gráficas a la vez y avisa del instante"""

    moved = QtCore.Signal(float)               # t bajo el ratón; NaN cuando el ratón sale

    def __init__(self, plots: Sequence[pg.PlotItem], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.lines: list[Any] = []
        pen = pg.mkPen((255, 255, 255, 140), width=1, style=QtCore.Qt.PenStyle.DashLine)
        for p in plots:
            ln = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            ln.setZValue(20)
            ln.hide()
            p.addItem(ln, ignoreBounds=True)
            self.lines.append(ln)
            p.scene().sigMouseMoved.connect(partial(self._on_move, p))
            p.viewport().installEventFilter(self)

    def _on_move(self, plot: pg.PlotItem, pos: Any) -> None:
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        x = float(vb.mapSceneToView(pos).x())
        for ln in self.lines:
            ln.setPos(x)
            ln.show()
        self.moved.emit(x)

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if ev.type() == QtCore.QEvent.Type.Leave:
            for ln in self.lines:
                ln.hide()
            self.moved.emit(float("nan"))
        return False


class FlowLayout(QtWidgets.QLayout):
    """Coloca los widgets en fila y baja a otra línea cuando no caben (nada queda cortado)"""

    def __init__(self, parent: QtWidgets.QWidget | None = None, hspacing: int = 10, vspacing: int = 6) -> None:
        super().__init__(parent)
        self._items, self._h, self._v = [], hspacing, vspacing

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int) -> QtWidgets.QLayoutItem | None:
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int) -> QtWidgets.QLayoutItem | None:
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return self._layout(QtCore.QRect(0, 0, w, 0), True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        s = QtCore.QSize()
        for it in self._items:
            if not it.isEmpty():
                s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect: QtCore.QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for it in self._items:
            if it.isEmpty():
                continue
            sz = it.sizeHint()
            if x + sz.width() > r.right() + 1 and line_h > 0:
                x, y, line_h = r.x(), y + line_h + self._v, 0
            if not test_only:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), sz))
            x += sz.width() + self._h
            line_h = max(line_h, sz.height())
        return y + line_h - rect.y() + m.bottom()


class AdaptiveSplitter(QtWidgets.QSplitter):
    """Reparte en horizontal si hay sitio y en vertical si la ventana es estrecha"""

    def __init__(self, threshold: int = 1000, sizes_h: Sequence[int] = (1000, 1000),
                 sizes_v: Sequence[int] = (1000, 1000), parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.threshold, self.sizes_h, self.sizes_v = threshold, sizes_h, sizes_v
        self._sized = False                 # el reparto inicial se aplica una sola vez, al mostrarse
        self.setChildrenCollapsible(False)

    def showEvent(self, e: QtGui.QShowEvent) -> None:
        super().showEvent(e)
        if not self._sized:
            self._sized = True
            self.setSizes(list(self.sizes_h if self.orientation() == QtCore.Qt.Orientation.Horizontal
                               else self.sizes_v))

    def minimumSizeHint(self) -> QtCore.QSize:
        # el mínimo es el del modo vertical (el más estrecho); en horizontal ya no se activa
        w = max((self.widget(i).minimumSizeHint().width() for i in range(self.count())), default=0)
        return QtCore.QSize(w, 2 * 220)

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        want = (QtCore.Qt.Orientation.Horizontal if e.size().width() >= self.threshold
                else QtCore.Qt.Orientation.Vertical)
        if want != self.orientation():
            self.setOrientation(want)
            self.setSizes(list(self.sizes_h if want == QtCore.Qt.Orientation.Horizontal else self.sizes_v))
        super().resizeEvent(e)


class Card(QtWidgets.QFrame):
    """Marco con título, subtítulo opcional y un widget dentro"""

    def __init__(self, title: str, widget: QtWidgets.QWidget, subtitle: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel(title)
        t.setObjectName("cardTitle")
        head.addWidget(t)
        head.addStretch(1)
        self.subtitle = QtWidgets.QLabel(subtitle)
        self.subtitle.setObjectName("muted")
        head.addWidget(self.subtitle)
        lay.addLayout(head)
        lay.addWidget(widget, 1)


def build_stylesheet(font: str) -> str:
    """Hoja de estilo de la aplicación con la paleta y la fuente dadas"""
    return f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: "{font}"; font-size: 13px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
             margin-top: 16px; padding: 12px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {MUTED}; font-weight: 600; letter-spacing: 0.5px; }}
QGroupBox QWidget {{ background: transparent; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card QLabel {{ background: transparent; }}
QFrame#card QWidget#plain {{ background: transparent; }}
QLabel#cardTitle {{ font-size: 15px; font-weight: 700; color: #ffffff; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#title {{ font-size: 24px; font-weight: 700; color: #ffffff; letter-spacing: 1px; }}
QLabel#warning {{ color: {AMBER}; }}
QLabel#notice {{ background: rgba(255, 215, 64, 30); color: #ffe9a0; border: 1px solid {AMBER}; border-radius: 6px; padding: 6px 10px; }}
QLabel#bannerTitle {{ color: {MUTED}; font-weight: 600; font-size: 13px; }}
QPushButton {{ background: {BUTTON}; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 6px 10px; }}
QPushButton:hover {{ background: #575c64; }}
QPushButton:pressed {{ background: #33363b; }}
QPushButton#primary {{ background: #00b37e; border-color: #3ddc97; font-weight: 700; color: white; }}
QPushButton#primary:hover {{ background: #00c98d; }}
QPushButton#accent {{ background: #7c3aed; border-color: #a78bfa; font-weight: 700; color: white; }}
QPushButton#accent:hover {{ background: #8b5cf6; }}
QComboBox, QDoubleSpinBox, QSpinBox {{ background: {FIELD}; border: 1px solid {BORDER};
               border-radius: 5px; padding: 3px 6px; }}
QComboBox QAbstractItemView {{ background: {FIELD}; selection-background-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: #1f2124; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #7FE7FF; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 14px; margin: -6px 0; border-radius: 7px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {BG}; }}
QTabBar::tab {{ background: {PANEL}; padding: 6px 14px; border-top-left-radius: 6px;
                border-top-right-radius: 6px; margin-right: 2px; color: {MUTED}; }}
QTabBar::tab:selected {{ background: {BUTTON}; color: white; }}
QTabWidget#views::pane {{ border: none; background: {BG}; }}
QTabWidget#views QTabBar::tab {{ padding: 8px 14px; font-size: 14px; font-weight: 600;
                                 margin-right: 4px; border: 1px solid {BORDER}; border-bottom: none; }}
QTabWidget#views QTabBar::tab:selected {{ background: {ACCENT}; color: white;
                                          border-bottom: 3px solid {AMBER}; }}
QTabWidget#views QTabBar::tab:hover:!selected {{ background: #4a4e55; color: white; }}
QSplitter::handle {{ background: {BG}; }}
QTableWidget {{ background: transparent; color: {TEXT}; border: none; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: transparent; color: {MUTED}; border: none;
                        border-bottom: 1px solid {BORDER}; padding: 4px 8px; font-weight: 600; }}
QTableWidget::item {{ padding: 0 8px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {MUTED};
                         border-radius: 3px; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background: #00b37e; border-color: #3ddc97; }}
"""


DEFAULT_CONFIGS = [
    SimConfig("A", SIM_COLORS[0], True, M_EULER),
    SimConfig("B", SIM_COLORS[1], True, M_CN),
    SimConfig("C", SIM_COLORS[2], True, M_RK4),
]

HELP_HTML = """
<h2>Atajos</h2>
<table cellspacing="6">
<tr><td><b>Espacio</b></td><td>iniciar / pausar</td></tr>
<tr><td><b>R</b></td><td>reiniciar (t = 0)</td></tr>
<tr><td><b>P</b></td><td>recalcular la previsualización</td></tr>
<tr><td><b>,</b> &nbsp; <b>.</b></td><td>retroceder / avanzar un paso de vista</td></tr>
<tr><td><b>Ctrl+1 … Ctrl+5</b></td><td>cambiar de pestaña</td></tr>
<tr><td><b>F1</b></td><td>esta ayuda</td></tr>
</table>

<h2>Las tres líneas de cada ficha de estabilidad</h2>
<ul>
<li><b>Órbita</b>: propiedad <i>física</i>; sólo depende de I y de Ω₀. Mide cuánto se aleja el eje de giro
de su dirección inicial a lo largo de la órbita exacta.
<span style="color:{ok}">estable</span>: se desvía poco (menos de 90°), del orden de la perturbación inicial:
es lo que ocurre girando casi sobre el eje de mayor o menor inercia.
<span style="color:{bad}">inestable</span>: <i>vuelca</i>; el eje llega a más de 90° (casi 180° para perturbaciones
pequeñas) de donde empezó, por pequeña que sea la perturbación: es el teorema del eje intermedio, y también
ocurre sobre la separatriz. <span style="color:{mar}">marginal</span>: reposo, esfera o sólido simétrico
(no hay eje intermedio).</li>
<li><b>Esquema</b>: lo que la teoría lineal prevé para el método numérico elegido y Δt en el centro
que rodea la órbita (módulo y fase por paso, extrapolado al horizonte de la previsualización).
No depende del eje: es el comportamiento del <i>método</i>.</li>
<li><b>Observado</b>: lo medido en la simulación frente a la solución exacta, en el instante mostrado.</li>
</ul>

<h2>Error de módulo y de fase</h2>
<p>La órbita exacta se recorre con una amplitud y una velocidad angular de fase. El
<b>error de módulo</b> r es la desviación relativa de la amplitud (r &gt; 0: el esquema inyecta energía de
oscilación; r &lt; 0: la disipa). El <b>error de fase</b> ψ es cuánto se adelanta (+) o se retrasa (−) la
solución numérica sobre la exacta, en grados de ciclo (y su equivalente en segundos).</p>
<p>Cada error se clasifica comparándolo con el ruido de redondeo de la máquina y con lo
visible en una gráfica:</p>
<ul>
<li><span style="color:{g0}"><b>≈ 0</b></span>: no se distingue del redondeo de la máquina; se puede ignorar.</li>
<li><span style="color:{g1}"><b>ínfimo</b></span>: real, pero por debajo de 10<sup>−6</sup> (módulo) o 10<sup>−3</sup>° (fase);
invisible en cualquier gráfica.</li>
<li><span style="color:{g2}"><b>relevante</b></span>: hay que tenerlo en cuenta.</li>
</ul>
<p>Teoría (centro lineal): Euler explícito → módulo crece, fase se retrasa; Euler implícito → módulo
decae, fase se retrasa; Crank–Nicolson → módulo se conserva (|R| = 1), fase se retrasa; RK4 → error
minúsculo en ambos (Δt⁵ en módulo, Δt⁴ en fase).</p>

<h2>Otros avisos</h2>
<ul>
<li><b>Divergió</b>: ‖ω‖ supera 10³ veces lo posible en la órbita exacta; esa simulación se congela y las
demás continúan. La simulación se pausa sólo cuando divergen todas las activas.</li>
<li><b>Falló</b>: <code>numerical_engine</code> lanzó una excepción (p. ej. Newton sin converger); se congela igual
y el mensaje del error aparece en la ficha y en la barra de estado.</li>
<li><b>Δt demasiado grande</b>: Δt·β &gt; 0.25, es decir, menos de ≈ 25 pasos por ciclo de la oscilación.</li>
<li><b>Convergencia</b>: pendiente log-log del error frente a Δt; debe coincidir con el orden teórico del
esquema mientras el error esté por encima del ruido de redondeo.</li>
</ul>
"""


class MainWindow(QtWidgets.QMainWindow):
    """Ventana principal con los controles, las vistas y el bucle de animación"""
    FRAME_MS = 30

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Janibekov Lab — Dinámica de rotación del sólido rígido")
        avail = (QtGui.QGuiApplication.primaryScreen().availableGeometry()
                 if QtGui.QGuiApplication.primaryScreen() else QtCore.QRect(0, 0, 1680, 980))
        self.resize(min(1680, int(avail.width() * 0.96)), min(980, int(avail.height() * 0.92)))

        self.sims: list[Simulation] = []
        self.previews: dict[int, dict] = {}
        self.clock = 0.0
        self.running = False
        self.frame = 0
        self.family_items = []
        self.separatrix_items = []
        self.sat_items = []
        self._plane_k, self._plane_sign_val = 1, 1
        self._workers: list[QtCore.QThread] = []          # hilos vivos (previsualización y análisis)
        self._preview_token = 0
        self._analysis: dict[str, AnalysisWorker] = {}    # análisis en curso: rich (Richardson) y conv
        self._rich_T = 0.0
        self.view_t = None
        self._div_notified = set()
        self._sat_key = None
        self._last_cfgs = None
        self._lin_cache = {}
        self._w_range_key = None

        self._build_ui()
        self._build_phase_items()
        self._build_plot_items()

        self._param_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._param_timer.timeout.connect(self.apply_params)

        self.apply_preset(DEFAULT_SCENARIO)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.FRAME_MS)

        for key, slot in (("Space", self.toggle_run), ("R", self.reset),
                          ("P", self.compute_preview)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)
        QtGui.QShortcut(QtGui.QKeySequence("F1"), self, activated=self.show_help)
        QtGui.QShortcut(QtGui.QKeySequence("."), self, activated=self.step_forward)
        QtGui.QShortcut(QtGui.QKeySequence(","), self, activated=self.step_back)
        for k in range(self.view_tabs.count()):
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{k + 1}"), self,
                            activated=partial(self.view_tabs.setCurrentIndex, k))

    def _build_ui(self) -> None:
        """Construye la ventana: controles a la izquierda y pestañas de vistas a la derecha"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        root = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.setChildrenCollapsible(False)
        root.setHandleWidth(10)
        outer.addWidget(root)

        scroll = scroll_area(self._build_controls(), frame=True)
        left_w = QtWidgets.QWidget()
        left_w.setMinimumWidth(330)
        left_w.setMaximumWidth(560)
        left = QtWidgets.QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(8)
        left.addWidget(self._build_header())     # fija: Iniciar/Reiniciar siempre a la vista
        left.addWidget(scroll, 1)
        root.addWidget(left_w)

        main_w = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout(main_w)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(8)
        main.addLayout(self._build_banner())

        self.dyn_tab = self._build_dynamics_tab()

        self.drift_tab = self._build_drift_tab()

        self.view_tabs = self._build_view_tabs()
        main.addWidget(self.view_tabs, 1)
        main.addLayout(self._build_timeline())
        root.addWidget(main_w)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([380, 1300])

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def _build_dynamics_tab(self) -> QtWidgets.QWidget:
        """Pestaña Dinámica: órbita en el espacio de fases y orientación del cuerpo"""
        self.phase_view = gl.GLViewWidget()
        self.phase_view.setBackgroundColor(VIEW_BG)
        self.sat_view = gl.GLViewWidget()
        self.sat_view.setBackgroundColor(VIEW_BG)
        self.phase_view.setMinimumSize(200, 200)
        self.sat_view.setMinimumSize(200, 200)

        def surface_check(text: str, color: str, tip: str) -> QtWidgets.QCheckBox:
            c = QtWidgets.QCheckBox(text)
            c.setChecked(True)
            c.setToolTip(tip)
            c.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            c.toggled.connect(self._update_static_visibility)
            return c

        self.chk_E = surface_check("Energía", "#40D98C", "Elipsoide de energía (2T constante)")
        self.chk_L = surface_check("Momento", "#F259BF", "Elipsoide de momento angular (‖L‖ constante)")
        self.chk_sep = surface_check("Separatriz", "#FFFFFF", "Curvas que separan los dos tipos de órbita")
        self.chk_fam = surface_check("Polhodias", "#8C9EC7", "Órbitas posibles con esta energía y este momento")
        self.chk_exact = surface_check("Exacta", AMBER,
                                       "Órbita exacta (solución analítica) y, en blanco, dónde estaría ω "
                                       "en este instante: lo que se separa de ahí es error de fase")
        self.trail_spin = QtWidgets.QDoubleSpinBox()
        self.trail_spin.setRange(1, 5000)
        self.trail_spin.setDecimals(0)
        self.trail_spin.setSingleStep(10)
        self.trail_spin.setValue(60)
        self.trail_spin.setSuffix(" s")
        self.trail_spin.setKeyboardTracking(False)
        self.trail_spin.setToolTip("Duración visible de la estela, igual para todas las simulaciones "
                                   "(órbita 3D y plano de fases)")
        bar = FlowLayout(hspacing=12, vspacing=4)
        for c in (self.chk_E, self.chk_L, self.chk_sep, self.chk_fam, self.chk_exact):
            bar.addWidget(c)
        trail_box = QtWidgets.QWidget()
        tb = QtWidgets.QHBoxLayout(trail_box)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.addWidget(QtWidgets.QLabel("Estela"))
        tb.addWidget(self.trail_spin)
        bar.addWidget(trail_box)

        self.surface_notice = self._notice_label()
        phase_body = QtWidgets.QWidget()
        phase_body.setObjectName("plain")
        pv = QtWidgets.QVBoxLayout(phase_body)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)
        pv.addLayout(bar)
        pv.addWidget(self.surface_notice)
        pv.addWidget(self.phase_view, 1)
        self.phase_card = Card("Órbita en el espacio de fases (ω₁, ω₂, ω₃)", phase_body,
                               "elipsoide T · elipsoide ‖L‖ · separatriz")

        self.shape_combo = QtWidgets.QComboBox()
        self.shape_combo.addItems(["Satélite", "Paralelepípedo", "Elipsoide", "Vaca", "Caza Estelar", "OVNI"])
        self.shape_combo.currentIndexChanged.connect(lambda _: self._rebuild_satellites(force=True))
        sat_bar = QtWidgets.QHBoxLayout()
        sat_bar.addWidget(QtWidgets.QLabel("Forma"))
        sat_bar.addWidget(self.shape_combo)
        sat_bar.addStretch(1)
        sat_body = QtWidgets.QWidget()
        sat_body.setObjectName("plain")
        sv = QtWidgets.QVBoxLayout(sat_body)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(6)
        sv.addLayout(sat_bar)
        sv.addWidget(self.sat_view, 1)
        self.sat_card = Card("Orientación del cuerpo (marco inercial)", sat_body)
        self.sat_card.subtitle.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.sat_card.subtitle.setText(
            "<span style='color:%s'>e₁</span> <span style='color:%s'>e₂</span> "
            "<span style='color:%s'>e₃</span> ejes · "
            "<span style='color:#FFD933'>L</span> momento · <span style='color:#FFFFFF'>ω</span> vel. angular"
            % tuple(AXIS_COLORS))

        self.dyn_tab = AdaptiveSplitter(1000)
        self.dyn_tab.setChildrenCollapsible(False)
        self.dyn_tab.addWidget(self.phase_card)
        self.dyn_tab.addWidget(self.sat_card)
        self.dyn_tab.setSizes([1, 1])
        return self.dyn_tab

    def _build_drift_tab(self) -> QtWidgets.QWidget:
        """Pestaña Derivas: deriva de energía, deriva del momento y velocidad angular de la referencia"""
        self.plot_T = pg.PlotWidget()
        self.plot_L = pg.PlotWidget()
        self.plot_w = pg.PlotWidget()
        for p, ylab in ((self.plot_T, "ΔT"), (self.plot_L, "Δ‖L‖"), (self.plot_w, "ω")):
            self._style_plot(p, "t", ylab)
            p.getPlotItem().setDownsampling(auto=True, mode="peak")
            p.getPlotItem().setClipToView(True)
            p.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.plot_L.setXLink(self.plot_T)
        self.plot_w.setXLink(self.plot_T)
        self.card_T = Card("Deriva de energía  ΔT(t) = T(t) − T₀", self.plot_T)
        self.card_L = Card("Deriva del momento  Δ‖L‖(t) = ‖L(t)‖ − ‖L₀‖", self.plot_L)
        self.card_w = Card("Componentes ω(t) — simulación de referencia", self.plot_w)
        drift = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        for c in (self.card_T, self.card_L, self.card_w):
            drift.addWidget(c)
        drift.setSizes([360, 360, 300])
        self.chk_log = QtWidgets.QCheckBox("Escala logarítmica |Δ|")
        self.chk_log.toggled.connect(self._on_log_toggled)
        log_row = QtWidgets.QHBoxLayout()
        self.drift_readout = QtWidgets.QLabel()
        self.drift_readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.drift_readout.setObjectName("muted")
        log_row.addWidget(self.drift_readout, 1)
        log_row.addWidget(self.chk_log)
        self.drift_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(self.drift_tab)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(6)
        dv.addLayout(log_row)
        dv.addWidget(drift, 1)
        return self.drift_tab

    def _build_view_tabs(self) -> QtWidgets.QTabWidget:
        """Las cinco pestañas con el reloj y el selector de referencia en la esquina"""
        self.view_tabs = QtWidgets.QTabWidget()
        self.view_tabs.setObjectName("views")
        self.view_tabs.setDocumentMode(True)
        self.err_tab = self._build_error_tab()
        self.plane_tab = self._build_plane_tab()
        self.analysis_tab = self._build_analysis_tab()
        for widget, label in ((self.dyn_tab, "🌐  Dinámica"),
                              (self.drift_tab, "📈  Derivas"),
                              (self.err_tab, "📐  Error vs exacta"),
                              (self.plane_tab, "🧭  Plano de fases"),
                              (self.analysis_tab, "📋  Análisis")):
            self.view_tabs.addTab(widget, label)
        for i in range(self.view_tabs.count()):
            self.view_tabs.setTabToolTip(i, f"Ctrl+{i + 1}")
        self.view_tabs.currentChanged.connect(self._on_tab_changed)
        corner = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 0, 4)
        cl.setSpacing(8)
        self.clock_label = QtWidgets.QLabel("t = 0.00 s")
        self.clock_label.setObjectName("muted")
        self.clock_label.setMinimumWidth(80)
        ref_lbl = QtWidgets.QLabel("Referencia")
        ref_lbl.setObjectName("muted")
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems([c.name for c in DEFAULT_CONFIGS])
        self.ref_combo.setToolTip("Simulación cuyas superficies, carta de fases y ω(t) se muestran")
        self.ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        for w_ in (self.clock_label, ref_lbl, self.ref_combo):
            cl.addWidget(w_)
        self.view_tabs.setCornerWidget(corner, QtCore.Qt.Corner.TopRightCorner)
        return self.view_tabs

    def _on_tab_changed(self, _: int | None = None) -> None:
        self._refresh_views()
        self._update_status()

    @staticmethod
    def _style_plot(p: pg.PlotItem, xlab: str, ylab: str) -> None:
        """Aplica a una gráfica la rejilla, las etiquetas y las fuentes comunes"""
        p.showGrid(x=True, y=True, alpha=0.18)
        p.setLabel("bottom", xlab, **axis_label_style())
        p.setLabel("left", ylab, **axis_label_style())
        tick_font = QtGui.QFont(UI_FONT, 9)
        for ax in ("left", "bottom"):
            p.getAxis(ax).setTickFont(tick_font)
            p.getAxis(ax).setTextPen(TEXT)
        p.getAxis("left").enableAutoSIPrefix(False)

    def _build_banner(self) -> QtWidgets.QHBoxLayout:
        """Construye la fila de tarjetas de estado, una por simulación"""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        title = QtWidgets.QLabel("ESTADO")
        title.setObjectName("bannerTitle")
        title.setToolTip("Una tarjeta por simulación, con tres cosas distintas:\n"
                         "· Órbita: estabilidad física del movimiento (solo depende de I y Ω₀). Da el color del borde.\n"
                         "· Esquema: qué prevé la teoría lineal del método y el Δt para el módulo y la fase.\n"
                         "· Observado: el error medido contra la solución exacta; en ámbar si es relevante.")
        row.addWidget(title)
        self.badges, self._badge_state = [], [None] * len(DEFAULT_CONFIGS)
        for _ in DEFAULT_CONFIGS:
            b = QtWidgets.QLabel()
            b.setTextFormat(QtCore.Qt.TextFormat.RichText)
            b.setWordWrap(True)
            b.setMinimumWidth(150)
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
            b.hide()
            row.addWidget(b, 1)
            self.badges.append(b)
        return row

    def _build_error_tab(self) -> QtWidgets.QWidget:
        """Pestaña Error vs exacta: comparación con la solución exacta y errores de módulo y fase"""
        self._err_prev_stale = True
        self.err_comp = QtWidgets.QComboBox()
        self.err_comp.addItems(["ω₁", "ω₂", "ω₃"])
        self.err_comp.setCurrentIndex(1)
        self.err_comp.setToolTip("Componente de ω que se dibuja arriba (los errores usan las tres)")
        self.err_unit = QtWidgets.QComboBox()
        self.err_unit.addItems(["° de ciclo", "segundos"])
        self.err_unit.setToolTip("Unidad del error de fase: grados de un ciclo de la órbita, o "
                                 "tiempo de adelanto/retraso en segundos")
        self.chk_log_err = QtWidgets.QCheckBox("Escala logarítmica |error|")
        self.chk_log_err.setChecked(True)
        self.chk_log_err.setToolTip("Los errores de un esquema y otro difieren en muchos órdenes de "
                                    "magnitud: en log se ven todos a la vez (se pierde el signo; "
                                    "el panel de la derecha lo da)")
        for w in (self.err_comp, self.err_unit):
            w.currentIndexChanged.connect(self._on_error_options)
        self.chk_log_err.toggled.connect(self._on_error_options)
        self.err_readout = QtWidgets.QLabel()
        self.err_readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.err_readout.setObjectName("muted")

        bar = FlowLayout(hspacing=12, vspacing=4)
        for w in (QtWidgets.QLabel("Componente"), self.err_comp, QtWidgets.QLabel("Fase en"),
                  self.err_unit, self.chk_log_err, self.err_readout):
            bar.addWidget(w)
        self.err_notice = self._notice_label()

        self.plot_ew = pg.PlotWidget()
        self.plot_er = pg.PlotWidget()
        self.plot_ep = pg.PlotWidget()
        for p, ylab in ((self.plot_ew, "ω₂"), (self.plot_er, "r"), (self.plot_ep, "ψ")):
            self._style_plot(p, "t", ylab)
            p.getPlotItem().setDownsampling(auto=True, mode="peak")
            p.getPlotItem().setClipToView(True)
        self.plot_ew.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.plot_er.setXLink(self.plot_ew)
        self.plot_ep.setXLink(self.plot_ew)
        zero = pg.mkPen((255, 255, 255, 70), width=1, style=QtCore.Qt.PenStyle.DotLine)
        self.err_zero = [self.plot_er.addLine(y=0, pen=zero), self.plot_ep.addLine(y=0, pen=zero)]
        self.err_floor = [noise_floor_line(p) for p in (self.plot_er, self.plot_ep)]   # bajo ella: ruido de redondeo

        card_w = Card("ω(t): numérica frente a la solución exacta", self.plot_ew,
                      "discontinua blanca = exacta")
        card_r = Card("Error de módulo  r(t)", self.plot_er,
                      "0 = sobre la órbita exacta")
        card_p = Card("Error de fase  ψ(t)", self.plot_ep,
                      "+ adelanta · − retrasa")
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        for c in (card_w, card_r, card_p):
            split.addWidget(c)
        split.setSizes([360, 250, 250])

        left = QtWidgets.QWidget()
        left.setObjectName("plain")
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        lv.addLayout(bar)
        lv.addWidget(self.err_notice)
        lv.addWidget(split, 1)

        self.err_info = QtWidgets.QLabel()
        self.err_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.err_info.setWordWrap(True)
        self.err_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        info = Card("Lectura del error", scroll_area(self.err_info, transparent=True))
        info.setMinimumWidth(250)

        w = AdaptiveSplitter(640, sizes_h=(900, 330), sizes_v=(700, 280))
        w.addWidget(left)
        w.addWidget(info)
        return w

    HOVER_HINT = "Pasa el ratón por las gráficas para leer los valores en un instante."

    def _sims_at(self, t: float) -> list[Any]:
        """Pares simulación e índice de fila de cada simulación activa en el instante t"""
        out = []
        for sim in self.sims:
            if sim.cfg.enabled and sim.nv > 1:
                out.append((sim, min(sim.index_at(t), sim.nv - 1)))
        return out

    def _on_drift_cursor(self, t: float) -> None:
        """Muestra en la pestaña Derivas los valores bajo el cursor"""
        if t != t:
            self.drift_readout.setText(f"<span style='color:{MUTED}'>{self.HOVER_HINT}</span>")
            return
        parts = [f"<span style='color:{s.cfg.color}'>●</span> {s.cfg.name}: ΔT {s.vdT[i]:+.2e} · "
                 f"Δ‖L‖ {s.vdL[i]:+.2e}" for s, i in self._sims_at(t)]
        self.drift_readout.setText(f"<span style='font-family:{MONO}'>t = {max(t, 0):.2f} s</span> &nbsp; "
                                   + " &nbsp; ".join(parts))

    def _on_err_cursor(self, t: float) -> None:
        """Muestra en la pestaña de error los valores bajo el cursor"""
        if t != t:
            self.err_readout.setText(f"<span style='color:{MUTED}'>{self.HOVER_HINT}</span>")
            return
        def mark(v: float, floor: float) -> str:
            return " ≈0" if abs(v) <= floor else ""
        parts = []
        for s, i in self._sims_at(t):
            fl = noise_floor(i)
            parts.append(f"<span style='color:{s.cfg.color}'>●</span> {s.cfg.name}: "
                         f"r {s.v_er[i]:+.2e}{mark(s.v_er[i], fl)} · "
                         f"ψ {s.v_ep[i]:+.3g}°{mark(s.v_ep[i], math.degrees(fl))}")
        self.err_readout.setText(f"<span style='font-family:{MONO}'>t = {max(t, 0):.2f} s</span> &nbsp; "
                                 + " &nbsp; ".join(parts))

    def _on_error_options(self, *_) -> None:
        self._apply_error_axes()
        self._err_prev_stale = True
        self._ew_cache = None
        self._ew_fit = True
        self._refresh_views()

    def _apply_error_axes(self) -> None:
        """Ajusta ejes, etiquetas y escala log de las gráficas de error según las opciones"""
        log = self.chk_log_err.isChecked()
        use_s = self.err_unit.currentIndex() == 1
        comp = "ω" + "₁₂₃"[self.err_comp.currentIndex()]
        self.plot_ew.setLabel("left", comp, **axis_label_style())
        self.plot_er.setLogMode(False, log)
        self.plot_ep.setLogMode(False, log)
        self.plot_er.setLabel("left", "|r|" if log else "r", **axis_label_style())
        unit = "s" if use_s else "°"
        self.plot_ep.setLabel("left", f"|ψ| ({unit})" if log else f"ψ ({unit})", **axis_label_style())
        for z in self.err_zero:
            z.setVisible(not log)
        self._update_error_floor()

    def _update_error_floor(self) -> None:
        """Coloca la línea del ruido de redondeo en las gráficas de error"""
        log = self.chk_log_err.isChecked()
        n = max([s.v_steps for s in self.sims if s.cfg.enabled] + [1])
        fl = noise_floor(n)
        seconds = self.err_unit.currentIndex() == 1
        for ln, v, shown in ((self.err_floor[0], fl, True), (self.err_floor[1], math.degrees(fl), not seconds)):
            ln.setVisible(log and shown)
            if log and shown:
                ln.setPos(math.log10(v))

    @staticmethod
    def _fmt_err(d: Array, log: bool) -> Array:
        """En escala log omite los puntos por debajo del ruido de redondeo"""
        return mask_below(d, 1e-16) if log else d

    @staticmethod
    def _same_exact(a: SimConfig, b: SimConfig) -> bool:
        return same_problem(a.I, a.w0, b.I, b.w0)

    def _update_error_notice(self) -> None:
        """Avisa si alguna simulación tiene otra física o un paso demasiado grande"""
        cfgs = self.configs()
        ref = cfgs[self.ref_index()]
        diff = [c.name for k, c in enumerate(cfgs)
                if c.enabled and k != self.ref_index() and not self._same_exact(ref, c)]
        coarse = []
        for c in cfgs:
            lin = self._lin(c) if c.enabled else None
            res = step_resolution(lin, c.dt) if lin is not None else None
            if res is not None and res["coarse"]:
                coarse.append(f"<b>{c.name}</b> ({res['steps_per_cycle']:.0f} pasos por ciclo"
                              + (", |R| > 1: explota" if res["explodes"] else "") + ")")
        if diff or coarse:
            msgs = []
            if diff:
                msgs.append(f"<b>{' y '.join(diff)}</b> {'usa' if len(diff) == 1 else 'usan'} otra I u otro Ω₀: "
                            f"en la gráfica de arriba su solución exacta propia va punteada con su color; "
                            f"los errores siempre se miden frente a la exacta de cada simulación.")
            if coarse:
                msgs.append("⚠ Δt demasiado grande para la dinámica (Δt·β > "
                            f"{COARSE_BETA_DT:g}): {', '.join(coarse)}. Los errores serán grandes; "
                            "reduce Δt o cambia de esquema.")
            self.err_notice.setText("<br>".join(msgs))
            self.err_notice.show()
        else:
            self.err_notice.hide()

    @staticmethod
    def _exact_curve(exact: EulerExact, t_end: float, comp: int, live: bool) -> tuple[Array, Array]:
        """Componente de la solución exacta en una malla propia, independiente del paso"""
        t, W = exact_sample(exact, t_end, 8000 if live else 40000)
        return t, W[:, comp]

    def _build_plane_tab(self) -> QtWidgets.QWidget:
        """Pestaña Plano de fases: carta de fases con su panel de estabilidad del equilibrio"""
        self.plot_plane = pg.PlotWidget()
        self._style_plot(self.plot_plane, "ω_i", "ω_j")
        self.plot_plane.setAspectLocked(True)
        self.plot_plane.addLegend(offset=(8, 8), labelTextSize="9pt")

        controls = FlowLayout(hspacing=10, vspacing=4)
        controls.addWidget(QtWidgets.QLabel("Equilibrio"))
        self.plane_axis = QtWidgets.QComboBox()
        self.plane_axis.addItems(["Automático (eje de la órbita de ref.)",
                                  "e₁  (ω₁ = ±Ω)", "e₂  (ω₂ = ±Ω)", "e₃  (ω₃ = ±Ω)"])
        self.plane_axis.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.plane_axis.setMinimumContentsLength(12)
        self.plane_sign = QtWidgets.QComboBox()
        self.plane_sign.addItems(["+Ω", "−Ω"])
        self.plane_sign.setEnabled(False)
        self.chk_plane_field = QtWidgets.QCheckBox("Campo de direcciones")
        self.chk_plane_field.setChecked(True)
        self.plane_sign.setToolTip("Solo se elige con un equilibrio fijo; en automático lo decide "
                                   "la órbita de la simulación de referencia")
        for w in (self.plane_axis, self.plane_sign):
            w.currentIndexChanged.connect(self._update_phase_plane)
        controls.addWidget(self.plane_axis)
        self.plane_sign_lbl = QtWidgets.QLabel("Signo")
        controls.addWidget(self.plane_sign_lbl)
        controls.addWidget(self.plane_sign)
        self.chk_plane_field.toggled.connect(self._update_phase_plane)
        controls.addWidget(self.chk_plane_field)

        body = QtWidgets.QWidget()
        body.setObjectName("plain")
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(controls)
        self.plane_notice = self._notice_label()
        v.addWidget(self.plane_notice)
        v.addWidget(self.plot_plane, 1)
        self.plane_card = Card("Plano de fases sobre el elipsoide de energía", body)

        info = QtWidgets.QFrame()
        info.setObjectName("card")
        info.setMinimumWidth(240)
        iv = QtWidgets.QVBoxLayout(info)
        iv.setContentsMargins(14, 12, 14, 12)
        iv.setSpacing(10)
        t = QtWidgets.QLabel("Estabilidad del equilibrio")
        t.setObjectName("cardTitle")
        iv.addWidget(t)
        self.plane_badge = QtWidgets.QLabel()
        self.plane_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plane_badge.setWordWrap(True)
        iv.addWidget(self.plane_badge)
        self.plane_info = QtWidgets.QLabel()
        self.plane_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.plane_info.setWordWrap(True)
        self.plane_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        iv.addWidget(scroll_area(self.plane_info, transparent=True), 1)

        w = AdaptiveSplitter(900, sizes_h=(720, 330), sizes_v=(620, 300))
        w.addWidget(self.plane_card)
        w.addWidget(info)
        return w

    def _build_analysis_tab(self) -> QtWidgets.QWidget:
        """Pestaña Análisis: conservación, Richardson y convergencia a la izquierda; equilibrios y estabilidad a la derecha"""
        cons_card, rich_card, conv_card = self._conservation_card(), self._richardson_card(), self._convergence_card()
        eq_card, stab_card = self._equilibria_card(), self._stability_card()
        right_w = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(eq_card)
        right.addWidget(stab_card)
        right.addStretch(1)

        left_w = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.addWidget(cons_card)
        left.addWidget(rich_card)
        left.addWidget(conv_card)
        left.addStretch(1)
        w = AdaptiveSplitter(900, sizes_h=(620, 480))
        w.addWidget(scroll_area(left_w))
        w.addWidget(scroll_area(right_w))
        return w

    def _conservation_card(self) -> Card:
        """Tabla de conservación de las simulaciones, en vivo"""
        self.cons_table = DataTable(["", "Esquema", "Δt", "ΔT", "Δ‖L‖", "pasos"])
        cons_card = Card("Conservación", plain_body(
            self.cons_table,
            muted_label("ΔT y Δ‖L‖: deriva de la energía cinética y del módulo del momento angular respecto a sus "
                        "valores iniciales (la solución exacta los conserva).")), "en vivo")
        return cons_card

    def _richardson_card(self) -> Card:
        """Orden y error de cada esquema por Richardson (se calcula en un hilo)"""
        self.rich_T = QtWidgets.QDoubleSpinBox()
        self.rich_T.setRange(0.1, 200)
        self.rich_T.setValue(10)
        self.rich_T.setSuffix(" s")
        self.btn_rich = QtWidgets.QPushButton("Calcular")
        self.btn_rich.setToolTip("Integra con n y 2n pasos hasta t_f (n = t_f/Δt) para cada simulación activa")
        self.btn_rich.clicked.connect(self.run_richardson)
        ctl = control_row(QtWidgets.QLabel("Instante final t_f"), self.rich_T, self.btn_rich)
        self.rich_table = DataTable(["", "Esquema", "Δt", "p", "error estimado", "error real"])
        self._rich_note_idle = ("Pulsa «Calcular». Compara ω(t_f) con n y 2n pasos: su diferencia, dividida por 2ᵖ − 1 "
                                "(p = orden teórico del esquema), estima el error de ω (no el de la energía) en la "
                                "solución con 2n pasos. Real: ‖ω − ω exacta‖ de esa solución en t_f (solución de Jacobi). "
                                "El orden observado se mide en «Convergencia frente a Δt».")
        self._rich_note_done = (self._rich_note_idle.replace("Pulsa «Calcular». Compara", "t_f = {T:g} s. Compara")
                                + " En gris: diferencia entre mallas bajo el ruido de redondeo (error estimado no fiable).")
        self.rich_note = muted_label(self._rich_note_idle)
        self.rich_table.hide()
        rich_card = Card("Orden y error del método", plain_body(ctl, self.rich_table, self.rich_note))
        return rich_card

    def _convergence_card(self) -> Card:
        """Tarjeta de convergencia: error final frente al paso, con la pendiente como orden observado"""
        self.conv_levels = QtWidgets.QSpinBox()
        self.conv_levels.setRange(3, 7)
        self.conv_levels.setValue(5)
        self.conv_levels.setToolTip("Δt, Δt/2, Δt/4… hasta t_f (el de arriba): más niveles, más pasos")
        self.btn_conv = QtWidgets.QPushButton("Calcular")
        self.btn_conv.setToolTip("Error en t_f con Δt, Δt/2, Δt/4… para cada simulación activa, y pendiente en "
                                 "log-log. Con esquemas implícitos puede tardar unos segundos.")
        self.btn_conv.clicked.connect(self.run_convergence)
        cctl = control_row(QtWidgets.QLabel("Niveles"), self.conv_levels, self.btn_conv)
        self.plot_conv = pg.PlotWidget()
        self._style_plot(self.plot_conv, "Δt", "error en t_f")
        self.plot_conv.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_conv.setLogMode(True, True)
        self.plot_conv.setMinimumHeight(190)
        self.plot_conv.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.conv_floor = noise_floor_line(self.plot_conv)
        self.conv_lines, self.conv_bad, self.conv_ref = [], [], []
        for k, cfg in enumerate(DEFAULT_CONFIGS):
            self.conv_lines.append(self.plot_conv.plot(
                pen=self._live_pen(k, cfg.color), symbol="o", symbolSize=7, symbolBrush=cfg.color,
                symbolPen=cfg.color, name=cfg.name, connect="finite"))
            self.conv_bad.append(self.plot_conv.plot(           # puntos no fiables: huecos
                pen=None, symbol="o", symbolSize=7, symbolBrush=None, symbolPen=pg.mkPen(cfg.color, width=1.2)))
            self.conv_ref.append(self.plot_conv.plot(
                pen=pg.mkPen(QtGui.QColor(cfg.color).lighter(130), width=1, style=QtCore.Qt.PenStyle.DotLine)))
        self.conv_table = DataTable(["", "Esquema", "p nominal", "p Richardson", "p observada", "pendientes locales"])
        self.conv_table.hide()
        self._conv_note_idle = ("Pulsa «Calcular». Se integra hasta t_f con Δt, Δt/2, Δt/4… y se compara con la "
                                "solución exacta: la pendiente en log-log es el orden p. Los puntos huecos no "
                                "se usan en el ajuste: por debajo del ruido de redondeo (línea discontinua) o "
                                "saturados (error comparable a ‖ω‖). La línea de puntos es la pendiente nominal.")
        self.conv_note = muted_label(self._conv_note_idle)
        conv_card = Card("Convergencia frente a Δt", plain_body(cctl, self.plot_conv, self.conv_table, self.conv_note),
                         "complementa a Richardson")
        return conv_card

    def _equilibria_card(self) -> Card:
        """Equilibrios de la simulación de referencia y efecto de cada esquema sobre ellos"""
        self.eq_table = DataTable(["Eje", "Ω", "Tipo (lineal)", "λ", "No lineal"])
        self.eff_table = DataTable(["", "Esquema", "Rodea", "|R| − 1", "Efecto"])
        eq_card = Card("Equilibrios", plain_body(
            muted_label("Rotaciones puras ω = ±Ω eₖ de la simulación de referencia (Iₖ Ω² = 2T)."),
            self.eq_table,
            muted_label("«Lineal»: lo que dice la Jacobiana (un centro es solo marginal). «No lineal»: con T y ‖L‖² "
                 "los ejes de mayor y menor inercia son estables de verdad y el intermedio es inestable."),
            muted_label("Efecto de cada esquema sobre el centro que rodea su órbita: |R(iβΔt)| − 1 "
                 "mide cuánto crece (+) o decae (−) la amplitud en cada paso."),
            self.eff_table), "ref.")
        self.eq_card = eq_card
        return eq_card

    def _stability_card(self) -> Card:
        """Tarjeta de regiones de estabilidad absoluta y posición de cada simulación en ellas"""
        self.stab_shade = QtWidgets.QComboBox()
        self.stab_shade.addItems([METHOD_SHORT[m] for m in ne.SCHEMES])
        self.stab_shade.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.stab_shade.setMinimumContentsLength(10)
        self.stab_zoom = QtWidgets.QCheckBox("Acercar a los puntos")
        self.stab_zoom.setToolTip("Encuadra los z = λΔt de las simulaciones (con Δt pequeño quedan junto al origen)")
        sctl = control_row(QtWidgets.QLabel("Sombrear"), self.stab_shade, self.stab_zoom)
        self.plot_stab = pg.PlotWidget()
        self._style_plot(self.plot_stab, "Re z", "Im z")
        self.plot_stab.setAspectLocked(True)
        self.plot_stab.setMinimumHeight(300)
        self.plot_stab.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.stab_fill = pg.ImageItem()
        self.plot_stab.addItem(self.stab_fill)
        for ang in (0, 90):
            self.plot_stab.addItem(pg.InfiniteLine(angle=ang, pen=pg.mkPen((255, 255, 255, 70), width=1)),
                                   ignoreBounds=True)
        self.stab_curves = {m: self.plot_stab.plot(pen=pg.mkPen(SCHEME_COLORS[m], width=1.8),
                                                   name=METHOD_SHORT[m], connect="finite") for m in ne.SCHEMES}
        self.stab_points = pg.ScatterPlotItem(size=10)
        self.plot_stab.addItem(self.stab_points)
        self.stab_shade.currentIndexChanged.connect(lambda _: self._update_stability())
        self.stab_zoom.toggled.connect(lambda _: self._update_stability())
        stab_card = Card("Regiones de estabilidad absoluta", plain_body(
            sctl, self.plot_stab,
            muted_label("Contorno |R(z)| = 1 de cada esquema; la zona sombreada es su región |R| ≤ 1. Los puntos son "
                 "z = λΔt de cada simulación (λ: autovalores del equilibrio que rodea su órbita, o del eje "
                 "intermedio si está en la separatriz). Dentro de la región el método numérico no hace crecer la "
                 "amplitud por sí mismo: es una propiedad del esquema y de Δt, no de la estabilidad del problema "
                 "físico (el eje intermedio es inestable aunque z caiga dentro; un buen esquema debe reproducirlo).")),
            "|R(z)| ≤ 1")
        return stab_card

    def _build_header(self) -> QtWidgets.QWidget:
        """Construye la cabecera fija con título y controles de ejecución"""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 8, 0)
        lay.setSpacing(8)
        title = QtWidgets.QLabel("Janibekov Lab")
        title.setObjectName("title")
        sub = QtWidgets.QLabel("Ecuaciones de Euler · teorema del eje intermedio")
        sub.setObjectName("muted")
        lay.addWidget(title)
        lay.addWidget(sub)

        box = QtWidgets.QGroupBox("Simulación")
        g = QtWidgets.QGridLayout(box)
        self.btn_run = QtWidgets.QPushButton("▶  Iniciar")
        self.btn_run.setObjectName("primary")
        self.btn_run.setToolTip("Iniciar / pausar  [Espacio]")
        self.btn_run.clicked.connect(self.toggle_run)
        btn_reset = QtWidgets.QPushButton("⟲  Reiniciar")
        btn_reset.setToolTip("Volver a t = 0  [R]")
        btn_reset.clicked.connect(self.reset)
        g.addWidget(self.btn_run, 0, 0)
        g.addWidget(btn_reset, 0, 1)
        self.speed = FloatSlider("Vel.", 0.2, 30.0, 0.1, 3.0, 1)
        self.speed.spin.setPrefix("×")
        self.speed.setToolTip("Segundos simulados por cada segundo real (aprox.)")
        g.addWidget(self.speed, 1, 0, 1, 2)
        g.addWidget(QtWidgets.QLabel("Parar en"), 2, 0)
        self.stop_T = QtWidgets.QDoubleSpinBox()
        self.stop_T.setRange(0, 5000)
        self.stop_T.setDecimals(1)
        self.stop_T.setSingleStep(5)
        self.stop_T.setSuffix(" s")
        self.stop_T.setSpecialValueText("sin límite")
        self.stop_T.setKeyboardTracking(False)
        self.stop_T.setToolTip("Pausa automática al llegar a este instante (0 = sin límite)")
        g.addWidget(self.stop_T, 2, 1)
        btn_help = QtWidgets.QPushButton("?  Ayuda y atajos")
        btn_help.setToolTip("Atajos de teclado y cómo leer los indicadores  [F1]")
        btn_help.clicked.connect(self.show_help)
        g.addWidget(btn_help, 3, 0, 1, 2)
        lay.addWidget(box)
        return w

    def _build_controls(self) -> QtWidgets.QWidget:
        """Construye el panel izquierdo: escenarios, física, esquemas y previsualización"""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 8, 4)
        lay.setSpacing(10)

        # ---- escenarios
        box = QtWidgets.QGroupBox("Escenarios predefinidos")
        v = QtWidgets.QVBoxLayout(box)
        h = QtWidgets.QHBoxLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(SCENARIOS))
        self.preset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(16)
        btn_apply = QtWidgets.QPushButton("Aplicar")
        btn_apply.clicked.connect(lambda: self.apply_preset(self.preset_combo.currentText()))
        h.addWidget(self.preset_combo, 1)
        h.addWidget(btn_apply)
        v.addLayout(h)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setObjectName("muted")
        self.preset_note.setWordWrap(True)
        self.preset_note.setStyleSheet("padding-bottom: 5px;")   # los subíndices bajan de la línea
        v.addWidget(self.preset_note)
        self.preset_combo.currentTextChanged.connect(
            lambda t: self.preset_note.setText(SCENARIOS[t].note if t in SCENARIOS else ""))
        self.preset_note.setText(SCENARIOS[self.preset_combo.currentText()].note)
        lay.addWidget(box)

        # ---- problema físico compartido y esquema numérico de cada simulación
        base = DEFAULT_CONFIGS[0]
        self.physics = PhysicsSliders(base.I, base.w0)
        self.physics.changed.connect(self._schedule_apply)
        lay.addWidget(_group_box("Problema físico · común a las simulaciones", [self.physics]))

        caption = QtWidgets.QLabel("Cada simulación elige su esquema; ✎ marca las que usan física propia.")
        caption.setObjectName("muted")
        caption.setWordWrap(True)
        lay.addWidget(caption)
        self.tabs = QtWidgets.QTabWidget()
        self.panels = []
        for k, cfg in enumerate(DEFAULT_CONFIGS):
            p = SimConfigPanel(k, cfg, self.physics)
            p.changed.connect(self._schedule_apply)
            self.panels.append(p)
            self.tabs.addTab(p, f"● {cfg.name}")
            self.tabs.tabBar().setTabTextColor(k, QtGui.QColor(cfg.color))
        lay.addWidget(self.tabs)

        # ---- previsualización
        box = QtWidgets.QGroupBox("Previsualización")
        g = QtWidgets.QGridLayout(box)
        btn_prev = QtWidgets.QPushButton("↻  Calcular")
        btn_prev.setObjectName("accent")
        btn_prev.setToolTip("Recalcula la trayectoria completa hasta el horizonte  [P]")
        btn_prev.clicked.connect(self.compute_preview)
        g.addWidget(btn_prev, 0, 0)
        hz = QtWidgets.QHBoxLayout()
        hz.addWidget(QtWidgets.QLabel("Horizonte"))
        self.preview_T = QtWidgets.QDoubleSpinBox()
        self.preview_T.setRange(1, 2000)
        self.preview_T.setValue(100)
        self.preview_T.setSuffix(" s")
        self.preview_T.setKeyboardTracking(False)
        self.preview_T.valueChanged.connect(self._on_horizon_changed)
        hz.addWidget(self.preview_T, 1)
        g.addLayout(hz, 0, 1)
        self.chk_show_prev = QtWidgets.QCheckBox("Mostrar")
        self.chk_show_prev.setChecked(True)
        self.chk_show_prev.setToolTip("Dibuja la trayectoria prevista (discontinua) en las vistas")
        self.chk_show_prev.toggled.connect(self._update_preview_items)
        self.chk_auto = QtWidgets.QCheckBox("Automática")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip("Recalcula sola al cambiar cualquier parámetro")
        g.addWidget(self.chk_show_prev, 1, 0)
        g.addWidget(self.chk_auto, 1, 1)
        self.preview_status = QtWidgets.QLabel("")
        self.preview_status.setObjectName("muted")
        self.preview_status.setWordWrap(True)
        g.addWidget(self.preview_status, 2, 0, 1, 2)
        lay.addWidget(box)
        lay.addStretch(1)
        return w

    def _build_phase_items(self) -> None:
        """Crea los elementos 3D de la vista de fases y la rejilla de la vista del cuerpo"""
        v = self.phase_view
        self.phase_axes = []
        self.phase_axis_labels = []
        for k in range(3):
            line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(AXIS_COLORS[k], 0.8),
                                     width=1.5, antialias=True, mode="lines")
            v.addItem(line)
            lbl = gl.GLTextItem(text=f"ω{'₁₂₃'[k]}", color=QtGui.QColor(AXIS_COLORS[k]),
                                font=QtGui.QFont(UI_FONT, 13, QtGui.QFont.Weight.Bold))
            v.addItem(lbl)
            self.phase_axes.append(line)
            self.phase_axis_labels.append(lbl)

        sphere = gl.MeshData.sphere(rows=36, cols=72)
        self.mesh_E = gl.GLMeshItem(meshdata=sphere, smooth=True, color=(0.25, 0.85, 0.55, 0.10),
                                    shader="balloon", glOptions="additive")
        self.mesh_L = gl.GLMeshItem(meshdata=sphere, smooth=True, color=(0.95, 0.35, 0.75, 0.10),
                                    shader="balloon", glOptions="additive")
        v.addItem(self.mesh_E)
        v.addItem(self.mesh_L)

        self.eq_points = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=14, pxMode=True)
        self.eq_points.setGLOptions("translucent")
        v.addItem(self.eq_points)

        # órbita exacta por simulación y marcador de dónde debería estar ahora
        self.exact_lines, self.exact_heads, self._exact_show = [], [], [False] * len(DEFAULT_CONFIGS)
        for _ in DEFAULT_CONFIGS:
            el = gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=2, cols=3), smooth=True,
                               color=(1.0, 0.84, 0.25, 1.0), shader=None, glOptions="opaque")
            eh = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=20, pxMode=True)
            eh.setGLOptions("translucent")
            for it in (el, eh):
                it.hide()
                v.addItem(it)
            self.exact_lines.append(el)
            self.exact_heads.append(eh)

        self.prev_lines, self.trail_lines, self.heads = [], [], []
        for cfg in DEFAULT_CONFIGS:
            pl = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(cfg.color, 0.35),
                                   width=1.0, antialias=True)
            tl = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(cfg.color, 1.0),
                                   width=2.5, antialias=True)
            hd = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=rgba(cfg.color, 1.0, 1.3),
                                      size=12, pxMode=True)
            for it in (pl, tl, hd):
                it.setGLOptions("translucent")
                it.hide()
                v.addItem(it)
            self.prev_lines.append(pl)
            self.trail_lines.append(tl)
            self.heads.append(hd)

        grid = gl.GLGridItem()
        grid.setSize(20, 10)
        grid.setSpacing(1, 1)
        grid.translate(0, 0, -2.0)
        grid.setColor((170, 170, 175, 90))
        self.sat_view.addItem(grid)
        self.sat_view.setCameraPosition(distance=9, elevation=18, azimuth=-60)

    def _build_plot_items(self) -> None:
        """Crea las curvas de todas las gráficas, los cursores enlazados y los elementos de la carta"""
        self.curves_T, self.curves_L = [], []
        self.prev_T, self.prev_L = [], []
        for k, cfg in enumerate(DEFAULT_CONFIGS):
            for plot, live, prev in ((self.plot_T, self.curves_T, self.prev_T),
                                     (self.plot_L, self.curves_L, self.prev_L)):
                prev.append(plot.plot(pen=self._dash_pen(cfg.color)))
                live.append(plot.plot(pen=self._live_pen(k, cfg.color), name=cfg.name))
        self.curves_w, self.prev_w = [], []
        for k in range(3):
            self.prev_w.append(self.plot_w.plot(pen=self._dash_pen(AXIS_COLORS[k])))
            self.curves_w.append(self.plot_w.plot(pen=pg.mkPen(AXIS_COLORS[k], width=2),
                                                  name=f"ω{k + 1}"))

        self.ew_live, self.ew_prev, self.ew_own = [], [], []
        self.er_live, self.er_prev, self.ep_live, self.ep_prev = [], [], [], []
        self.ew_exact = self.plot_ew.plot(
            pen=pg.mkPen("#FFFFFF", width=2.2, style=QtCore.Qt.PenStyle.DashLine), name="exacta")
        self.ew_exact.setZValue(5)
        for k_, cfg in enumerate(DEFAULT_CONFIGS):
            self.ew_prev.append(self.plot_ew.plot(pen=self._dash_pen(cfg.color), connect="finite"))
            own = self.plot_ew.plot(pen=pg.mkPen(cfg.color, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
            own.setZValue(4)
            self.ew_own.append(own)
            self.ew_live.append(self.plot_ew.plot(pen=self._live_pen(k_, cfg.color), name=cfg.name,
                                                  connect="finite"))
            for plot, live, prev in ((self.plot_er, self.er_live, self.er_prev),
                                     (self.plot_ep, self.ep_live, self.ep_prev)):
                prev.append(plot.plot(pen=self._dash_pen(cfg.color), connect="finite"))
                live.append(plot.plot(pen=self._live_pen(k_, cfg.color), connect="finite"))
        self._ew_cache = None
        self._ew_fit = True
        self._apply_error_axes()
        self.cursor_drift = LinkedCursor([self.plot_T, self.plot_L, self.plot_w], self)
        self.cursor_drift.moved.connect(self._on_drift_cursor)
        self.cursor_err = LinkedCursor([self.plot_ew, self.plot_er, self.plot_ep], self)
        self.cursor_err.moved.connect(self._on_err_cursor)
        self._on_drift_cursor(float("nan"))
        self._on_err_cursor(float("nan"))

        pp = self.plot_plane
        self.plane_domain = pp.plot(pen=pg.mkPen((255, 255, 255, 120), width=1.5,
                                                 style=QtCore.Qt.PenStyle.DotLine))
        self.plane_field = pp.plot(pen=pg.mkPen((200, 200, 205, 120), width=1), connect="pairs")
        self.plane_orbit_items, self.plane_sep_items = [], []
        self.plane_ref_orbit = pp.plot(pen=pg.mkPen(AMBER, width=2,
                                                    style=QtCore.Qt.PenStyle.DashLine),
                                       name="órbita exacta (ref.)")
        self.plane_prev, self.plane_trail, self.plane_head = [], [], []
        for k_, cfg in enumerate(DEFAULT_CONFIGS):
            self.plane_prev.append(pp.plot(pen=self._dash_pen(cfg.color), connect="finite"))
            self.plane_trail.append(pp.plot(pen=self._live_pen(k_, cfg.color, 2.2), connect="finite",
                                            name=cfg.name))
            self.plane_head.append(pp.plot(pen=None, symbol="o", symbolSize=10,
                                           symbolBrush=cfg.color, symbolPen="w"))
        self.plane_eq = pp.plot(pen=None, symbol="star", symbolSize=22, symbolPen="w")
        self.plane_eq.setZValue(10)

    LINE_PATTERNS = (None, [5, 2.5], [1.2, 2.4])      # A continuo, B rayas, C puntos

    @classmethod
    def _live_pen(cls, k: int, color: str, width: float = 2.0) -> QtGui.QPen:
        """Trazo de la simulación k: A continuo, B rayado y C punteado, para distinguirlas sin el color"""
        pen = pg.mkPen(color, width=width)
        pat = cls.LINE_PATTERNS[k % len(cls.LINE_PATTERNS)]
        if pat is not None:
            pen.setStyle(QtCore.Qt.PenStyle.CustomDashLine)
            pen.setDashPattern(pat)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        return pen

    @staticmethod
    def _dash_pen(color: str, alpha: float = 150) -> QtGui.QPen:
        """Trazo discontinuo y semitransparente para las previsualizaciones"""
        c = QtGui.QColor(color)
        c.setAlpha(alpha)
        return pg.mkPen(c, width=1, style=QtCore.Qt.PenStyle.DashLine)

    def _schedule_apply(self, *_) -> None:
        self._param_timer.start()

    def _notice_label(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setObjectName("notice")
        lbl.setWordWrap(True)
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl.setOpenExternalLinks(False)
        lbl.linkActivated.connect(self._on_notice_link)
        lbl.hide()
        return lbl

    def _on_notice_link(self, href: str) -> None:
        if href.startswith("ref:"):
            self.ref_combo.setCurrentIndex(int(href[4:]))

    def _refresh_tab_titles(self) -> None:
        for k, p in enumerate(self.panels):
            own = not p.is_linked()
            self.tabs.setTabText(k, f"● {p.name}" + ("  ✎" if own else ""))
            self.tabs.setTabToolTip(
                k, f"{p.name}: física propia (desvinculada del problema compartido)" if own
                else f"{p.name}: usa el problema físico compartido")

    def _update_surface_notice(self) -> None:
        """Avisa si alguna simulación no yace sobre las superficies de la simulación de referencia"""
        cfgs = self.configs()
        ref_k = self.ref_index()
        ref = cfgs[ref_k]
        rows = []
        for k, c in enumerate(cfgs):
            if k == ref_k or not c.enabled:
                continue
            m = surface_mismatch(ref.I, ref.w0, c.I, c.w0)
            if m is None:
                continue
            why = ("otra I" if not m["same_I"] else
                   f"otra energía ({m['dT']:+.1%}) y otro momento ({m['dL']:+.1%})")
            rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> {why} "
                        f"(<a href='ref:{k}' style='color:{AMBER}'>ver sus superficies</a>)")
        for label, what in ((self.surface_notice, "Las superficies de esta vista"),
                            (self.plane_notice, "La carta y las órbitas exactas")):
            if rows:
                label.setText(f"⚠ {what} son las de la simulación <b>{ref.name}</b> y no "
                              f"contienen exactamente la trayectoria de: " + " · ".join(rows) + ".")
                label.show()
            else:
                label.hide()

    def apply_preset(self, name: str) -> None:
        """Carga un escenario en los paneles y aplica los parámetros"""
        scenario = SCENARIOS[name]
        upd = scenario.sims
        I0, w00 = self.physics.get()
        I0, w00 = tuple(upd[0].get("I", I0)), tuple(upd[0].get("w0", w00))
        self.physics.set(I0, w00)           # el problema físico del escenario es el de A
        for panel, u in zip(self.panels, upd):
            panel.set_config(replace(panel.get_config(), **u))
            own = (tuple(u.get("I", I0)), tuple(u.get("w0", w00))) != (I0, w00)
            panel.set_linked(not own)       # solo se desvincula quien trae otra física
        self.preset_combo.setCurrentText(name)
        self.apply_params()
        if scenario.log_errors:         # errores de órdenes de magnitud distintos: en log se ven todos
            self.chk_log.setChecked(True)

    def configs(self) -> list[SimConfig]:
        return [p.get_config() for p in self.panels]

    def ref_index(self) -> int:
        return self.ref_combo.currentIndex()

    def apply_params(self) -> None:
        """Aplica los parámetros de los paneles y reinicia las simulaciones"""
        self._param_timer.stop()
        cfgs = self.configs()
        changed = self._describe_changes(self._last_cfgs, cfgs)
        old_cfgs, self._last_cfgs = self._last_cfgs, cfgs
        if not cfgs[self.ref_index()].enabled:
            first = next((k for k, c in enumerate(cfgs) if c.enabled), None)
            if first is not None:
                self.ref_combo.blockSignals(True)
                self.ref_combo.setCurrentIndex(first)
                self.ref_combo.blockSignals(False)
        if self._only_scheme_changed(old_cfgs, cfgs):
            self._switch_schemes(cfgs, changed)
            return
        # todas las simulaciones se reinician juntas: así siguen en el mismo instante
        self.sims = [Simulation(c) for c in cfgs]
        self.clock = 0.0
        self.view_t = None
        self._div_notified = set()
        self._pause()
        self._sync_timeline()
        self.previews = {}
        self._update_static_phase()
        self._update_exact_orbits()
        self._update_equilibria()
        self._update_phase_plane()
        self._rebuild_satellites()
        self._refresh_tab_titles()
        self._update_surface_notice()
        self._update_error_notice()
        self._clear_analysis_results()
        self._badge_state = [None] * len(self.badges)
        for b in self.badges:               # si no, quedarían a la vista las fichas de simulaciones desactivadas
            b.hide()
        self.plot_w.setTitle(None)
        self._update_preview_items()
        if self.chk_auto.isChecked():
            self.request_preview()
        else:
            self._set_preview_status("desactualizada · pulsa ↻ Calcular [P]")
        self.status.showMessage(f"Parámetros aplicados · simulación reiniciada (t = 0){changed}", 4000)
        self._refresh_views()

    @staticmethod
    def _only_scheme_changed(old: list[SimConfig], new: list[SimConfig]) -> bool:
        """Indica si solo cambió el esquema numérico, caso en que no hace falta reiniciar"""
        if not old or len(old) != len(new):
            return False
        return (all(replace(a, method=b.method) == b for a, b in zip(old, new))
                and any(a.method != b.method for a, b in zip(old, new)))

    def _switch_schemes(self, cfgs: list[SimConfig], changed: str) -> None:
        """Cambia de integrador al vuelo, conservando el tiempo y el historial"""
        for k, (sim, c) in enumerate(zip(self.sims, cfgs)):
            if sim.cfg.method != c.method:
                sim.switch_method(c.method)
                self._div_notified.discard(k)
        self._update_equilibria()
        self._refresh_tab_titles()
        self._update_error_notice()
        self._clear_analysis_results()
        self._badge_state = [None] * len(self.badges)
        self._rebuild_satellites()
        self.previews = {}
        self._update_preview_items()
        if self.chk_auto.isChecked():
            self.request_preview()
        else:
            self._set_preview_status("desactualizada · pulsa ↻ Calcular [P]")
        self.status.showMessage(f"Esquema cambiado al vuelo (el tiempo continúa){changed}", 4000)
        self._refresh_views()

    @staticmethod
    def _describe_changes(old: list[SimConfig], new: list[SimConfig]) -> str:
        """Describe qué cambió desde la última aplicación, para explicar el reinicio"""
        if not old or len(old) != len(new):
            return ""
        fields = (("enabled", "activación"), ("method", "método"), ("dt", "Δt"), ("I", "I"), ("w0", "Ω₀"))
        out = []
        for a, b in zip(old, new):
            ch = [lab for attr, lab in fields if getattr(a, attr) != getattr(b, attr)]
            if ch:
                out.append(f"{b.name}: {', '.join(ch)}")
        if not out:
            return ""
        return " · cambió " + ("; ".join(out) if len(out) <= 3 else "varias simulaciones")

    def _lin(self, cfg: SimConfig) -> LinearErrors | None:
        """Errores lineales del esquema, con caché porque se consultan en cada refresco"""
        key = (cfg.method, cfg.dt, tuple(cfg.I), tuple(cfg.w0))
        if key not in self._lin_cache:
            if len(self._lin_cache) > 64:
                self._lin_cache.clear()
            self._lin_cache[key] = linear_errors(cfg)
        return self._lin_cache[key]

    def _horizon_steps(self, sim: Simulation) -> int:
        """Pasos hasta el mayor entre el tiempo actual y el horizonte previsto"""
        return ne.n_steps(max(sim.vt, self.preview_T.value()), sim.cfg.dt)

    def _on_horizon_changed(self) -> None:
        if self.chk_auto.isChecked() or self.previews:
            self.request_preview()

    def reset(self) -> None:
        for s in self.sims:
            s.reset()
        self.clock = 0.0
        self.view_t = None
        self._div_notified = set()
        self._ew_cache = None
        self._sync_timeline()
        self._refresh_views()

    def _build_timeline(self) -> QtWidgets.QHBoxLayout:
        """Construye la fila de rebobinado con botones de paso, deslizador y etiqueta de tiempo"""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.btn_back = QtWidgets.QPushButton("◀")
        self.btn_fwd = QtWidgets.QPushButton("▶")
        for b, tip, slot in ((self.btn_back, "Un paso atrás  [,]", self.step_back),
                             (self.btn_fwd, "Un paso adelante  [.]", self.step_forward)):
            b.setFixedWidth(38)
            b.setToolTip(tip + "\nUn paso = el Δt más pequeño de las simulaciones activas")
            b.clicked.connect(slot)
            row.addWidget(b)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 10000)
        self.timeline.setValue(10000)
        self.timeline.setToolTip("Arrastra hacia atrás para revisar lo ya simulado; con Espacio se "
                                 "reproduce hasta el presente y sigue en vivo")
        self.timeline.sliderPressed.connect(self._pause)
        self.timeline.valueChanged.connect(self._on_timeline)
        row.addWidget(self.timeline, 1)
        self.timeline_label = QtWidgets.QLabel("t = 0.00 s")
        self.timeline_label.setObjectName("muted")
        self.timeline_label.setMinimumWidth(120)
        self.timeline_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.timeline_label)
        self.btn_live = QtWidgets.QPushButton("● En vivo")
        self.btn_live.setToolTip("Volver al presente de la simulación")
        self.btn_live.clicked.connect(lambda: self._set_view_time(None))
        row.addWidget(self.btn_live)
        self._syncing = False
        return row

    # ------------------------------------------------------------------ tiempo
    def show_help(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Ayuda · Janibekov Lab")
        dlg.resize(640, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(False)
        view.setHtml(HELP_HTML.format(muted=MUTED, ok=STABILITY_COLORS["estable"], mar=STABILITY_COLORS["marginal"],
                                      bad=STABILITY_COLORS["inestable"], g1=GRADE_COLORS[1], g2=GRADE_COLORS[2],
                                      g0=GRADE_COLORS[0]))
        lay.addWidget(view)
        close = QtWidgets.QPushButton("Cerrar")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def _pause(self, message: str | None = None) -> None:
        self.running = False
        self.btn_run.setText("▶  Iniciar")
        if message:
            self.status.showMessage(message, 8000)

    def toggle_run(self) -> None:
        self.running = not self.running
        self.btn_run.setText("⏸  Pausa" if self.running else "▶  Iniciar")
        if self.running and self.view_t is None:
            stop = self.stop_T.value()
            if stop > 0 and self.clock >= stop - 1e-12:
                self._pause(f"Pausa automática: ya se alcanzó t = {stop:g} s (cambia «Parar en» para seguir)")

    def _set_view_time(self, t: float) -> None:
        """Fija el instante mostrado del historial; None vuelve al presente"""
        if t is None or t >= self.clock - 1e-12:
            self.view_t = None
            for s in self.sims:
                s.view_n = None
        else:
            self.view_t = max(0.0, float(t))
            for s in self.sims:
                s.view_n = int(round(self.view_t / s.cfg.dt)) + 1
        self._sync_timeline()
        self._ew_cache = None
        self._refresh_views()

    def _sync_timeline(self) -> None:
        self._syncing = True
        frac = 1.0 if (self.view_t is None or self.clock <= 0) else self.view_t / self.clock
        self.timeline.setValue(int(round(10000 * min(max(frac, 0.0), 1.0))))
        self._syncing = False
        live = self.view_t is None
        self.btn_live.setEnabled(not live)
        self.timeline.setEnabled(self.clock > 0)
        self.btn_back.setEnabled(self.clock > 0)
        shown = self.clock if live else self.view_t
        self.clock_label.setText(f"t = {self.clock:.2f} s" if live else f"⏪ t = {shown:.2f} s")
        self.timeline_label.setText(f"t = {self.clock:.2f} s" if live
                                    else f"t = {shown:.2f} / {self.clock:.2f} s")

    def _on_timeline(self, v: int) -> None:
        if self._syncing or self.clock <= 0:
            return
        self._pause()
        self._set_view_time(None if v >= 10000 else v / 10000.0 * self.clock)

    def _step_size(self) -> float:
        dts = [s.cfg.dt for s in self.sims if s.cfg.enabled]
        return min(dts) if dts else 0.0

    def step_forward(self) -> None:
        self._pause()
        h = self._step_size()
        if h <= 0:
            return
        if self.view_t is not None:
            self._set_view_time(self.view_t + h)
            return
        self._advance_live(h)
        self._refresh_views()

    def step_back(self) -> None:
        self._pause()
        h = self._step_size()
        if h <= 0 or self.clock <= 0:
            return
        base = self.clock if self.view_t is None else self.view_t
        self._set_view_time(max(0.0, base - h))

    def _advance_live(self, dt_clock: float) -> bool:
        """Avanza el reloj y las simulaciones aplicando la parada, la divergencia y el límite de pasos"""
        stop = self.stop_T.value()
        target = self.clock + dt_clock
        if stop > 0:
            if self.clock >= stop - 1e-12:
                self._pause(f"Pausa automática: ya se alcanzó t = {stop:g} s")
                return False
            target = min(target, stop)
        self.clock = target
        for sim in self.sims:
            if sim.cfg.enabled:
                sim.advance_to(self.clock)
        self._sync_timeline()
        active = [(k, s) for k, s in enumerate(self.sims) if s.cfg.enabled]
        for k, sim in active:
            if sim.diverged and k not in self._div_notified:
                self._div_notified.add(k)       # se congela donde diverge; las demás siguen
                what = (f"falló en t = {sim.t:.2f} s ({sim.error})" if sim.error
                        else f"divergió en t = {sim.t:.2f} s")
                self.status.showMessage(f"Divergencia: {sim.cfg.name} · {METHOD_SHORT[sim.cfg.method]} {what} "
                                        f"(queda congelada; las demás continúan)", 12000)
            elif sim.n_steps >= MAX_LIVE_STEPS:
                self._pause(f"Pausa automática: {sim.cfg.name} alcanzó {MAX_LIVE_STEPS:,} pasos")
        if active and all(s.diverged for _, s in active):
            self._pause("Pausa automática: todas las simulaciones activas divergieron")
        if stop > 0 and self.clock >= stop - 1e-12:
            self._pause(f"Pausa automática: se alcanzó t = {stop:g} s")
        return True


    # ---- análisis pesados: se calculan en un hilo y se pintan al terminar
    def _start_analysis(self, kind: str, job: Callable[[SimConfig], Any], button: QtWidgets.QPushButton) -> None:
        """Lanza un análisis en un hilo, cancelando el anterior del mismo tipo"""
        old = self._analysis.pop(kind, None)
        if old is not None:
            old.cancel()                        # el resultado anterior ya no interesa
        worker = AnalysisWorker(kind, self.configs(), job)
        worker.done.connect(self._on_analysis_done)
        worker.finished.connect(partial(self._worker_finished, worker))
        self._analysis[kind] = worker
        self._workers.append(worker)
        button.setEnabled(False)
        button.setText("⏳ …")
        self.status.showMessage("Calculando…")
        worker.start()

    def _on_analysis_done(self, kind: str, results: dict[int, Any], elapsed: float) -> None:
        """Muestra el resultado de un análisis si sigue vigente"""
        worker = self._analysis.get(kind)
        if worker is None or self.sender() is not worker:       # parámetros que ya no son los actuales
            return
        del self._analysis[kind]
        button = self.btn_rich if kind == "rich" else self.btn_conv
        button.setEnabled(True)
        button.setText("Calcular")
        if kind == "rich":
            self._show_richardson(worker.cfgs, results, elapsed)
        else:
            self._show_convergence(worker.cfgs, results, elapsed)

    def run_richardson(self) -> None:
        T = self.rich_T.value()
        self._rich_T = T
        self._start_analysis("rich", lambda c: richardson_report(c, T), self.btn_rich)

    def run_convergence(self) -> None:
        T, levels = self.rich_T.value(), self.conv_levels.value()
        self._rich_T = T
        self._start_analysis("conv", lambda c: convergence_report(c, T, levels), self.btn_conv)

    def _show_richardson(self, cfgs: list[SimConfig], results: dict[int, Any], elapsed: float) -> None:
        """Rellena la tabla de Richardson con el error estimado de cada simulación"""
        T, rows = self._rich_T, []
        for k, c in enumerate(cfgs):
            r = results.get(k)
            if r is None:
                continue
            head = [dot(c), (METHOD_SHORT[c.method], {}), (f"{c.dt:g}", {"mono": True}),
                    (f"{ne.SCHEMES[c.method].order}", {"mono": True, "right": True})]
            if isinstance(r, Exception):
                head[1] = (METHOD_SHORT[c.method], {"tip": f"{type(r).__name__}: {r}"})
                rows.append(head + [("falló", {"right": True, "color": DataTable.ALERT}), DASH])
                self.status.showMessage(f"Richardson: {c.name} falló · {type(r).__name__}: {r}", 12000)
                continue
            finite = np.isfinite(r["err"])
            w = r["u_extr"][:3]
            tip = (f"n = {r['n']} pasos · ω extrapolada = ({w[0]:+.5f}, {w[1]:+.5f}, {w[2]:+.5f})" if finite
                   else "no estimable (divergencia)")
            if r["noise"]:
                tip = "diferencia entre mallas al nivel del ruido de redondeo: el error estimado no es fiable"
            head[1] = (METHOD_SHORT[c.method], {"tip": tip})
            if r["noise"]:
                err = (f"{r['err']:.2e}", {"mono": True, "right": True, "color": MUTED, "tip": "ruido de redondeo"})
            elif finite:
                err = (f"{r['err']:.2e}", {"mono": True, "right": True})
            else:
                err = ("divergió", {"right": True, "color": DataTable.ALERT})
            rows.append(head + [err, num_or_dash(r["err_real"], "{:.2e}")])
        self.rich_table.set_rows(rows)
        self.rich_table.setVisible(bool(rows))
        self.rich_note.setText(self._rich_note_done.format(T=T) if rows else self._rich_note_idle)
        self.status.showMessage(f"Richardson hasta t_f = {T:g} s calculado en {1e3 * elapsed:.0f} ms", 6000)

    def _show_convergence(self, cfgs: list[SimConfig], results: dict[int, Any], elapsed: float) -> None:
        """Dibuja el estudio de convergencia y rellena su tabla"""
        T, rows, floors = self._rich_T, [], []
        for k, c in enumerate(cfgs):
            lines = (self.conv_lines[k], self.conv_bad[k], self.conv_ref[k])
            r = results.get(k)
            if r is None or isinstance(r, Exception):
                for it in lines:
                    it.setData([], [])
                if r is not None:
                    rows.append([dot(c), METHOD_SHORT[c.method], (f"{ne.SCHEMES[c.method].order}", {"mono": True, "right": True}),
                                 DASH, DASH, (f"falló · {type(r).__name__}: {r}", {"color": DataTable.ALERT})])
                    self.status.showMessage(f"Convergencia: {c.name} falló · {type(r).__name__}: {r}", 12000)
                continue
            res, p_rich = r["study"], r["p_richardson"]
            dts, errs, ok = res["dt"], res["err"], res["ok"]
            good = np.isfinite(errs) & (errs > 0)
            self.conv_lines[k].setData(dts[good & ok], errs[good & ok])
            self.conv_bad[k].setData(dts[good & ~ok], errs[good & ~ok])
            if ok.any():
                i0 = int(np.argmax(ok))
                self.conv_ref[k].setData(dts, errs[i0] * (dts / dts[i0]) ** ne.SCHEMES[c.method].order)
            else:
                self.conv_ref[k].setData([], [])
            floors.append(res["floor"])
            loc = " · ".join("—" if not np.isfinite(v) else f"{v:.2f}" for v in res["local"])
            rows.append([dot(c), METHOD_SHORT[c.method], (f"{ne.SCHEMES[c.method].order}", {"mono": True, "right": True}),
                         num_or_dash(p_rich), num_or_dash(res["p"]), (loc, {"mono": True})])
        for k in range(len(cfgs), len(self.conv_lines)):
            for it in (self.conv_lines[k], self.conv_bad[k], self.conv_ref[k]):
                it.setData([], [])
        if floors:
            self.conv_floor.setPos(math.log10(max(floors)))
        self.conv_floor.setVisible(bool(floors))
        self.conv_table.set_rows(rows)
        self.conv_table.setVisible(bool(rows))
        self.plot_conv.autoRange()
        self.conv_note.setText(self._conv_note_idle if rows else "No hay simulaciones activas.")
        self.status.showMessage(f"Convergencia hasta t_f = {T:g} s calculada en {1e3 * elapsed:.0f} ms", 6000)

    def _clear_analysis_results(self) -> None:
        """Borra los resultados de Richardson y convergencia, que dependen de los parámetros"""
        for kind, button in (("rich", self.btn_rich), ("conv", self.btn_conv)):
            worker = self._analysis.pop(kind, None)
            if worker is not None:                          # cálculo en curso con parámetros que ya no valen
                worker.cancel()
                button.setEnabled(True)
                button.setText("Calcular")
        self.rich_note.setText(self._rich_note_idle)
        self.conv_note.setText(self._conv_note_idle)
        self.rich_table.hide()
        self.conv_table.hide()
        for k in range(len(self.conv_lines)):
            for it in (self.conv_lines[k], self.conv_bad[k], self.conv_ref[k]):
                it.setData([], [])
        self.conv_floor.hide()

    def compute_preview(self) -> None:
        self.request_preview()

    def request_preview(self) -> None:
        """Calcula la trayectoria prevista en un hilo, cancelando el cálculo anterior"""
        for w in self._workers:
            if isinstance(w, PreviewWorker):
                w.cancel()
        self._preview_token += 1
        jobs = [(k, c) for k, c in enumerate(self.configs()) if c.enabled]
        if not jobs:
            self.previews = {}
            self._update_preview_items()
            self._set_preview_status("sin simulaciones activas")
            return
        worker = PreviewWorker(self._preview_token, jobs, self.preview_T.value())
        worker.done.connect(self._on_preview_done)
        worker.finished.connect(partial(self._worker_finished, worker))
        self._workers.append(worker)
        self._set_preview_status("⏳ calculando…")
        worker.start()

    def _worker_finished(self, worker: PreviewWorker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def _on_preview_done(self, token: int, previews: dict[int, dict[str, Any]], elapsed: float) -> None:
        """Guarda la previsualización si no está obsoleta y avisa de errores del esquema"""
        if token != self._preview_token:        # resultado de una versión anterior de los parámetros
            return
        self.previews = previews
        note = f"✔ {self.preview_T.value():g} s calculados en {1e3 * elapsed:.0f} ms"
        if any(p["truncated"] for p in previews.values()):
            note += f" · limitada a {MAX_PREVIEW_STEPS:,} pasos"
        failed = [self.sims[k].cfg.name for k, p in previews.items() if p.get("error") and k < len(self.sims)]
        if failed:
            note += f" · ⚠ error del esquema en {', '.join(failed)}"
            self.status.showMessage("Previsualización: " + "; ".join(
                f"{self.sims[k].cfg.name}: {p['error']}" for k, p in previews.items()
                if p.get("error") and k < len(self.sims)), 12000)
        self._set_preview_status(note)
        self._update_preview_items()

    def _set_preview_status(self, text: str) -> None:
        self.preview_status.setText(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for w in list(self._workers):
            w.cancel()
        for w in list(self._workers):
            w.wait(3000)
        super().closeEvent(event)

    def _update_preview_items(self) -> None:
        """Redibuja las curvas de previsualización en todas las vistas"""
        show = self.chk_show_prev.isChecked()
        log = self.chk_log.isChecked()
        for k in range(3):
            p = self.previews.get(k)
            visible = show and p is not None and len(p["t"]) > 1
            self.prev_lines[k].setVisible(visible)
            if visible:
                self.prev_lines[k].setData(pos=p["W"])
                self.prev_T[k].setData(p["t"], self._fmt(p["dT"], log))
                self.prev_L[k].setData(p["t"], self._fmt(p["dL"], log))
            else:
                self.prev_T[k].setData([], [])
                self.prev_L[k].setData([], [])
        p = self.previews.get(self.ref_index())
        for k in range(3):
            if show and p is not None:
                self.prev_w[k].setData(p["t"], p["W"][:, k])
            else:
                self.prev_w[k].setData([], [])
        self._update_plane_preview()
        self._err_prev_stale = True
        self._ew_cache = None
        self._ew_fit = True
        if self.view_tabs.currentWidget() is self.err_tab:
            self._update_error_preview()

    def _update_plane_preview(self) -> None:
        show = self.chk_show_prev.isChecked()
        for k in range(3):
            p = self.previews.get(k)
            if show and p is not None and len(p["t"]) > 1:
                P = self._project_plane(p["W"])
                self.plane_prev[k].setData(P[:, 0], P[:, 1], connect="finite")
            else:
                self.plane_prev[k].setData([], [])

    def _plane_equilibrium(self) -> tuple[int, int]:
        """Devuelve el eje y el signo del equilibrio de la carta, automáticos o elegidos"""
        cfg = self.configs()[self.ref_index()]
        idx = self.plane_axis.currentIndex()
        self.plane_sign.setEnabled(idx != 0)
        self.plane_sign_lbl.setText("Signo" if idx != 0 else "Signo (auto)")
        if idx == 0:
            return default_chart(cfg.I, cfg.w0)
        return idx - 1, (1 if self.plane_sign.currentIndex() == 0 else -1)

    def _project_plane(self, W: Array) -> Array:
        return project_to_chart(W, self._plane_k, self._plane_sign_val)

    def _update_phase_plane(self) -> None:
        """Redibuja la carta del plano de fases para la simulación de referencia"""
        cfg = self.configs()[self.ref_index()]
        I, w0 = np.asarray(cfg.I, dtype=float), np.asarray(cfg.w0, dtype=float)
        twoT = 2 * float(kinetic_energy(w0, I))
        k, sign = self._plane_equilibrium()
        self._plane_k, self._plane_sign_val = k, sign
        i, j = chart_axes(k)
        sub = "₁₂₃"
        pp = self.plot_plane
        pp.setLabel("bottom", f"ω{sub[i]}", **axis_label_style())
        pp.setLabel("left", f"ω{sub[j]}", **axis_label_style())

        for it in self.plane_orbit_items + self.plane_sep_items:
            pp.removeItem(it)
        self.plane_orbit_items, self.plane_sep_items = [], []
        if twoT <= 0:
            for it in (self.plane_domain, self.plane_field, self.plane_ref_orbit, self.plane_eq):
                it.setData([], [])
            self.plane_badge.setText("Ω₀ = 0: sin rotación")
            self.plane_badge.setStyleSheet(badge_css(STABILITY_COLORS["marginal"]))
            self.plane_info.setText("")
            self._update_plane_preview()
            return

        D = chart_domain(I, twoT, k)
        xm, ym = D[:, 0].max(), D[:, 1].max()
        self.plane_domain.setData(D[:, 0], D[:, 1])

        if self.chk_plane_field.isChecked():
            X, Y, U, V, spacing = reduced_flow_field(I, twoT, k, sign)
            seg = arrow_segments(X, Y, U, V, 0.8 * spacing)
            self.plane_field.setData(seg[:, 0], seg[:, 1], connect="pairs")
        else:
            self.plane_field.setData([], [])

        orbits, seps = phase_plane_orbits(I, twoT, k)
        for P in orbits:
            it = pp.plot(P[:, 0], P[:, 1], pen=pg.mkPen((215, 215, 220, 110), width=1),
                         connect="finite")
            self.plane_orbit_items.append(it)
        for P in seps:
            it = pp.plot(P[:, 0], P[:, 1], pen=pg.mkPen((255, 255, 255, 235), width=2.5),
                         connect="finite")
            self.plane_sep_items.append(it)

        P = chart_reference_orbit(I, w0, k, sign)
        if P is None:
            self.plane_ref_orbit.setData([], [])
        else:
            self.plane_ref_orbit.setData(P[:, 0], P[:, 1], connect="finite")

        e = equilibrium_analysis(I, twoT)[k]
        col = STABILITY_COLORS[e["stability"]]
        self.plane_eq.setData([0.0], [0.0], symbolBrush=col)
        pp.setRange(xRange=(-1.08 * xm, 1.08 * xm), yRange=(-1.08 * ym, 1.08 * ym), padding=0)

        self.plane_badge.setText(f"{'+' if sign > 0 else '−'}Ω e{sub[k]}  ·  "
                                 f"{e['type'].upper()}  ·  {e['stability']}")
        self.plane_badge.setStyleSheet(badge_css(col, 15))
        self._update_plane_info(cfg, I, twoT, k, sign, e)
        self._update_plane_preview()

    def _update_plane_info(self, cfg: SimConfig, I: Array, twoT: float, k: int, sign: int, e: Any) -> None:
        """Escribe el panel de información de la carta y el efecto de cada esquema"""
        sub = "₁₂₃"
        i, j = chart_axes(k)
        Om = math.sqrt(twoT / I[k])
        lam = ", ".join(f"{z.real:+.3f}{z.imag:+.3f}i" for z in e["eigvals"])
        if e["type"] == "centro":
            shape = "elipses cerradas → las órbitas próximas rodean el equilibrio (estable)"
        elif e["type"] == "punto de silla":
            shape = ("hipérbolas → las órbitas se alejan por la variedad inestable "
                     "(efecto Janibekov); las rectas blancas son las separatrices")
        else:
            shape = "dos inercias iguales (sólido simétrico): caso degenerado, sin separatriz"
        rows = [
            f"<b>Ω</b> = {Om:.4f} &nbsp;(I{sub[k]} Ω² = 2T)",
            f"<b>λ</b> = <span style='font-family:{MONO}'>{lam}</span>",
            f"<b>Qué se ve</b><br>{shape}",
            f"<span style='color:{MUTED}'>Trazo continuo: simulación · discontinuo: previsualización · "
            f"amarillo: órbita exacta de ref.</span>",
            "<b>Efecto de cada esquema aquí</b>",
        ]
        for c in self.configs():
            if not c.enabled:
                continue
            head = (f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                    f"{METHOD_SHORT[c.method]} · Δt={c.dt:g}: ")
            amp = equilibrium_amplification(c.I, c.w0, k, ne.SCHEMES[c.method].amplification, c.dt)
            if amp is None:
                rows.append(head + "—")
                continue
            R = amp["R"]
            if amp["type"] == "centro":
                g = R - 1
                what = amplitude_trend(g)
                rows.append(head + f"<span style='font-family:{MONO}'>|R|−1 = {g:+.2e}</span> · amplitud {what}")
            else:
                rows.append(head + f"<span style='font-family:{MONO}'>|R(λΔt)| = {R:.4f}</span>")
        self.plane_info.setText("<br><br>".join(rows))
    def _refresh_phase_plane(self) -> None:
        """Actualiza la estela y la cabeza de cada simulación en la carta"""
        for k, sim in enumerate(self.sims):
            if sim.cfg.enabled and sim.nv > 1:
                W = sim.vY[max(0, sim.nv - self._trail_points(sim)):, :3]
                P = self._project_plane(W)
                self.plane_trail[k].setData(P[:, 0], P[:, 1], connect="finite")
                last = P[-1:]
                if np.isfinite(last).all():
                    self.plane_head[k].setData(last[:, 0], last[:, 1])
                else:
                    self.plane_head[k].setData([], [])
            else:
                self.plane_trail[k].setData([], [])
                self.plane_head[k].setData([], [])

    def _on_ref_changed(self) -> None:
        self._update_static_phase()
        self._update_exact_orbits()
        self._update_equilibria()
        self._update_phase_plane()
        self._update_surface_notice()
        self._update_error_notice()
        self._update_preview_items()
        self._refresh_views()

    def _update_static_phase(self) -> None:
        """Reconstruye elipsoides, separatrices y polhodias de la vista 3D de fases"""
        cfg = self.configs()[self.ref_index()]
        I, w0 = np.array(cfg.I), np.array(cfg.w0)
        semi_E, semi_L = poinsot_semiaxes(I, w0)
        twoT = 2 * float(kinetic_energy(w0, I))

        for mesh, semi in ((self.mesh_E, semi_E), (self.mesh_L, semi_L)):
            md = gl.MeshData.sphere(rows=36, cols=72)
            md.setVertexes(md.vertexes() * np.maximum(semi, 1e-6))
            mesh.setMeshData(meshdata=md)

        for it in self.separatrix_items + self.family_items:
            self.phase_view.removeItem(it)
        self.separatrix_items, self.family_items = [], []
        for P in separatrix_curves(I, twoT):
            it = gl.GLLinePlotItem(pos=P, color=(1, 1, 1, 0.95), width=2.5, antialias=True)
            self.phase_view.addItem(it)
            self.separatrix_items.append(it)
        for P in polhode_family(I, twoT):
            it = gl.GLLinePlotItem(pos=P, color=(0.55, 0.62, 0.78, 0.35), width=1, antialias=True)
            it.setGLOptions("translucent")
            self.phase_view.addItem(it)
            self.family_items.append(it)

        lim = 1.25 * max(np.max(semi_E), np.max(semi_L), 1e-3)
        self._phase_lim = lim
        for k in range(3):
            e = np.zeros(3)
            e[k] = lim
            self.phase_axes[k].setData(pos=np.array([-e, e]))
            self.phase_axis_labels[k].setData(pos=1.08 * e)
        self.phase_view.setCameraPosition(distance=1.9 * lim)
        self.phase_card.subtitle.setText(
            f"ref. {cfg.name}:  2T = {twoT:.4g}   ‖L‖ = {np.sqrt(angular_momentum_sq(w0, I)):.4g}")
        self._update_static_visibility()

    def _update_equilibria(self) -> None:
        """Rellena las tablas de equilibrios y de efecto numérico por esquema"""
        cfgs = self.configs()
        ref_cfg = cfgs[self.ref_index()]
        pos, cols, rows = [], [], []
        for k, e in enumerate(equilibrium_analysis(ref_cfg.I, 2 * float(kinetic_energy(ref_cfg.w0, ref_cfg.I)))):
            col = STABILITY_COLORS[e["stability"]]
            Om = e["omega"][k]
            lam = e["eigvals"][np.argmax(np.abs(e["eigvals"]))]
            lam_txt = "0" if abs(lam) < 1e-12 else (f"±{abs(lam.imag):.4g} i" if abs(lam.imag) > abs(lam.real)
                                                    else f"±{abs(lam.real):.4g}")
            bad = e["stability"] == "inestable"
            rows.append([(f"ω{'₁₂₃'[k]}", {}),
                         (f"±{Om:.4g}", {"mono": True, "right": True}),
                         (f"{e['type']} · {e['stability']}", {"color": DataTable.ALERT if bad else TEXT}),
                         (lam_txt, {"mono": True, "right": True}),
                         (e["nonlinear"], {"color": STABILITY_COLORS[e["nonlinear"]]})])
            pos += [e["omega"], -e["omega"]]
            cols += [rgba(col, 1.0)] * 2
        self.eq_table.set_rows(rows)
        self.eq_card.subtitle.setText(f"ref. {ref_cfg.name}")
        self.eq_points.setData(pos=np.array(pos), color=np.array(cols))

        eff = []
        for c in cfgs:
            if not c.enabled:
                continue
            lin = self._lin(c)
            if lin is None:
                eff.append([dot(c), METHOD_SHORT[c.method], ("—", {"color": MUTED}), ("", {}),
                            ("separatriz o simétrico", {"color": MUTED})])
                continue
            g = lin["dmod"]
            (_, gr), _ = scheme_verdict(lin, ne.n_steps(self.preview_T.value(), c.dt))
            tail = " · ínfimo" if gr == 1 else ""
            if gr == 0:
                txt, alert = "conserva la amplitud (≈ 0)", False
            elif g > 0:
                txt, alert = "la amplitud crece" + tail, gr == 2
            else:
                txt, alert = "la amplitud decae (disipa)" + tail, False
            eff.append([dot(c), METHOD_SHORT[c.method], (f"ω{'₁₂₃'[lin['axis']]}", {}),
                        num(g, "{:+.2e}", alert), (txt, {"color": DataTable.ALERT if alert else TEXT})])
        self.eff_table.set_rows(eff)
        self._update_stability()

    def _update_stability(self) -> None:
        """Dibuja las regiones de estabilidad de los esquemas y los autovalores por el paso"""
        z = [(lam * c.dt, c.color) for c in self.configs() if c.enabled
             for lam in relevant_eigenvalues(c.I, c.w0)]
        view, re, im = stability_window(np.array([v for v, _ in z]), self.stab_zoom.isChecked())
        # la malla es más grande que la vista: con el aspecto fijo se ve más de lo pedido
        for m, curve in self.stab_curves.items():
            curve.setData(*stability_layer(m, re, im)[:2])
        m = list(ne.SCHEMES)[self.stab_shade.currentIndex()]
        _, _, inside, rect = stability_layer(m, re, im)
        img = np.zeros(inside.shape + (4,), dtype=np.ubyte)
        img[inside] = (*QtGui.QColor(SCHEME_COLORS[m]).getRgb()[:3], 50)
        self.stab_fill.setImage(img, autoLevels=False)
        self.stab_fill.setRect(QtCore.QRectF(*rect))
        self.stab_points.setData([dict(pos=(v.real, v.imag), brush=col, pen=pg.mkPen("w", width=0.8))
                                  for v, col in z])
        self.plot_stab.setRange(xRange=view[0], yRange=view[1], padding=0)

    def _update_exact_orbits(self) -> None:
        """Órbita exacta de la referencia y de las simulaciones con otra física (otra I u otra velocidad inicial)"""
        if not self.sims:
            return
        ref_k = self.ref_index()
        ref = self.sims[ref_k]
        for k, sim in enumerate(self.sims):
            own = sim.cfg.enabled and (k == ref_k or not self._same_exact(ref.cfg, sim.cfg))
            P = exact_orbit(sim.exact, 720) if own else None
            self._exact_show[k] = P is not None and len(P) > 1
            if self._exact_show[k]:
                is_ref = k == ref_k
                # tubo (no una línea): el grosor de las líneas GL depende de la tarjeta gráfica
                r = (0.011 if is_ref else 0.0075) * self._phase_lim
                V, F = tube(P[::2], r, n=10, per=1, caps=False)
                self.exact_lines[k].setMeshData(meshdata=gl.MeshData(vertexes=V, faces=F))
                self.exact_lines[k].setColor((1.0, 0.84, 0.25, 1.0) if is_ref else rgba(sim.cfg.color, 1.0))
        self._update_static_visibility()
        self._update_exact_markers()

    def _update_exact_markers(self) -> None:
        """Coloca la cabeza de cada órbita exacta en el instante mostrado"""
        if not self.sims:
            return
        t_now = self.clock if self.view_t is None else self.view_t
        for k, sim in enumerate(self.sims):
            if self._exact_show[k] and self.chk_exact.isChecked():
                self.exact_heads[k].setData(pos=sim.exact.omega(np.array([t_now])))
            else:
                self.exact_heads[k].setData(pos=np.zeros((1, 3)))

    def _update_static_visibility(self) -> None:
        """Aplica las casillas de visibilidad a los elementos estáticos del plano de fases"""
        for k in range(len(self.exact_lines)):
            vis = self._exact_show[k] and self.chk_exact.isChecked()
            self.exact_lines[k].setVisible(vis)
            self.exact_heads[k].setVisible(vis)

        self.mesh_E.setVisible(self.chk_E.isChecked())
        self.mesh_L.setVisible(self.chk_L.isChecked())
        for it in self.separatrix_items:
            it.setVisible(self.chk_sep.isChecked())
        for it in self.family_items:
            it.setVisible(self.chk_fam.isChecked())

    def _rebuild_satellites(self, force: bool = False) -> None:
        """Reconstruye los satélites 3D solo si cambió algo del cuerpo (o se fuerza)"""
        cfgs = [c for c in self.configs() if c.enabled]
        # reconstruir las mallas es lo más caro de aplicar parámetros: solo si algo del cuerpo cambió
        key = (self.shape_combo.currentText(),
               tuple((c.name, c.color, c.method, tuple(c.I)) for c in cfgs))
        if not force and key == self._sat_key and self.sat_items:
            self._update_satellites()
            return
        self._sat_key = key
        for group in self.sat_items:
            for it in group["all"]:
                self.sat_view.removeItem(it)
        self.sat_items = []
        spacing = 3.4
        for i, cfg in enumerate(cfgs):
            k = [c.name for c in DEFAULT_CONFIGS].index(cfg.name)
            offset = np.array([(i - (len(cfgs) - 1) / 2) * spacing, 0.0, 0.0])
            half = body_half_sizes(cfg.I)
            parts, extent = self._make_body(half, cfg.color)
            axes = [gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(AXIS_COLORS[a]),
                                      width=3, antialias=True) for a in range(3)]
            Lvec = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 0.85, 0.2, 1.0),
                                     width=3, antialias=True)
            wvec = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 1, 1, 0.9),
                                     width=2, antialias=True)
            label = gl.GLTextItem(pos=offset + np.array([0, 0, 1.9]),
                                  text=f"{cfg.name} · {METHOD_SHORT[cfg.method]}",
                                  color=QtGui.QColor(cfg.color),
                                  font=QtGui.QFont(UI_FONT, 12, QtGui.QFont.Weight.Bold))
            items = parts + axes + [Lvec, wvec, label]
            for it in items:
                self.sat_view.addItem(it)
            self.sat_items.append(dict(k=k, offset=offset, extent=extent, parts=parts,
                                       axes=axes, L=Lvec, w=wvec, all=items))
        self.sat_view.setCameraPosition(distance=4 + 2.6 * len(cfgs))
        self._update_satellites()

    def _make_body(self, half: ArrayLike, color: str) -> tuple[list[Any], Array]:
        """Crea las mallas del cuerpo según la forma elegida y devuelve su extensión"""
        shape = self.shape_combo.currentText()
        if shape == "Elipsoide":
            md = gl.MeshData.sphere(rows=24, cols=48)
            md.setVertexes(md.vertexes() * half)
            body = gl.GLMeshItem(meshdata=md, smooth=True, color=rgba(color, 1.0),
                                 shader="shaded", glOptions="opaque")
            return [body], half
        if shape == "Paralelepípedo":
            shades = [0.55, 1.15, 0.85]
            return list(box_items(np.zeros(3), half, color, shades)), half
        if shape == "Vaca":
            return cow_items(half, color)
        if shape == "Caza Estelar":
            return fighter_items(half, color)
        if shape == "OVNI":
            return ufo_items(half, color)
        return satellite_items(half, color)

    def _update_satellites(self) -> None:
        """Orienta cada satélite y sus ejes, momento angular y velocidad angular"""
        for g in self.sat_items:
            sim = self.sims[g["k"]] if self.sims else None
            if sim is None:
                continue
            y = np.array(sim.vy)
            R, L_dir, w_dir = attitude_vectors(y[3:], sim.I, y[:3], float(np.linalg.norm(sim.Ls0)),
                                               float(np.linalg.norm(sim.cfg.w0)))
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = g["offset"]
            tr = pg.Transform3D(M)
            for part in g["parts"]:
                part.setTransform(tr)
            o = g["offset"]
            for a in range(3):
                g["axes"][a].setData(pos=np.array([o, o + R[:, a] * (g["extent"][a] + 0.35)]))
            g["L"].setData(pos=np.array([o, o + 1.6 * L_dir]))
            g["w"].setData(pos=np.array([o, o + 1.25 * w_dir]))

    def _tick(self) -> None:
        """Avanza la simulación un fotograma"""
        if not self.running:
            return
        step = self.speed.value() * self.FRAME_MS / 1000.0
        if self.view_t is not None:                 # reproduciendo el historial hasta el presente
            self._set_view_time(self.view_t + step)
            return
        self._advance_live(step)
        self._refresh_views()

    def _trail_points(self, sim: Simulation) -> int:
        """Puntos de la estela: la misma duración en segundos para todas, aunque el paso dt difiera"""
        return max(2, int(round(self.trail_spin.value() / sim.cfg.dt)))

    @staticmethod
    def _fmt(d: Array, log: bool) -> Array:
        return log_safe(d) if log else d

    def _on_log_toggled(self, on: bool) -> None:
        for p in (self.plot_T, self.plot_L):
            p.setLogMode(False, on)
            p.setLabel("left", ("|ΔT|" if p is self.plot_T else "|Δ‖L‖|") if on
                       else ("ΔT" if p is self.plot_T else "Δ‖L‖"), **axis_label_style())
        self._update_preview_items()
        self._refresh_views()

    def _refresh_error_tab(self) -> None:
        """Actualiza las curvas en vivo de la pestaña de error"""
        log = self.chk_log_err.isChecked()
        use_s = self.err_unit.currentIndex() == 1
        comp = self.err_comp.currentIndex()
        for k, sim in enumerate(self.sims):
            if sim.cfg.enabled and sim.nv > 1:
                t = sim.vtimes
                self.ew_live[k].setData(t, sim.vY[:, comp])
                self.er_live[k].setData(t, self._fmt_err(sim.v_er, log))
                self.ep_live[k].setData(t, self._fmt_err(sim.v_el if use_s else sim.v_ep, log))
            else:
                for c in (self.ew_live, self.er_live, self.ep_live):
                    c[k].setData([], [])
        if self._err_prev_stale:
            self._update_error_preview()
        self._update_exact_curves(comp)
        if self.frame % 4 == 0 or not self.running:
            self._update_error_info()
            self._update_error_floor()

    def _update_exact_curves(self, comp: int) -> None:
        """Dibuja la solución exacta de la referencia y de las simulaciones con otra física"""
        if not self.sims:
            return
        show = self.chk_show_prev.isChecked()
        ref_k = self.ref_index()
        ref = self.sims[ref_k]
        t_cut = {}
        for k, sim in enumerate(self.sims):
            horizon = 0.0
            p = self.previews.get(k)
            if show and p is not None and len(p["t"]) > 1:
                horizon = float(p["t"][-1])
            t_cut[k] = max(sim.vt, horizon)
        key = (comp, id(ref.exact), round(t_cut[ref_k], 6), tuple(
            (k, id(s.exact), round(t_cut[k], 6)) for k, s in enumerate(self.sims)))
        # con la estela en vivo el tiempo cambia en cada fotograma: se recalcula siempre
        if ref.cfg.enabled and t_cut[ref_k] > 0:
            live_only = t_cut[ref_k] == ref.vt
            if self._ew_cache is None or self._ew_cache[0] != key or live_only:
                t, y = self._exact_curve(ref.exact, t_cut[ref_k], comp, live_only)
                self.ew_exact.setData(t, y)
                self._ew_cache = (key,)
                if self._ew_fit and len(y) > 1:
                    # escala según la solución exacta: si la numérica diverge se sale del recuadro
                    lo, hi = float(np.min(y)), float(np.max(y))
                    pad = 0.3 * (hi - lo) if hi - lo > 1e-12 else max(1.0, abs(hi))
                    self.plot_ew.setYRange(lo - pad, hi + pad, padding=0)
                    self._ew_fit = False
        else:
            self.ew_exact.setData([], [])
        for k, sim in enumerate(self.sims):
            own = self.ew_own[k]
            if (sim.cfg.enabled and k != ref_k and ref.cfg.enabled and t_cut[k] > 0
                    and not self._same_exact(ref.cfg, sim.cfg)):
                t, y = self._exact_curve(sim.exact, t_cut[k], comp, True)
                own.setData(t, y)
            else:
                own.setData([], [])

    def _update_error_preview(self) -> None:
        """Actualiza las curvas de previsualización de la pestaña de error"""
        self._err_prev_stale = False
        show = self.chk_show_prev.isChecked()
        log = self.chk_log_err.isChecked()
        use_s = self.err_unit.currentIndex() == 1
        comp = self.err_comp.currentIndex()
        for k in range(len(self.ew_prev)):
            p = self.previews.get(k)
            if show and p is not None and len(p["t"]) > 1:
                self.ew_prev[k].setData(p["t"], p["W"][:, comp])
                self.er_prev[k].setData(p["t"], self._fmt_err(p["r"], log))
                self.ep_prev[k].setData(p["t"], self._fmt_err(p["lag"] if use_s else p["psi"], log))
            else:
                for c in (self.ew_prev, self.er_prev, self.ep_prev):
                    c[k].setData([], [])

    def _update_error_info(self) -> None:
        """Compone el texto explicativo con errores medidos y predicción lineal"""
        def tag(g: int) -> str:
            return pill(GRADE_WORDS[g], GRADE_COLORS[g])
        rows = [f"<span style='color:{MUTED}'><b>Módulo r</b>: separación radial respecto a la órbita "
                f"exacta (0 si el punto está sobre ella).<br><b>Fase ψ</b>: cuánto adelanta (+) o "
                f"retrasa (−) respecto a la exacta <i>en el mismo instante</i>.</span><br>"]
        any_row = False
        for k, sim in enumerate(self.sims):
            c = sim.cfg
            if not c.enabled:
                continue
            any_row = True
            ex = sim.exact
            if ex.kind == "jacobi":
                per = (f"periodo {ex.period:.4g} s" if math.isfinite(ex.period) else "no periódica (separatriz)")
                what = f"Jacobi · m = {ex.m:.5f} · {per}"
            elif ex.kind == "simetrico":
                what = f"precesión uniforme · periodo {ex.period:.4g} s"
            else:
                what = "equilibrio: ω constante"
            rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                        f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g}<br>"
                        f"<span style='color:{MUTED}'>exacta: {what}</span>")
            if sim.diverged:
                rows.append(pill("FALLÓ" if sim.error else "DIVERGIÓ", STABILITY_COLORS["inestable"]) +
                            f" <span style='color:{MUTED}'>en t = {sim.t:.2f} s; queda congelada"
                            f"{' · ' + html.escape(sim.error) if sim.error else ''}</span>")
            if sim.nv > 1:
                r, psi, lag = sim.v_er[-1], sim.v_ep[-1], sim.v_el[-1]
                gr, gp = grade_pair(r, psi, sim.v_steps)
                rows.append(f"<span style='font-family:{MONO}'>medido (t={sim.vt:.2f} s)<br>"
                            f"&nbsp;módulo r = {r:+.3e} </span>{tag(gr)}<br>"
                            f"<span style='font-family:{MONO}'>&nbsp;fase ψ = {psi:+.4g}° ({lag:+.3e} s) </span>{tag(gp)}")
            lin = self._lin(c)
            if lin is None:
                rows.append(f"<span style='color:{MUTED}'>sin predicción lineal: la órbita está en (o muy cerca de) la "
                            f"separatriz o el sólido es simétrico; no oscila alrededor de ningún eje estable</span>")
            else:
                n_h = self._horizon_steps(sim)
                (mod, gr), (pha, gp) = scheme_verdict(lin, n_h)
                pr, pp = predicted_errors(lin, n_h)
                o_mod, o_pha = SCHEME_ORDER[c.method]
                rows.append(f"<span style='font-family:{MONO}'>teoría lineal (β = {lin['beta']:.4g}), por paso:<br>"
                            f"&nbsp;módulo |R|−1 = {lin['dmod']:+.2e} → {mod}</span> {tag(gr)}<br>"
                            f"<span style='font-family:{MONO}'>&nbsp;fase arg R − βΔt = {lin['dphase']:+.2e} rad → {pha}</span> {tag(gp)}<br>"
                            f"<span style='font-family:{MONO}'>previsto a t = {n_h * c.dt:.4g} s: r ≈ {pr:+.2e}, ψ ≈ {pp:+.3g}°<br>"
                            f"orden global: módulo {o_mod} · fase {o_pha}</span>")
            rows.append("")
        if not any_row:
            rows.append("Sin simulaciones activas.")
        else:
            rows.append(f"<span style='color:{MUTED}'><b>≈ 0</b>: por debajo del ruido de redondeo "
                        f"(línea discontinua de las gráficas). <b>ínfimo</b>: por encima del redondeo "
                        f"pero |r| &lt; {TINY_R:g} y |ψ| &lt; {TINY_PSI:g}°, invisible en cualquier gráfica. "
                        f"<b>relevante</b>: a tener en cuenta.</span>")
        self.err_info.setText("<br>".join(rows))

    def _refresh_views(self) -> None:
        """Actualiza solo la pestaña visible"""
        cur = self.view_tabs.currentWidget()
        log = self.chk_log.isChecked()
        if cur is self.dyn_tab:
            for k, sim in enumerate(self.sims):
                live = sim.cfg.enabled and sim.nv > 1
                self.trail_lines[k].setVisible(live)
                self.heads[k].setVisible(sim.cfg.enabled)
                if sim.cfg.enabled:
                    W = sim.vY[max(0, sim.nv - self._trail_points(sim)):, :3]
                    if live:
                        self.trail_lines[k].setData(pos=W)
                    self.heads[k].setData(pos=W[-1:])
            self._update_satellites()
            self._update_exact_markers()
        elif cur is self.drift_tab:
            for k, sim in enumerate(self.sims):
                if sim.cfg.enabled and sim.nv > 1:
                    t = sim.vtimes
                    self.curves_T[k].setData(t, self._fmt(sim.vdT, log))
                    self.curves_L[k].setData(t, self._fmt(sim.vdL, log))
                else:
                    self.curves_T[k].setData([], [])
                    self.curves_L[k].setData([], [])
            ref = self.sims[self.ref_index()] if self.sims else None
            if ref is not None and self._w_range_key != id(ref):
                # la velocidad angular no supera el máximo de la exacta: si la numérica explota, se sale del recuadro
                lim = 1.15 * ref.w_bound
                if lim > 0:
                    self.plot_w.setYRange(-lim, lim, padding=0)
                self._w_range_key = id(ref)
            for k in range(3):
                if ref is not None and ref.cfg.enabled and ref.nv > 1:
                    self.curves_w[k].setData(ref.vtimes, ref.vY[:, k])
                else:
                    self.curves_w[k].setData([], [])
        elif cur is self.err_tab:
            self._refresh_error_tab()
        elif cur is self.plane_tab:
            self._refresh_phase_plane()
        if self.frame % 4 == 0 or not self.running:
            self._update_status()
        self.frame += 1

    def _update_status(self) -> None:
        """Actualiza las insignias, la tabla de conservación y la barra de estado"""
        show_detail = self.view_tabs.currentWidget() is self.analysis_tab
        rows = []
        for k, sim in enumerate(self.sims):
            c = sim.cfg
            if not c.enabled:
                self._set_badge(k, None)
                rows.append([dot(c), (f"{c.name} — inactiva", {"color": MUTED}), "", "", "", ""])
                continue
            orbit, okind = orbit_stability(c.I, c.w0)
            lin = self._lin(c)
            if lin is None:
                scheme, s_alert = "sin predicción lineal (separatriz o simétrico)", False
            else:
                (mod, gr_t), (pha, gp_t) = scheme_verdict(lin, self._horizon_steps(sim))
                tg = lambda g: " (ínfimo)" if g == 1 else ""
                scheme, s_alert = f"módulo {mod}{tg(gr_t)} · fase {pha}{tg(gp_t)}", max(gr_t, gp_t) == 2
            if sim.diverged and sim.nv >= sim.n:
                obs, lvl = (f"falló en t = {sim.t:.2f} s · {sim.error}" if sim.error
                            else f"divergió en t = {sim.t:.2f} s"), 3
            elif sim.nv <= 1:
                obs, lvl = "aún sin simular", 0
            else:
                r_, p_ = sim.v_er[-1], sim.v_ep[-1]
                lvl = max(grade_pair(r_, p_, sim.v_steps))
                obs = f"r {r_:+.1e} · ψ {p_:+.1e}° · {GRADE_WORDS[lvl]}"
            self._set_badge(k, (c.name, c.color, METHOD_SHORT[c.method], orbit,
                                STABILITY_COLORS[okind], scheme, s_alert, obs, lvl))
            if show_detail:
                rows.append([dot(c), METHOD_SHORT[c.method], (f"{c.dt:g}", {"mono": True, "right": True}),
                             num(sim.vdT[-1]), num(sim.vdL[-1]),
                             (f"{sim.v_steps}", {"mono": True, "right": True})])
        for k in range(len(self.sims), len(self.badges)):
            self._set_badge(k, None)
        if show_detail:
            self.cons_table.set_rows(rows)
        mensajes_propios = ("Previsualización", "Parámetros", "Richardson", "Convergencia",
                            "Pausa automática", "Divergencia")
        if not self.status.currentMessage().startswith(mensajes_propios):
            self.status.showMessage(
                f"{'▶ en ejecución' if self.running else '⏸ en pausa'}   ·   t = {self.clock:.2f} s"
                f"   ·   [Espacio] iniciar/pausar  [R] reiniciar  [P] recalcular previsualización"
                f"  [,] [.] paso")

    def _set_badge(self, k: int, state: tuple[Any, ...] | None) -> None:
        """Redibuja la insignia solo si su estado cambió"""
        if state == self._badge_state[k]:
            return
        prev = self._badge_state[k]
        self._badge_state[k] = state
        b = self.badges[k]
        if state is None:
            b.hide()
            return
        name, color, method, orbit, col, scheme, s_alert, obs, lvl = state
        amber, neutral = AMBER, "#d3d6db"
        sch_html = f"<span style='color:{amber if s_alert else neutral}'>{scheme}</span>"
        obs_html = f"<span style='color:{amber if lvl >= 2 else neutral}'>{obs}</span>"
        lab = lambda t: f"<span style='color:{MUTED}'>{t}</span>"
        b.setText(f"<span style='color:{color}'>●</span>&nbsp;<b>{name} · {method}</b><br>"
                  f"<span style='font-weight:400; font-size:11px'>{lab('Órbita')} {orbit}<br>"
                  f"{lab('Esquema')} {sch_html}<br>"
                  f"{lab('Observado')} {obs_html}</span>")
        if prev is None or prev[4] != col:
            b.setStyleSheet(badge_css(col, 13))
        b.show()


def main() -> None:
    """Arranca la aplicación"""
    global UI_FONT, MONO
    pg.setConfigOptions(antialias=True, background=VIEW_BG, foreground=TEXT)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    UI_FONT = pick_font(UI_FONT_CANDIDATES)
    MONO = f'"{pick_font(MONO_FONT_CANDIDATES)}", monospace'
    app.setFont(QtGui.QFont(UI_FONT, 10))
    app.setStyleSheet(build_stylesheet(UI_FONT))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

