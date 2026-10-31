"""Geometría para dibujar (sin Qt): forma y actitud del cuerpo, elipsoides de Poinsot, polhodias,
separatriz, cartas del plano de fases y puntos de la solución exacta"""

from __future__ import annotations

import math

import numpy as np

from space_physics import (angular_momentum_sq, Array, ArrayLike, euler_equations, EulerExact, is_degenerate,
                           kinetic_energy, SEP_M, sorted_axes, surrounded_axis, Vec)


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
