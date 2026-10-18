"""Definición analítica de las ecuaciones del problema, matriz Jacobiana exacta, invariantes 2T y L²,
geometría del espacio de fases, solución exacta de Euler (funciones elípticas de Jacobi) y
herramientas para medir el error de los esquemas numéricos frente a ella."""


import numpy as np
import math
from functools import lru_cache


def euler_equations(omega, I):
    return np.array([
        (I[1,1] - I[2,2]) / I[0,0] * omega[1] * omega[2],
        (I[2,2] - I[0,0]) / I[1,1] * omega[2] * omega[0],
        (I[0,0] - I[1,1]) / I[2,2] * omega[0] * omega[1],
    ])

def jacobian_matrix(omega, I): 
    return np.array([
        [0, ((I[1,1] - I[2,2]) / I[0,0]) * omega[2], ((I[1,1] - I[2,2]) / I[0,0]) * omega[1]],
        [((I[2,2] - I[0,0]) / I[1,1]) * omega[2], 0,  ((I[2,2] - I[0,0]) / I[1,1]) * omega[0]],
        [((I[0,0] - I[1,1]) / I[2,2])* omega[1], ((I[0,0] - I[1,1]) / I[2,2]) * omega[0], 0],
    ])

def principal_equilibria(I, twoT):
    I = np.asarray(I, dtype=float)
    eq = np.zeros((3, 3))
    for k in range(3):
        eq[k, k] = np.sqrt(twoT / I[k])
    return eq

def classify_equilibrium(eigvals, tol=1e-9):
    lam = np.asarray(eigvals, dtype=complex)
    scale = max(1.0, np.max(np.abs(lam)))
    active = lam[np.abs(lam) > tol * scale]
    if active.size == 0:
        return "degenerado (todos los autovalores nulos)", "marginal"
    re = active.real
    complex_pair = np.any(np.abs(active.imag) > tol * scale)
    pos = np.any(re > tol * scale)
    neg = np.any(re < -tol * scale)
    if pos and neg:
        return "punto de silla", "inestable"
    if pos:
        return ("foco inestable" if complex_pair else "nodo inestable"), "inestable"
    if neg:
        return ("foco estable" if complex_pair else "nodo estable"), "estable"
    return "centro", "marginal"

def equilibrium_analysis(I, twoT):
    I_mat = np.diag(np.asarray(I, dtype=float))
    result = []
    for omega in principal_equilibria(I, twoT):
        lam = np.linalg.eigvals(jacobian_matrix(omega, I_mat))
        kind, stability = classify_equilibrium(lam)
        result.append({"omega": omega, "eigvals": lam, "type": kind, "stability": stability})
    return result

def kinetic_energy(omega, I):
    return 0.5 * np.sum(np.array(I) * np.asarray(omega)**2, axis=-1)

def angular_momentum_sq(omega, I):
    return np.sum((np.array(I) * np.asarray(omega)) ** 2, axis=-1)

# ---------------------------------------------------------------- ampliación: solución exacta, esquemas y error
# Geometría del espacio de fases, solución exacta de Euler (Jacobi), análisis lineal de los
# esquemas, relevancia del error y estudio de convergencia.


# ---------------------------------------------------------------- geometría del espacio de fases


def sorted_axes(I):
    a, b, c = np.argsort(np.asarray(I, dtype=float))
    return int(a), int(b), int(c)


def is_degenerate(I, tol=1e-6):
    Is = np.sort(np.asarray(I, dtype=float))
    return (Is[1] - Is[0] < tol) or (Is[2] - Is[1] < tol)


def poinsot_semiaxes(I, w0):
    I = np.asarray(I, dtype=float)
    twoT = 2 * kinetic_energy(w0, I)
    L = np.sqrt(angular_momentum_sq(w0, I))
    return np.sqrt(twoT / I), L / I


def regime(I, w0):
    if is_degenerate(I):
        return "trompo simétrico"
    I = np.asarray(I, dtype=float)
    _, b, _ = sorted_axes(I)
    twoT = 2 * kinetic_energy(w0, I)
    L2 = angular_momentum_sq(w0, I)
    D = L2 - twoT * I[b]
    if abs(D) < 1e-3 * twoT * I[b]:
        return "≈ separatriz (inestable)"
    return "precesa eje mayor (estable)" if D > 0 else "precesa eje menor (estable)"


def surrounded_axis(I, w0):
    if is_degenerate(I):
        return None
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    twoT = 2 * kinetic_energy(w0, I)
    D = angular_momentum_sq(w0, I) - twoT * I[b]
    if abs(D) < 1e-3 * twoT * I[b]:
        return None
    return c if D > 0 else a


def separatrix_curves(I, twoT, n_pts=300):
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    k = math.sqrt(Ia * (Ib - Ia) / (Ic * (Ic - Ib)))
    s = np.linspace(0, 2 * np.pi, n_pts)
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


def polhode_family(I, twoT, n_levels=5, n_pts=200):
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


def chart_axes(k):
    return (k + 1) % 3, (k + 2) % 3


def reduced_flow(I, twoT, k, sign, X, Y):
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    W = np.zeros((3,) + X.shape)
    W[i], W[j] = X, Y
    W[k] = sign * np.sqrt(np.maximum(twoT - I[i] * X ** 2 - I[j] * Y ** 2, 0) / I[k])
    dW = euler_equations(W, np.diag(I))
    return dW[i], dW[j]


def phase_plane_orbits(I, twoT, k, n_levels=9, n_pts=400):
    if is_degenerate(I) or twoT <= 0:
        return [], []
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    fr = np.linspace(0.12, 1.0, n_levels) ** 2
    qa, qb = a * twoT / I[i], b * twoT / I[j]
    if a * b > 0:
        levels = fr * (qa if abs(qa) > abs(qb) else qb)
        seps = []
    else:
        levels = np.concatenate([fr * qa, fr * qb])
        seps = chart_level_curve(I, twoT, k, 0.0, n_pts)
    orbits = [P for c in levels for P in chart_level_curve(I, twoT, k, c, n_pts)]
    return orbits, seps


def chart_level_curve(I, twoT, k, c, n_pts=400):
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    xm, ym = math.sqrt(twoT / I[i]), math.sqrt(twoT / I[j])

    def clip(x, y):
        P = np.column_stack([x, y]).astype(float)
        P[I[i] * P[:, 0] ** 2 + I[j] * P[:, 1] ** 2 > twoT * (1 + 1e-9)] = np.nan
        return P

    if a * b > 0:
        if c / a <= 0:
            return []
        s = np.linspace(0, 2 * np.pi, n_pts)
        return [clip(math.sqrt(c / a) * np.cos(s), math.sqrt(c / b) * np.sin(s))]

    swap = a < 0
    A, B, XM, YM = (b, a, ym, xm) if swap else (a, b, xm, ym)
    if c > 0:
        r, q = math.sqrt(c / A), math.sqrt(c / -B)
        s = np.linspace(-1, 1, n_pts) * math.asinh(YM / q)
        raw = [(sg * r * np.cosh(s), q * np.sinh(s)) for sg in (1, -1)]
    elif c < 0:
        r, q = math.sqrt(c / B), math.sqrt(-c / A)
        s = np.linspace(-1, 1, n_pts) * math.asinh(XM / q)
        raw = [(q * np.sinh(s), sg * r * np.cosh(s)) for sg in (1, -1)]
    else:
        m = math.sqrt(A / -B)
        u = np.linspace(-XM, XM, n_pts)
        raw = [(u, m * u), (u, -m * u)]
    return [clip(v, u) if swap else clip(u, v) for u, v in raw]


SEP_M = 1.0 - 1e-13          # a partir de aquí m = 1: límite hiperbólico (separatriz)


@lru_cache(maxsize=64)
def _agm(m):
    """Media aritmético-geométrica descendente: listas (a_n, b_n, c_n) hasta c_N ≈ 0.
    Con ella se obtienen K, F, sn, cn y dn sin necesidad de scipy."""
    a, b, c = 1.0, math.sqrt(1.0 - m), math.sqrt(m)
    A, B, C = [a], [b], [c]
    while abs(c) > 1e-16 and len(A) < 40:
        a, b, c = (a + b) / 2, math.sqrt(a * b), (a - b) / 2
        A.append(a), B.append(b), C.append(c)
    return A, B, C


def ellip_K(m):
    return math.inf if m >= SEP_M else math.pi / (2 * _agm(m)[0][-1])


def ellip_F(phi, m):
    """F(φ | m) para φ real cualquiera (continua en φ): inversa de la amplitud de Jacobi."""
    phi = np.asarray(phi, dtype=float)
    if m >= SEP_M:
        return np.arctanh(np.clip(np.sin(phi), -1 + 1e-16, 1 - 1e-16))
    A, B, _ = _agm(m)
    for n in range(len(A) - 1):                 # φ_{n+1} = φ_n + arctan(b_n/a_n · tan φ_n) + kπ
        phi = phi + np.arctan(B[n] / A[n] * np.tan(phi)) + math.pi * np.round(phi / math.pi)
    return phi / (2.0 ** (len(A) - 1) * A[-1])


def jacobi(u, m):
    """(sn, cn, dn)(u | m), 0 ≤ m ≤ 1."""
    u = np.asarray(u, dtype=float)
    if m >= SEP_M:
        sech = 1.0 / np.cosh(u)
        return np.tanh(u), sech, sech
    A, _, C = _agm(m)
    N = len(A) - 1
    phi = 2.0 ** N * A[N] * u
    for n in range(N, 0, -1):
        phi = (phi + np.arcsin(np.clip(C[n] / A[n] * np.sin(phi), -1, 1))) / 2
    sn, cn = np.sin(phi), np.cos(phi)
    return sn, cn, np.sqrt(cn * cn + (1.0 - m) * sn * sn)


class EulerExact:
    """Solución analítica de  I_i ω̇_i = (I_j − I_k) ω_j ω_k  desde ω(0) = w0.

    Toda la solución es  ω[p] = A_p·dn(u),  ω[q] = A_q·cn(u),  ω[r] = A_r·sn(u),  u = u0 + σΩ_u·t,
    con p el eje (mín. o máx.) que nunca cambia de signo, q el otro extremo y r el intermedio.
    El trompo simétrico es el caso m = 0 (dn = 1, sn y cn = seno y coseno): misma fórmula.

    kind: 'equilibrio' (ω constante) | 'trompo' (precesión uniforme) | 'jacobi' (caso general).
    El error de una trayectoria numérica se separa en
      · módulo  r = distancia radial a la órbita exacta (adimensional),
      · fase    ψ = adelanto (+) o retraso (−) a lo largo de la órbita, en grados de ciclo,
        y su equivalente temporal `lag` en segundos."""

    def __init__(self, I, w0):
        self.I, self.w0 = I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
        self.kind, self.m, self.period, self._theta_last = "equilibrio", 0.0, math.inf, None
        wdot = np.asarray(euler_equations(w0, np.diag(I)), dtype=float).reshape(3)
        if np.linalg.norm(wdot) <= 1e-9 * (w0 @ w0):        # reposo, esfera, eje principal, ω ⟂ eje del trompo
            return
        a, b, c = sorted_axes(I)
        tol = 1e-9 * I[c]
        if I[b] - I[a] < tol or I[c] - I[b] < tol:            # trompo simétrico: p = eje distinto
            self.kind = "trompo"
            p = c if I[b] - I[a] < tol else a
            q, r = (p + 1) % 3, (p + 2) % 3
            rho = math.hypot(w0[q], w0[r])
            A = np.array([abs(w0[p]), rho, rho])
            self.omega_u = abs((I[p] - I[q]) / I[q] * w0[p])
        else:
            self.kind = "jacobi"
            p, q = (c, a) if (I * w0) @ (I * w0) >= (I @ (w0 * w0)) * I[b] else (a, c)   # ‖L‖² ≥ 2T·I_b
            r = b
            gp = I[r] * (I[q] - I[r]) / (I[p] * (I[q] - I[p]))   # ω_p² + gp·ω_r² = A_p²  (invariantes,
            gq = I[r] * (I[r] - I[p]) / (I[q] * (I[q] - I[p]))   # ω_q² + gq·ω_r² = A_q²   sin cancelaciones)
            Ap2, Aq2 = w0[p] ** 2 + gp * w0[r] ** 2, w0[q] ** 2 + gq * w0[r] ** 2
            A = np.sqrt([Ap2, Aq2, Aq2 / gq])
            self.m = min(gp * A[2] ** 2 / Ap2, 1.0)
            self.omega_u = abs(I[p] - I[q]) * A[0] * A[1] / (I[r] * A[2])
        m = self.m
        # signos: ω_p nunca cambia; en la separatriz ω_q tampoco (cn = sech > 0 solo cubre una rama)
        A = A * [1.0 if w0[p] >= 0 else -1.0, -1.0 if m >= SEP_M and w0[q] < 0 else 1.0, 1.0]
        self.axes, self.amp = [p, q, r], A
        self.theta0 = math.atan2(w0[r] / A[2], w0[q] / A[1])
        self.u0 = float(ellip_F(self.theta0, m))
        self.K = ellip_K(m)
        self.period = 4 * self.K / self.omega_u
        self._cycle = 90.0 / self.K if math.isfinite(self.K) else math.degrees(1.0)     # ° de ciclo por unidad de u
        # sentido en que avanza u: el que reproduce ω̇(0) (derivadas de dn, cn, sn respecto a u)
        sn, cn = math.sin(self.theta0), math.cos(self.theta0)
        dn = math.sqrt(1.0 - m * sn * sn)
        slope = A * self.omega_u * np.array([-m * sn * cn, -sn * dn, cn * dn])
        self.sigma = 1.0 if slope @ wdot[self.axes] >= 0 else -1.0

    def omega(self, t):
        """ω_exacta(t), forma t.shape + (3,)."""
        t = np.asarray(t, dtype=float)
        if self.kind == "equilibrio":
            return np.broadcast_to(self.w0, t.shape + (3,)).copy()
        sn, cn, dn = jacobi(self.u0 + self.sigma * self.omega_u * t, self.m)
        out = np.empty(t.shape + (3,))
        out[..., self.axes] = self.amp * np.stack([dn, cn, sn], axis=-1)
        return out

    def orbit(self, n=720):
        """Puntos (n, 3) de la órbita de ω en ejes del cuerpo (cerrada si es periódica), o None si es un
        equilibrio. Se parametriza por el ángulo de amplitud φ y no por el tiempo: queda bien resuelta
        aun cerca de la separatriz, donde solo se dibuja la rama que recorre ω."""
        if self.kind == "equilibrio":
            return None
        if self.m >= SEP_M:
            phi = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n + 1)[1:-1]
        else:
            phi = np.linspace(0.0, 2 * math.pi, n + 1)
        out = np.empty((len(phi), 3))
        s = np.sin(phi)
        out[:, self.axes] = self.amp * np.stack([np.sqrt(1.0 - self.m * s * s), np.cos(phi), s], axis=-1)
        return out

    def error_parts(self, t, W, reset=False):
        """(r, psi_deg, lag_s) de la trayectoria numérica W(t) (n, 3). Se llama por tandas
        consecutivas (la fase se desenvuelve de una a otra); `reset=True` empieza de nuevo."""
        t, W = np.asarray(t, dtype=float), np.asarray(W, dtype=float)
        if self.kind == "equilibrio":
            return np.linalg.norm(W - self.w0, axis=1) / max(np.linalg.norm(self.w0), 1e-300), \
                np.zeros(len(t)), np.zeros(len(t))
        if reset:
            self._theta_last = None
        _, q, r = self.axes
        x, y = W[:, q] / self.amp[1], W[:, r] / self.amp[2]
        last = self.theta0 if self._theta_last is None else self._theta_last
        d = (np.diff(np.concatenate([[last], np.arctan2(y, x)])) + math.pi) % (2 * math.pi) - math.pi
        th = last + np.cumsum(d)
        self._theta_last = float(th[-1])
        du = (ellip_F(th, self.m) - self.u0 - self.sigma * self.omega_u * t) * self.sigma   # > 0: adelanto
        return np.hypot(x, y) - 1.0, du * self._cycle, du / self.omega_u


def euler_matches_exact(I):
    """¿Satisface `euler_equations` la ecuación que resuelve EulerExact? Si no (otra convención),
    la solución exacta no sería la de este problema."""
    I = np.asarray(I, dtype=float)
    std = lambda w: (np.roll(I, -1) - np.roll(I, -2)) / I * np.roll(w, -1) * np.roll(w, -2)
    ws = np.random.default_rng(1).normal(size=(3, 3))
    return all(np.allclose(np.reshape(euler_equations(w, np.diag(I)), 3), std(w), rtol=1e-8, atol=1e-10)
               for w in ws)


# ---------------------------------------------------------------- análisis lineal de un esquema


def equilibrium_diagnosis(I, w0, amplification, dt):
    """Equilibrios del cuerpo, eje que rodea la órbita y factor de amplificación |R(iβΔt)| por paso del
    esquema (amplification = su función de amplificación R(z)) en ese centro; amp = None si no hay centro."""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    twoT = 2 * float(kinetic_energy(w0, I))
    eqs = equilibrium_analysis(I, twoT)
    axis = surrounded_axis(I, w0)
    amp = None
    if axis is not None and eqs[axis]["type"] == "centro":
        beta = float(np.max(np.abs(eqs[axis]["eigvals"].imag)))
        amp = abs(amplification(1j * beta * dt))
    return {"eqs": eqs, "axis": axis, "amp": amp}


def linear_step_errors(I, w0, amplification, dt):
    """Error por paso del esquema frente a la solución exacta e^{z} en el centro que rodea la
    órbita (pequeñas oscilaciones): módulo |R(iβΔt)| − 1 y fase arg R − βΔt. None si no hay centro."""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    axis = surrounded_axis(I, w0)
    if axis is None:
        return None
    twoT = 2 * float(kinetic_energy(w0, I))
    e = equilibrium_analysis(I, twoT)[axis]
    if e["type"] != "centro":
        return None
    beta = float(np.max(np.abs(e["eigvals"].imag)))
    if beta <= 0:
        return None
    R = complex(amplification(1j * beta * dt))
    return {"axis": axis, "beta": beta, "R": R, "dmod": abs(R) - 1.0,
            "dphase": float(np.angle(R) - beta * dt), "period": 2 * math.pi / beta}



# ---------------------------------------------------------------- ¿cuándo un error es relevante?


EPS = float(np.finfo(float).eps)
TINY_R, TINY_PSI = 1e-6, 1e-3    # por debajo: invisible en cualquier gráfica (módulo adimensional, fase en °)


def noise_floor(n_steps):
    """Error de módulo por debajo del cual no se distingue del redondeo acumulado tras n pasos."""
    return 50.0 * EPS * math.sqrt(max(n_steps, 1))


def grade(x, floor, tiny):
    """0: ≤ ruido de redondeo · 1: ínfimo (invisible) · 2: a tener en cuenta."""
    ax = abs(x)
    return 0 if ax <= floor else (1 if ax < tiny else 2)


def grade_pair(r, psi, n):
    """Gravedad del error de módulo r y de fase psi (°) tras n pasos."""
    fl = noise_floor(n)
    return grade(r, fl, TINY_R), grade(psi, math.degrees(fl), TINY_PSI)


def predicted_errors(lin, n):
    """(r, psi en °) previstos por la teoría lineal tras n pasos."""
    R = abs(lin["R"])
    r = math.expm1(min(n * math.log(R), 700.0)) if R > 0 else -1.0
    return r, math.degrees(lin["dphase"]) * n



# ---------------------------------------------------------------- comparación de problemas


SURFACE_TOL = 1e-3


def surface_mismatch(ref_cfg, cfg, tol=SURFACE_TOL):
    """None si la trayectoria de cfg yace (a tol) sobre las superficies de ref_cfg
    (elipsoides de energía y momento); si no, dict(same_I, dT, dL) con las diferencias
    relativas de energía y de ‖L‖²."""
    Ir, Ic = np.asarray(ref_cfg.I, dtype=float), np.asarray(cfg.I, dtype=float)
    Tr, Tc = float(kinetic_energy(ref_cfg.w0, Ir)), float(kinetic_energy(cfg.w0, Ic))
    Lr, Lc = (float(angular_momentum_sq(ref_cfg.w0, Ir)),
              float(angular_momentum_sq(cfg.w0, Ic)))

    def rel(a, b):
        return (b / a - 1.0) if a > 0 else (0.0 if b == 0 else math.inf)

    same_I = bool(np.allclose(Ir, Ic, rtol=1e-9, atol=1e-12))
    dT, dL = rel(Tr, Tc), rel(Lr, Lc)
    if same_I and abs(dT) <= tol and abs(dL) <= tol:
        return None
    return {"same_I": same_I, "dT": dT, "dL": dL}


# ---------------------------------------------------------------- convergencia frente a Δt
def convergence_study(I, w0, integrate, T, dt, levels=5):
    """Error en T frente a Δt: ‖ω_num(T) − ω_exacta(T)‖ con Δt₀, Δt₀/2, Δt₀/4, … (Δt₀ = T/round(T/dt)).

    integrate(h, n) debe devolver la trayectoria (n+1, ≥3) con ω en las tres primeras columnas, partiendo
    de ω(0) = w0 con paso h. Complementa a Richardson: éste da un orden local con tres pasos y no necesita
    la solución exacta; aquí se usa la exacta y varios Δt, y se ve dónde el error deja de ser fiable.

    Un punto es fiable si es finito, está por encima del ruido de redondeo y no está saturado (error
    comparable a ‖ω‖). El orden observado p es la pendiente del ajuste en log-log de los fiables."""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    w_ex = EulerExact(I, w0).omega(np.array([float(T)]))[0]
    n0 = max(1, int(round(T / dt)))
    scale = max(float(np.linalg.norm(w0)), 1e-300)
    dts, errs = [], []
    for k in range(levels):
        n, h = n0 * 2 ** k, T / (n0 * 2 ** k)
        with np.errstate(all="ignore"):
            Y = np.asarray(integrate(h, n), dtype=float)
            e = float(np.linalg.norm(Y[-1, :3] - w_ex)) if np.isfinite(Y[-1, :3]).all() else math.nan
        dts.append(h)
        errs.append(e)
    dts, errs = np.array(dts), np.array(errs)
    floor = noise_floor(n0 * 2 ** (levels - 1)) * scale
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(errs) & (errs > floor) & (errs < 0.1 * scale)
    p = float(np.polyfit(np.log(dts[ok]), np.log(errs[ok]), 1)[0]) if ok.sum() >= 2 else math.nan
    local = [math.log(errs[k + 1] / errs[k]) / math.log(dts[k + 1] / dts[k])
             if ok[k] and ok[k + 1] else math.nan for k in range(levels - 1)]
    return {"dt": dts, "err": errs, "floor": floor, "ok": ok, "p": p, "local": np.array(local), "n0": n0}