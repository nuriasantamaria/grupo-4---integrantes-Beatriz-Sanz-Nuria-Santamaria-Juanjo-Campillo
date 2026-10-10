"""Definición analítica de las ecuaciones del problema,
cómputo de la matriz Jacobiana exacta y
verificación de los invariantes 2T y L2."""

import sympy as sp
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

# ================== Solución exacta del problema ==================
# Solución exacta de las ecuaciones de Euler (sólido libre) y error de módulo y fase.

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
        a, b, c = (int(k) for k in np.argsort(I))
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
