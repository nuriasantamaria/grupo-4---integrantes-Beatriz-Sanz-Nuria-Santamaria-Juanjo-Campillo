"""¿Cuál es la solución exacta? Funciones elípticas de Jacobi (AGM, sin scipy) y la clase EulerExact"""

import math
from functools import lru_cache

import numpy as np

from space_physics.model import euler_equations, sorted_axes

# desde este valor de m se trata como m = 1 (separatriz)
SEP_M = 1.0 - 1e-13


# ============================================================== ELLIPTIC (funciones de Jacobi, sin scipy)

@lru_cache(maxsize=64)
def _agm(m):
    """Listas (a_n, b_n, c_n) de la media aritmético-geométrica hasta c_N ~ 0, base de K, F, sn, cn y dn"""
    a, b, c = 1.0, math.sqrt(1.0 - m), math.sqrt(m)
    A, B, C = [a], [b], [c]
    while abs(c) > 1e-16 and len(A) < 40:
        a, b, c = (a + b) / 2, math.sqrt(a * b), (a - b) / 2
        A.append(a)
        B.append(b)
        C.append(c)
    return A, B, C


def ellip_K(m):
    """Integral elíptica completa K(m); inf en la separatriz"""
    return math.inf if m >= SEP_M else math.pi / (2 * _agm(m)[0][-1])


def ellip_F(phi, m):
    """F(phi | m) para phi real, continua en phi (inversa de la amplitud de Jacobi)"""
    phi = np.asarray(phi, dtype=float)
    if m >= SEP_M:
        return np.arctanh(np.clip(np.sin(phi), -1 + 1e-16, 1 - 1e-16))
    A, B, _ = _agm(m)
    for n in range(len(A) - 1):                 # phi_{n+1} = phi_n + arctan(b_n/a_n * tan(phi_n)) + k*pi
        phi = phi + np.arctan(B[n] / A[n] * np.tan(phi)) + math.pi * np.round(phi / math.pi)
    return phi / (2.0 ** (len(A) - 1) * A[-1])


def jacobi(u, m):
    """(sn, cn, dn) de u con parámetro 0 <= m <= 1"""
    u = np.asarray(u, dtype=float)
    if m >= SEP_M:                              # separatriz: fórmula cerrada
        sech = 1.0 / np.cosh(u)
        return np.tanh(u), sech, sech
    A, _, C = _agm(m)
    N = len(A) - 1
    phi = 2.0 ** N * A[N] * u
    for n in range(N, 0, -1):
        phi = (phi + np.arcsin(np.clip(C[n] / A[n] * np.sin(phi), -1, 1))) / 2
    sn, cn = np.sin(phi), np.cos(phi)
    return sn, cn, np.sqrt(cn * cn + (1.0 - m) * sn * sn)


# ============================================================== EXACT SOLUTION

class EulerExact:
    """Solución exacta de Euler desde omega(0) = w0, con dn, cn y sn de Jacobi (m = 0 si es simétrico)

    Atributos: kind ('equilibrio' | 'simetrico' | 'jacobi'), m, period (inf en equilibrio y separatriz),
    axes = [p, q, r] (p extremo sin cambio de signo, q el otro extremo, r el intermedio) y amp = (A_p, A_q, A_r)"""

    def __init__(self, I, w0):
        """Calcula amplitudes, parámetro m, periodo y fase inicial u0 para momentos I y omega(0) = w0"""
        self.I, self.w0 = I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
        self.kind, self.m, self.period, self._theta_last = "equilibrio", 0.0, math.inf, None
        wdot = np.asarray(euler_equations(w0, np.diag(I)), dtype=float).reshape(3)
        if np.linalg.norm(wdot) <= 1e-9 * (w0 @ w0):        # w' ~ 0: reposo, esfera, eje principal, etc.
            return
        a, b, c = sorted_axes(I)
        tol = 1e-9 * I[c]
        if I[b] - I[a] < tol or I[c] - I[b] < tol:            # sólido simétrico: p = eje distinto
            self.kind = "simetrico"
            p = c if I[b] - I[a] < tol else a
            q, r = (p + 1) % 3, (p + 2) % 3
            rho = math.hypot(w0[q], w0[r])
            A = np.array([abs(w0[p]), rho, rho])
            self.omega_u = abs((I[p] - I[q]) / I[q] * w0[p])
        else:
            self.kind = "jacobi"
            L2 = (I * w0) @ (I * w0)
            twoT_Ib = (I @ (w0 * w0)) * I[b]
            if L2 >= twoT_Ib:                                  # la órbita rodea el eje mayor
                p, q = c, a
            else:                                              # la órbita rodea el eje menor
                p, q = a, c
            r = b
            gp = I[r] * (I[q] - I[r]) / (I[p] * (I[q] - I[p]))   # w_p^2 + gp*w_r^2 = A_p^2
            gq = I[r] * (I[r] - I[p]) / (I[q] * (I[q] - I[p]))   # w_q^2 + gq*w_r^2 = A_q^2
            Ap2, Aq2 = w0[p] ** 2 + gp * w0[r] ** 2, w0[q] ** 2 + gq * w0[r] ** 2
            A = np.sqrt([Ap2, Aq2, Aq2 / gq])
            self.m = min(gp * A[2] ** 2 / Ap2, 1.0)
            self.omega_u = abs(I[p] - I[q]) * A[0] * A[1] / (I[r] * A[2])
        m = self.m
        # omega[p] no cambia de signo; en la separatriz omega[q] tampoco (cn = sech > 0)
        sign_p = 1.0 if w0[p] >= 0 else -1.0
        sign_q = -1.0 if (m >= SEP_M and w0[q] < 0) else 1.0
        A = A * np.array([sign_p, sign_q, 1.0])
        self.axes, self.amp = [p, q, r], A
        self.theta0 = math.atan2(w0[r] / A[2], w0[q] / A[1])
        self.u0 = float(ellip_F(self.theta0, m))
        self.K = ellip_K(m)
        self.period = 4 * self.K / self.omega_u
        if math.isfinite(self.K):
            self._cycle = 90.0 / self.K            # K equivale a 90 grados de ciclo
        else:
            self._cycle = math.degrees(1.0)        # separatriz: 1 unidad de u es 1 rad
        # sigma: sentido de u que reproduce dw/dt en t = 0
        sn, cn = math.sin(self.theta0), math.cos(self.theta0)
        dn = math.sqrt(1.0 - m * sn * sn)
        slope = A * self.omega_u * np.array([-m * sn * cn, -sn * dn, cn * dn])
        self.sigma = 1.0 if slope @ wdot[self.axes] >= 0 else -1.0

    def omega(self, t):
        """omega exacta en t, de forma t.shape + (3,)"""
        t = np.asarray(t, dtype=float)
        if self.kind == "equilibrio":
            return np.broadcast_to(self.w0, t.shape + (3,)).copy()
        sn, cn, dn = jacobi(self.u0 + self.sigma * self.omega_u * t, self.m)
        out = np.empty(t.shape + (3,))
        out[..., self.axes] = self.amp * np.stack([dn, cn, sn], axis=-1)
        return out

    def error_parts(self, t, W, reset=False):
        """(r, psi, lag) de W (n, 3): módulo, fase en grados de ciclo y retraso en s (positivos si adelanta)

        Se llama por tandas consecutivas; reset=True reinicia la fase"""
        t, W = np.asarray(t, dtype=float), np.asarray(W, dtype=float)
        if self.kind == "equilibrio":
            r = np.linalg.norm(W - self.w0, axis=1) / max(np.linalg.norm(self.w0), 1e-300)
            return r, np.zeros(len(t)), np.zeros(len(t))
        if reset:
            self._theta_last = None
        _, q, r = self.axes
        x, y = W[:, q] / self.amp[1], W[:, r] / self.amp[2]    # punto sobre la circunferencia unidad
        last = self.theta0 if self._theta_last is None else self._theta_last
        angle = np.arctan2(y, x)
        th = np.unwrap(np.concatenate([[last], angle]))[1:]      # ángulo continuo, sin saltos de 2pi
        self._theta_last = float(th[-1])
        du = (ellip_F(th, self.m) - self.u0 - self.sigma * self.omega_u * t) * self.sigma
        return np.hypot(x, y) - 1.0, du * self._cycle, du / self.omega_u
