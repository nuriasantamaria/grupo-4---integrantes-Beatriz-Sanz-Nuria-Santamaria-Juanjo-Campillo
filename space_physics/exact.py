"""Solución exacta de las ecuaciones de Euler. Funciones elípticas de Jacobi (media aritmético-geométrica, sin scipy)
y la clase EulerExact.

    Ecs de Euler -> (dw_3/dt)^2 = f(w_3, 2T, L^2, I) -> Integral elíptica -> Su inversa: funciones de Jacobi 

    w_p = A_p * dn(u, m)    w_q = A_q * cn(u, m)    w_r = A_r * sn(u, m)    con u = u0 + sigma * w_u * t
  
  -> p: eje extremo que rodea la órbita (w_p no cambia de signo), q: el otro extremo, r: el intermedio
  -> A_p, A_q, A_r: amplitudes, fijadas por las dos integrales primeras (2T y |L|^2)
  -> sn y cn: seno y coseno del ángulo de un punto que recorre la circunferencia unidad, dn: su velocidad angular
  -> con m = 0 valen sin, cos y 1 (sólido simétrico)
  -> m en [0, 1]: 0 = giro uniforme, 1 = separatriz (periodo infinito).
  -> w_u: escala de tiempo; periodo = 4 K(m) / w_u, con K la integral elíptica completa
  -> u0: fase inicial (hace que w(0) sea la condición inicial); sigma = +-1: sentido en que avanza u
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from space_physics.model import Array, euler_equations, sorted_axes

SEP_M = 1.0 - 1e-13 # Por encima de este valor se trata como m = 1 (separatriz).


# ================ [1] FUNCIONES ELÍPTICAS DE JACOBI ================
 
def _agm(m: float) -> tuple[list[float], list[float], list[float]]:
    """Medias aritmético-geométricas (a_n, b_n, c_n) hasta c_N ~ 0; son la base de K, F, sn, cn y dn"""
    a, b, c = 1.0, math.sqrt(1.0 - m), math.sqrt(m)
    A, B, C = [a], [b], [c]
    while abs(c) > 1e-16 and len(A) < 40:           # converge en 6 a 9 vueltas; el límite 40 es solo de seguridad
        a, b, c = (a + b) / 2, math.sqrt(a * b), (a - b) / 2
        A.append(a)
        B.append(b)
        C.append(c)
    return A, B, C
 
 
def ellip_K(m: float) -> float:
    """Integral elíptica completa K(m), que es un cuarto de periodo; infinita en la separatriz"""
    return math.inf if m >= SEP_M else math.pi / (2 * _agm(m)[0][-1])
 
 
def ellip_F(phi: ArrayLike, m: float) -> Array:
    """Integral elíptica F(phi | m), continua en phi; es la inversa de la amplitud de Jacobi"""
    phi = np.asarray(phi, dtype=float)
    if m >= SEP_M:
        return np.arctanh(np.clip(np.sin(phi), -1 + 1e-16, 1 - 1e-16))
    A, B, _ = _agm(m)
    for a, b in zip(A[:-1], B[:-1]):                # phi_{n+1} = phi_n + arctan(b_n/a_n * tan(phi_n)) + k*pi
        phi = phi + np.arctan(b / a * np.tan(phi)) + math.pi * np.round(phi / math.pi)
    return phi / (2.0 ** (len(A) - 1) * A[-1])
 
 
def jacobi(u: ArrayLike, m: float) -> tuple[Array, Array, Array]:
    """(sn, cn, dn) de u con parámetro 0 <= m <= 1"""
    u = np.asarray(u, dtype=float)
    if m >= SEP_M:                                  # separatriz: fórmula cerrada
        sech = 1.0 / np.cosh(u)
        return np.tanh(u), sech, sech
    A, _, C = _agm(m)
    N = len(A) - 1
    phi = 2.0 ** N * A[N] * u                       # se parte de phi_N y se desciende hasta la amplitud phi
    for n in range(N, 0, -1):
        phi = (phi + np.arcsin(np.clip(C[n] / A[n] * np.sin(phi), -1, 1))) / 2
    sn, cn = np.sin(phi), np.cos(phi)
    return sn, cn, np.sqrt(cn * cn + (1.0 - m) * sn * sn)
 
 
# ================ [2] SOLUCIÓN EXACTA ================
 
class EulerExact:
    """Solución exacta de Euler desde omega(0) = w0, con dn, cn y sn de Jacobi (m = 0 si es simétrico)
 
    Atributos: kind ('equilibrio' | 'simetrico' | 'jacobi'), m, period (inf en equilibrio y separatriz),
    axes = [p, q, r] (p extremo sin cambio de signo, q el otro extremo, r el intermedio),
    amp = (A_p, A_q, A_r) y omega_u (escala de tiempo)"""
 
    def __init__(self, I: ArrayLike, w0: ArrayLike) -> None:
        """Calcula amplitudes, parámetro m, periodo y fase inicial u0 para momentos I y omega(0) = w0"""
        self.I = np.asarray(I, dtype=float)
        self.w0 = np.asarray(w0, dtype=float)
        self.kind, self.m, self.period = "equilibrio", 0.0, math.inf
        self._theta_last: float | None = None
        wdot = euler_equations(self.w0, self.I)
        if np.linalg.norm(wdot) <= 1e-9 * (self.w0 @ self.w0):      # w' ~ 0: reposo, esfera, eje principal, etc.
            return
 
        # 1) ejes, amplitudes, parámetro m y escala de tiempo según el tipo de sólido
        i_min, i_mid, i_max = sorted_axes(self.I)
        tol = 1e-9 * self.I[i_max]                      # tolerancia relativa propia (is_degenerate usa una absoluta)
        low_equal = self.I[i_mid] - self.I[i_min] < tol
        if low_equal or self.I[i_max] - self.I[i_mid] < tol:
            self.kind = "simetrico"
            axes, A, self.omega_u = self._symmetric(i_max if low_equal else i_min)
        else:
            self.kind = "jacobi"
            axes, A, self.m, self.omega_u = self._general(i_min, i_mid, i_max)
        p, q, r = axes
        m = self.m
 
        # 2) signos: omega[p] no cambia de signo; en la separatriz omega[q] tampoco (cn = sech > 0)
        sign_p = 1.0 if self.w0[p] >= 0 else -1.0
        sign_q = -1.0 if (m >= SEP_M and self.w0[q] < 0) else 1.0
        A = A * np.array([sign_p, sign_q, 1.0])
        self.axes, self.amp = axes, A
 
        # 3) fase inicial y periodo
        self.theta0 = math.atan2(self.w0[r] / A[2], self.w0[q] / A[1])
        self.u0 = float(ellip_F(self.theta0, m))
        self.K = ellip_K(m)
        self.period = 4 * self.K / self.omega_u
        if math.isfinite(self.K):
            self._cycle = 90.0 / self.K            # K equivale a 90 grados de ciclo
        else:
            self._cycle = math.degrees(1.0)        # separatriz: 1 unidad de u es 1 rad
 
        # 4) sigma: sentido de u que reproduce dw/dt en t = 0
        sn, cn = math.sin(self.theta0), math.cos(self.theta0)
        dn = math.sqrt(1.0 - m * sn * sn)
        slope = A * self.omega_u * np.array([-m * sn * cn, -sn * dn, cn * dn])
        self.sigma = 1.0 if slope @ wdot[axes] >= 0 else -1.0
 
    def _symmetric(self, p: int) -> tuple[list[int], Array, float]:
        """(ejes, amplitudes, omega_u) del sólido simétrico; p es el eje distinto de los otros dos"""
        I, w0 = self.I, self.w0
        q, r = (p + 1) % 3, (p + 2) % 3
        rho = math.hypot(w0[q], w0[r])                  # omega_p es constante y (omega_q, omega_r) gira en un círculo
        A = np.array([abs(w0[p]), rho, rho])
        omega_u = abs((I[p] - I[q]) / I[q] * w0[p])
        return [p, q, r], A, omega_u
 
    def _general(self, i_min: int, i_mid: int, i_max: int) -> tuple[list[int], Array, float, float]:
        """(ejes, amplitudes, m, omega_u) del sólido con tres inercias distintas"""
        I, w0 = self.I, self.w0
        L2 = (I * w0) @ (I * w0)
        twoT_Imid = (I @ (w0 * w0)) * I[i_mid]             # |L|^2 sobre la separatriz con esta energía
        if L2 >= twoT_Imid:                                # la órbita rodea el eje mayor
            p, q = i_max, i_min
        else:                                              # la órbita rodea el eje menor
            p, q = i_min, i_max
        r = i_mid
        gp = I[r] * (I[q] - I[r]) / (I[p] * (I[q] - I[p]))   # w_p^2 + gp*w_r^2 = A_p^2
        gq = I[r] * (I[r] - I[p]) / (I[q] * (I[q] - I[p]))   # w_q^2 + gq*w_r^2 = A_q^2
        Ap2, Aq2 = w0[p] ** 2 + gp * w0[r] ** 2, w0[q] ** 2 + gq * w0[r] ** 2
        A = np.sqrt([Ap2, Aq2, Aq2 / gq])
        m = min(gp * A[2] ** 2 / Ap2, 1.0)
        omega_u = abs(I[p] - I[q]) * A[0] * A[1] / (I[r] * A[2])
        return [p, q, r], A, m, omega_u
 
    def omega(self, t: ArrayLike) -> Array:
        """omega exacta en t, de forma t.shape + (3,)"""
        t = np.asarray(t, dtype=float)
        if self.kind == "equilibrio":
            return np.broadcast_to(self.w0, t.shape + (3,)).copy()
        sn, cn, dn = jacobi(self.u0 + self.sigma * self.omega_u * t, self.m)
        out = np.empty(t.shape + (3,))
        out[..., self.axes] = self.amp * np.stack([dn, cn, sn], axis=-1)
        return out
 
    def sample(self, t_end: float, max_points: int = 8000) -> tuple[Array, Array]:
        """Muestrea omega(t) en [0, t_end] con unos 100 puntos por periodo (1500 si no hay periodo)"""
        if t_end <= 0:
            return np.empty(0), np.empty((0, 3))
        if math.isfinite(self.period):
            n = int(min(max_points, max(600, 100 * t_end / self.period)))
        else:
            n = 1500
        t = np.linspace(0.0, t_end, n)
        return t, self.omega(t)

    def orbit(self, n: int = 720) -> Array | None:
        """Puntos (n, 3) de la órbita de omega en ejes del cuerpo; None en equilibrio"""
        if self.kind == "equilibrio":
            return None
        # se parametriza por el ángulo de amplitud, no por el tiempo, para resolver bien cerca de la separatriz
        if self.m >= SEP_M:
            phi = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n + 1)[1:-1]
        else:
            phi = np.linspace(0.0, 2 * math.pi, n + 1)
        out = np.empty((len(phi), 3))
        s = np.sin(phi)
        out[:, self.axes] = self.amp * np.stack([np.sqrt(1.0 - self.m * s * s), np.cos(phi), s], axis=-1)
        return out

    def error_parts(self, t: ArrayLike, W: ArrayLike, reset: bool = False) -> tuple[Array, Array, Array]:
        """(r, psi, lag) de W (n, 3): módulo, fase en grados de ciclo y retraso en s (positivos si adelanta)
 
        Se llama por tandas consecutivas; reset=True reinicia la fase"""
        t, W = np.asarray(t, dtype=float), np.asarray(W, dtype=float)
        if self.kind == "equilibrio":
            modulus = np.linalg.norm(W - self.w0, axis=1) / max(np.linalg.norm(self.w0), 1e-300)
            return modulus, np.zeros(len(t)), np.zeros(len(t))
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