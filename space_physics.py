"""Capa física del sólido rígido libre: Euler, invariantes, equilibrios y solución exacta

Bloques:
  1. Ecuaciones de Euler, Jacobiana exacta e invariantes (2T y |L|^2)
  2. Momentos de inercia: orden de los ejes y simetrías
  3. Sistema completo (omega y cuaternión) para los integradores
  4. Funciones elípticas de Jacobi (sin scipy)
  5. Solución exacta de las ecuaciones de Euler
  6. Equilibrios, autovalores y estabilidad
  7. Posición de la órbita respecto a la separatriz
  8. Teoría lineal de un esquema numérico en un centro
  9. Umbrales de ruido y error previsto
  10. Estudio de convergencia en dt

Convenios: I = (I1, I2, I3) son los momentos principales (solo euler_equations, jacobian_matrix,
rigid_body_rhs y rigid_body_jacobian reciben la matriz diagonal 3x3). Omega va en ejes principales y una
trayectoria es un array (n, 3). El cuaternión es (w, x, y, z) y twoT = 2T. No depende de numerical_engine"""

from __future__ import annotations

import math
from functools import lru_cache, partial
from typing import Callable, Sequence, TypedDict, Union

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
ArrayLike = npt.ArrayLike
Vec = Union[Array, Sequence[float]]    # omega, I o w0 indexable
Amplification = Callable[[complex], complex]    # R(z) de un esquema


# ============================================================== 1. ecuaciones de Euler e invariantes

def _euler_coefficients(I: Array) -> tuple[float, float, float]:
    """Coeficientes c_i = (I_j - I_k) / I_i de dw_i/dt = c_i * w_j * w_k"""
    I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]
    return (I2 - I3) / I1, (I3 - I1) / I2, (I1 - I2) / I3


def euler_equations(omega: Vec, I: Array) -> Array:
    """Derivada de omega según Euler, con I matriz diagonal 3x3"""
    c1, c2, c3 = _euler_coefficients(I)
    return np.array([c1 * omega[1] * omega[2],
                     c2 * omega[2] * omega[0],
                     c3 * omega[0] * omega[1]])


def jacobian_matrix(omega: Vec, I: Array) -> Array:
    """Jacobiana de Euler respecto a omega, con I matriz diagonal 3x3"""
    c1, c2, c3 = _euler_coefficients(I)
    return np.array([
        [0, c1 * omega[2], c1 * omega[1]],
        [c2 * omega[2], 0, c2 * omega[0]],
        [c3 * omega[1], c3 * omega[0], 0],
    ])


def kinetic_energy(omega: ArrayLike, I: ArrayLike) -> Array | float:
    """T = 1/2 * suma de I_i * w_i^2; omega puede ser (..., 3)"""
    return 0.5 * np.sum(np.array(I) * np.asarray(omega) ** 2, axis=-1)


def angular_momentum_sq(omega: ArrayLike, I: ArrayLike) -> Array | float:
    """|L|^2 = suma de (I_i * w_i)^2; omega puede ser (..., 3)"""
    return np.sum((np.array(I) * np.asarray(omega)) ** 2, axis=-1)


def conserved_quantities(w0: ArrayLike, I: ArrayLike) -> tuple[float, float]:
    """(T, |L|) del estado inicial"""
    return float(kinetic_energy(w0, I)), math.sqrt(float(angular_momentum_sq(w0, I)))


def conservation_drift(W: Array, I: Array, T0: float, L0: float) -> tuple[Array, Array]:
    """(error de T, error de |L|) de la trayectoria W (n, 3) respecto a T0 y L0"""
    return np.asarray(kinetic_energy(W, I)) - T0, np.sqrt(angular_momentum_sq(W, I)) - L0


def omega_bound(I: ArrayLike, w0: ArrayLike) -> float:
    """Cota |omega|^2 <= 2T / I_min sobre la órbita exacta; superarla con creces indica divergencia numérica"""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    twoT = 2.0 * float(kinetic_energy(w0, I))
    bound = math.sqrt(twoT / float(I.min()))
    return max(bound, float(np.linalg.norm(w0)))

# ============================================================== 2. momentos de inercia

def sorted_axes(I: ArrayLike) -> tuple[int, int, int]:
    """Índices de los ejes de menor, intermedio y mayor momento de inercia"""
    a, b, c = np.argsort(np.asarray(I, dtype=float))
    return int(a), int(b), int(c)


def is_degenerate(I: ArrayLike, tol: float = 1e-6) -> bool:
    """True si hay al menos dos momentos iguales (simétrico o esfera)"""
    Is = np.sort(np.asarray(I, dtype=float))
    return bool((Is[1] - Is[0] < tol) or (Is[2] - Is[1] < tol))


# ============================================================== 3. sistema completo (omega y cuaternión)

def rigid_body_rhs(u: Array, t: float, I_mat: Array) -> Array:
    """Campo f(u, t) para u = (w1, w2, w3, q0, q1, q2, q3): Euler y q' = 1/2 * q * (0, w); t no se usa"""
    w1, w2, w3, q0, q1, q2, q3 = u
    dw = euler_equations(u[:3], I_mat)
    return np.array([dw[0], dw[1], dw[2],
                     0.5 * (-q1 * w1 - q2 * w2 - q3 * w3),
                     0.5 * (q0 * w1 + q2 * w3 - q3 * w2),
                     0.5 * (q0 * w2 + q3 * w1 - q1 * w3),
                     0.5 * (q0 * w3 + q1 * w2 - q2 * w1)])


def rigid_body_jacobian(u: Array, t: float, I_mat: Array) -> Array:
    """Jacobiana df/du (7x7) de rigid_body_rhs, con el mismo orden de u; t no se usa"""
    w1, w2, w3, q0, q1, q2, q3 = u
    J = np.zeros((7, 7))
    J[:3, :3] = jacobian_matrix(u[:3], I_mat)
    J[3:, :3] = 0.5 * np.array([[-q1, -q2, -q3], [q0, -q3, q2], [q3, q0, -q1], [-q2, q1, q0]])
    J[3:, 3:] = 0.5 * np.array([[0, -w1, -w2, -w3], [w1, 0, w3, -w2], [w2, -w3, 0, w1], [w3, w2, -w1, 0]])
    return J


def _inertia_matrix(I: ArrayLike) -> Array:
    """diag(I); lanza ValueError si no son tres momentos finitos y positivos"""
    I = np.asarray(I, dtype=float)
    if I.shape != (3,) or not np.all(np.isfinite(I)) or np.any(I <= 0):
        raise ValueError(f"I debe tener tres momentos principales positivos (recibido {I.tolist()})")
    return np.diag(I)


def make_rhs(I: ArrayLike) -> Callable[[Array, float], Array]:
    """f(u, t) del sólido de momentos I, listo para los integradores"""
    return partial(rigid_body_rhs, I_mat=_inertia_matrix(I))


def make_jacobian(I: ArrayLike) -> Callable[[Array, float], Array]:
    """J(u, t) = df/du del sólido I, pareja exacta de make_rhs"""
    return partial(rigid_body_jacobian, I_mat=_inertia_matrix(I))


def initial_state(w0: ArrayLike) -> Array:
    """Estado inicial (w0, q = 1, 0, 0, 0)"""
    return np.array([*np.asarray(w0, dtype=float), 1.0, 0.0, 0.0, 0.0])


# ============================================================== 4. funciones elípticas de Jacobi (sin scipy)

# desde este valor de m se trata como m = 1 (separatriz)
SEP_M = 1.0 - 1e-13


@lru_cache(maxsize=64)
def _agm(m: float) -> tuple[list[float], list[float], list[float]]:
    """Listas (a_n, b_n, c_n) de la media aritmético-geométrica hasta c_N ~ 0, base de K, F, sn, cn y dn"""
    a, b, c = 1.0, math.sqrt(1.0 - m), math.sqrt(m)
    A, B, C = [a], [b], [c]
    while abs(c) > 1e-16 and len(A) < 40:
        a, b, c = (a + b) / 2, math.sqrt(a * b), (a - b) / 2
        A.append(a)
        B.append(b)
        C.append(c)
    return A, B, C


def ellip_K(m: float) -> float:
    """Integral elíptica completa K(m); inf en la separatriz"""
    return math.inf if m >= SEP_M else math.pi / (2 * _agm(m)[0][-1])


def ellip_F(phi: ArrayLike, m: float) -> Array:
    """F(phi | m) para phi real, continua en phi (inversa de la amplitud de Jacobi)"""
    phi = np.asarray(phi, dtype=float)
    if m >= SEP_M:
        return np.arctanh(np.clip(np.sin(phi), -1 + 1e-16, 1 - 1e-16))
    A, B, _ = _agm(m)
    for n in range(len(A) - 1):                 # phi_{n+1} = phi_n + arctan(b_n/a_n * tan(phi_n)) + k*pi
        phi = phi + np.arctan(B[n] / A[n] * np.tan(phi)) + math.pi * np.round(phi / math.pi)
    return phi / (2.0 ** (len(A) - 1) * A[-1])


def jacobi(u: ArrayLike, m: float) -> tuple[Array, Array, Array]:
    """(sn, cn, dn) de u con parámetro 0 <= m <= 1"""
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

# ============================================================== 5. solución exacta de Euler

class EulerExact:
    """Solución exacta de Euler desde omega(0) = w0, con dn, cn y sn de Jacobi (m = 0 si es simétrico)"""

    _theta_last: float | None
    kind: str                # 'equilibrio' | 'simetrico' | 'jacobi'
    m: float                 # parámetro elíptico; m >= SEP_M es la separatriz
    period: float            # inf en equilibrio y separatriz
    axes: list[int]          # [p, q, r]: p extremo sin cambio de signo, q el otro extremo, r el intermedio
    amp: Array               # amplitudes con signo (A_p, A_q, A_r)

    def __init__(self, I: ArrayLike, w0: ArrayLike) -> None:
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
            if L2 >= twoT_Ib:      # la órbita rodea el eje mayor
                p, q = c, a
            else:
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

    def omega(self, t: ArrayLike) -> Array:
        """omega exacta en t, de forma t.shape + (3,)"""
        t = np.asarray(t, dtype=float)
        if self.kind == "equilibrio":
            return np.broadcast_to(self.w0, t.shape + (3,)).copy()
        sn, cn, dn = jacobi(self.u0 + self.sigma * self.omega_u * t, self.m)
        out = np.empty(t.shape + (3,))
        out[..., self.axes] = self.amp * np.stack([dn, cn, sn], axis=-1)
        return out

    def error_parts(self, t: ArrayLike, W: ArrayLike, reset: bool = False) -> tuple[Array, Array, Array]:
        """(r, psi, lag) de W (n, 3): módulo, fase en grados de ciclo y retraso en s (positivos si adelanta)

        Se llama por tandas consecutivas; reset=True reinicia la fase"""
        t, W = np.asarray(t, dtype=float), np.asarray(W, dtype=float)
        if self.kind == "equilibrio":
            r = np.linalg.norm(W - self.w0, axis=1) / max(np.linalg.norm(self.w0), 1e-300)
            return r, np.zeros(len(t)), np.zeros(len(t))
        if reset:
            self._theta_last = None
        _, q, r = self.axes
        x, y = W[:, q] / self.amp[1], W[:, r] / self.amp[2]
        last = self.theta0 if self._theta_last is None else self._theta_last
        angle = np.arctan2(y, x)
        jump = np.diff(np.concatenate([[last], angle]))
        jump = (jump + math.pi) % (2 * math.pi) - math.pi     # cada salto, reducido a [-pi, pi)
        th = last + np.cumsum(jump)                          # ángulo continuo, sin saltos de 2pi
        self._theta_last = float(th[-1])
        du = (ellip_F(th, self.m) - self.u0 - self.sigma * self.omega_u * t) * self.sigma
        return np.hypot(x, y) - 1.0, du * self._cycle, du / self.omega_u


# ============================================================== 6. equilibrios y su estabilidad

class Equilibrium(TypedDict):
    """Equilibrio w = Omega*e_k y su estabilidad, resultado de equilibrium_analysis"""
    omega: Array           # rotación pura, con Omega^2 = 2T/I_k
    eigvals: ComplexArray  # autovalores de la Jacobiana
    type: str              # clasificación lineal
    stability: str         # 'estable' | 'marginal' | 'inestable' (lineal)
    nonlinear: str         # 'estable' | 'inestable' | 'neutro' (con T y |L|^2)


def principal_equilibria(I: ArrayLike, twoT: float) -> Array:
    """Fila k = rotación pura sobre el eje k con energía 2T: omega = sqrt(2T/I_k) * e_k"""
    I = np.asarray(I, dtype=float)
    eq = np.zeros((3, 3))
    for k in range(3):
        eq[k, k] = np.sqrt(twoT / I[k])
    return eq


def analytic_eigenvalues(I: ArrayLike, twoT: float) -> ComplexArray:
    """Autovalores (0, mu, -mu) de la Jacobiana por rotación sobre el eje k (fila k); mu es imaginario en los ejes extremos"""
    I = np.asarray(I, dtype=float)
    lam = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        mu = np.sqrt(complex(twoT / I[k] * (I[k] - I[i]) * (I[j] - I[k]) / (I[i] * I[j])))
        lam[k] = [0, mu, -mu]
    return lam


def classify_equilibrium(eigvals: ArrayLike, tol: float = 1e-9) -> tuple[str, str]:
    """(tipo, estabilidad lineal) según los autovalores de la Jacobiana"""
    lam = np.asarray(eigvals, dtype=complex)
    scale = max(1.0, float(np.max(np.abs(lam))))
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


def _nonlinear_stability(I: Array, k: int) -> str:
    """Estabilidad no lineal del eje k: estable si es extremo estricto, inestable si intermedio, neutro con empates"""
    others = np.delete(I, k)
    if np.all(I[k] > others) or np.all(I[k] < others):
        return "estable"
    return "inestable" if np.min(others) < I[k] < np.max(others) else "neutro"


def equilibrium_analysis(I: ArrayLike, twoT: float) -> list[Equilibrium]:
    """Los tres equilibrios w = Omega_k*e_k con autovalores, tipo y estabilidad; con 2T = 0 la no lineal es neutro"""
    I = np.asarray(I, dtype=float)
    lam, result = analytic_eigenvalues(I, twoT), []
    for k, omega in enumerate(principal_equilibria(I, twoT)):
        kind, stability = classify_equilibrium(lam[k])
        result.append(Equilibrium(omega=omega, eigvals=lam[k], type=kind, stability=stability,
                                  nonlinear="neutro" if twoT <= 0 else _nonlinear_stability(I, k)))
    return result

# ============================================================== 7. posición de la órbita respecto a la separatriz


# si |side| queda por debajo, la órbita se da por sobre la separatriz
SEPARATRIX_TOL = 1e-3


def separatrix_side(I: ArrayLike, w0: ArrayLike) -> float:
    """(|L|^2 - 2T*I_med) / (2T*I_med): > 0 la órbita rodea el eje mayor, < 0 el menor, ~ 0 separatriz"""
    I = np.asarray(I, dtype=float)
    _, b, _ = sorted_axes(I)
    twoT_Ib = 2 * kinetic_energy(w0, I) * I[b]
    return float((angular_momentum_sq(w0, I) - twoT_Ib) / twoT_Ib)


def surrounded_axis(I: ArrayLike, w0: ArrayLike) -> int | None:
    """Eje (mayor o menor inercia) que rodea la órbita; None en separatriz, simétrico o reposo"""
    twoT = 2 * kinetic_energy(w0, I)
    if is_degenerate(I) or twoT <= 0:
        return None
    side = separatrix_side(I, w0)
    if abs(side) < SEPARATRIX_TOL:
        return None
    smallest, _, largest = sorted_axes(I)
    return largest if side > 0 else smallest


# ============================================================== 8. teoría lineal de un esquema en un centro

class LinearErrors(TypedDict):
    """Teoría lineal de un esquema en el centro que rodea la órbita, con z = i*beta*dt"""
    axis: int
    beta: float            # frecuencia de las pequeñas oscilaciones
    R: complex             # R(i*beta*dt)
    dmod: float            # |R| - 1 por paso
    dphase: float          # arg R - beta*dt por paso (rad)
    period: float          # 2*pi/beta


def _center(I: Array, w0: Array) -> tuple[int, float] | None:
    """(eje, beta) del centro que rodea la órbita; None en separatriz, simétrico o sin oscilación"""
    axis = surrounded_axis(I, w0)
    if axis is None:
        return None
    e = equilibrium_analysis(I, 2 * float(kinetic_energy(w0, I)))[axis]
    if e["type"] != "centro":
        return None
    beta = float(np.max(np.abs(e["eigvals"].imag)))
    return (axis, beta) if beta > 0 else None


def relevant_eigenvalues(I: ArrayLike, w0: ArrayLike) -> ComplexArray:
    """Autovalores no nulos del equilibrio que rodea la órbita o, en separatriz o simétrico, del eje intermedio"""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    k = surrounded_axis(I, w0)
    k = sorted_axes(I)[1] if k is None else k
    lam = analytic_eigenvalues(I, 2 * float(kinetic_energy(w0, I)))[k]
    return np.asarray(lam[np.abs(lam) > 1e-12], dtype=complex)


def linear_step_errors(I: ArrayLike, w0: ArrayLike, amplification: Amplification, dt: float) -> LinearErrors | None:
    """Errores por paso de módulo y fase del esquema en el centro que rodea la órbita; None si no hay centro"""
    center = _center(np.asarray(I, dtype=float), np.asarray(w0, dtype=float))
    if center is None:
        return None
    axis, beta = center
    R = complex(amplification(1j * beta * dt))
    return {"axis": axis, "beta": beta, "R": R, "dmod": abs(R) - 1.0,
            "dphase": float(np.angle(R) - beta * dt), "period": 2 * math.pi / beta}


class EquilibriumAmplification(TypedDict):
    """|R(lambda*dt)| de un esquema en un equilibrio, resultado de equilibrium_amplification"""
    type: str              # tipo lineal del equilibrio
    lam: complex           # autovalor de mayor |Im| en un centro, de mayor Re en otro caso
    R: float


def equilibrium_amplification(I: ArrayLike, w0: ArrayLike, k: int, amplification: Amplification,
                              dt: float) -> EquilibriumAmplification | None:
    """|R(lambda*dt)| en el equilibrio k con la energía de w0; None si no hay energía o es simétrico"""
    I = np.asarray(I, dtype=float)
    twoT = 2 * float(kinetic_energy(w0, I))
    if twoT <= 0 or is_degenerate(I):
        return None
    e = equilibrium_analysis(I, twoT)[k]
    ev = e["eigvals"]
    lam = ev[np.argmax(np.abs(ev.imag))] if e["type"] == "centro" else ev[np.argmax(ev.real)]
    return {"type": e["type"], "lam": complex(lam), "R": float(abs(amplification(lam * dt)))}

# ============================================================== 9. umbrales de ruido y error previsto

EPS = float(np.finfo(float).eps)


def noise_floor(n_steps: int) -> float:
    """Error de módulo indistinguible del redondeo acumulado tras n_steps pasos"""
    return 50.0 * EPS * math.sqrt(max(n_steps, 1))


def predicted_errors(lin: LinearErrors, n: int) -> tuple[float, float]:
    """(r, psi en grados) previstos por la teoría lineal tras n pasos"""
    R = abs(lin["R"])
    r = math.expm1(min(n * math.log(R), 700.0)) if R > 0 else -1.0
    return r, math.degrees(lin["dphase"]) * n


# ============================================================== 10. convergencia frente a dt

class Convergence(TypedDict):
    """Resultado de convergence_study"""
    dt: Array                   # dt0, dt0/2, dt0/4, ...
    err: Array                  # error en T para cada dt (NaN si divergió)
    floor: float                # ruido de redondeo
    ok: npt.NDArray[np.bool_]   # puntos fiables: finitos, sobre el ruido y sin saturar
    p: float                    # orden observado, pendiente log-log de los fiables (NaN si hay menos de 2)
    local: Array                # pendiente entre niveles consecutivos


def convergence_study(I: ArrayLike, w0: ArrayLike, integrate: Callable[[float, int], Array], T: float,
                      dt: float, levels: int = 5) -> Convergence:
    """Error en T con dt0 = T/round(T/dt), dt0/2, ...

    integrate(h, n) devuelve la trayectoria (n+1, >=3) con omega en las tres primeras columnas"""
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
    dts_a, errs_a = np.array(dts), np.array(errs)
    floor = noise_floor(n0 * 2 ** (levels - 1)) * scale
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(errs_a) & (errs_a > floor) & (errs_a < 0.1 * scale)
    p = float(np.polyfit(np.log(dts_a[ok]), np.log(errs_a[ok]), 1)[0]) if ok.sum() >= 2 else math.nan
    local = []
    for k in range(levels - 1):
        if ok[k] and ok[k + 1]:
            local.append(math.log(errs_a[k + 1] / errs_a[k]) / math.log(dts_a[k + 1] / dts_a[k]))
        else:
            local.append(math.nan)
    return {"dt": dts_a, "err": errs_a, "floor": floor, "ok": ok, "p": p, "local": np.array(local)}
