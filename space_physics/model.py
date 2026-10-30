"""¿Cuáles son las ecuaciones? Momentos de inercia, Euler, invariantes T y L² y el campo f(u, t) del sólido

No importa ningún otro módulo del proyecto: es la base sobre la que se construye todo lo demás"""

from typing import Callable, Sequence, Union

import numpy as np
import numpy.typing as npt

# alias de tipos: los usan las clases de equilibria y errors, simulation.py y la GUI
Array = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
ArrayLike = npt.ArrayLike
Vec = Union[Array, Sequence[float]]    # omega, I o w0 indexable
Amplification = Callable[[complex], complex]    # R(z) de un esquema


# ============================================================== INERTIA

def inertia_matrix(I):
    """diag(I); ValueError si no son tres momentos finitos y positivos"""
    I = np.asarray(I, dtype=float)
    if I.shape != (3,) or not np.all(np.isfinite(I)) or not np.all(I > 0):
        raise ValueError(f"I debe tener tres momentos principales positivos (recibido {I.tolist()})")
    return np.diag(I)


def sorted_axes(I):
    """Índices de los ejes de menor, intermedio y mayor momento de inercia"""
    orden = np.argsort(np.asarray(I, dtype=float))
    return int(orden[0]), int(orden[1]), int(orden[2])


def is_degenerate(I, tol=1e-6):
    """True si hay al menos dos momentos iguales (simétrico o esfera)"""
    I1, I2, I3 = np.sort(np.asarray(I, dtype=float))
    return bool(I2 - I1 < tol or I3 - I2 < tol)


# ============================================================== EULER (I es la matriz diagonal)

def euler_equations(omega, I):
    """Derivada de omega según Euler: dw_i/dt = (I_j - I_k) / I_i * w_j * w_k"""
    I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]
    w1, w2, w3 = omega[0], omega[1], omega[2]
    return np.array([(I2 - I3) / I1 * w2 * w3,
                     (I3 - I1) / I2 * w3 * w1,
                     (I1 - I2) / I3 * w1 * w2])


def jacobian_matrix(omega, I):
    """Jacobiana de Euler respecto a omega: fila i = derivadas de dw_i/dt"""
    I1, I2, I3 = I[0, 0], I[1, 1], I[2, 2]
    c1 = (I2 - I3) / I1
    c2 = (I3 - I1) / I2
    c3 = (I1 - I2) / I3
    w1, w2, w3 = omega[0], omega[1], omega[2]
    return np.array([[0.0,     c1 * w3, c1 * w2],
                     [c2 * w3, 0.0,     c2 * w1],
                     [c3 * w2, c3 * w1, 0.0]])


# ============================================================== INVARIANTS (omega puede ser (3,) o (n, 3))

def kinetic_energy(omega, I):
    """T = 1/2 * suma de I_i * w_i^2"""
    omega = np.asarray(omega, dtype=float)
    I = np.asarray(I, dtype=float)
    return 0.5 * np.sum(I * omega ** 2, axis=-1)


def angular_momentum_sq(omega, I):
    """|L|^2 = suma de (I_i * w_i)^2"""
    omega = np.asarray(omega, dtype=float)
    I = np.asarray(I, dtype=float)
    return np.sum((I * omega) ** 2, axis=-1)


def conserved_quantities(w0, I):
    """(T, |L|) del estado inicial: la referencia que debe conservarse"""
    T = float(kinetic_energy(w0, I))
    L = float(np.sqrt(angular_momentum_sq(w0, I)))
    return T, L




def omega_bound(I, w0):
    """Cota de |omega| en la órbita exacta: 2T = Σ Iᵢωᵢ² ≥ I_min·‖ω‖²  →  ‖ω‖ ≤ √(2T/I_min)"""
    I = np.asarray(I, dtype=float)
    twoT = 2.0 * kinetic_energy(w0, I)
    return float(np.sqrt(twoT / np.min(I)))


# ============================================================== RIGID BODY: u = (w1, w2, w3, q0, q1, q2, q3)

def rigid_body_rhs(u, I_mat):
    """Derivada de u: Euler para omega y q' = 1/2 * q ⊗ (0, omega) para el cuaternión"""
    w1, w2, w3, q0, q1, q2, q3 = u
    dw = euler_equations(u[:3], I_mat)
    dq0 = 0.5 * (-q1 * w1 - q2 * w2 - q3 * w3)
    dq1 = 0.5 * (q0 * w1 + q2 * w3 - q3 * w2)
    dq2 = 0.5 * (q0 * w2 + q3 * w1 - q1 * w3)
    dq3 = 0.5 * (q0 * w3 + q1 * w2 - q2 * w1)
    return np.array([dw[0], dw[1], dw[2], dq0, dq1, dq2, dq3])


def rigid_body_jacobian(u, I_mat):
    """Jacobiana 7x7 de rigid_body_rhs, con el mismo orden de u"""
    w1, w2, w3, q0, q1, q2, q3 = u
    J = np.zeros((7, 7))
    J[0:3, 0:3] = jacobian_matrix(u[:3], I_mat)
    J[3:7, 0:3] = 0.5 * np.array([[-q1, -q2, -q3],
                                  [q0, -q3, q2],
                                  [q3, q0, -q1],
                                  [-q2, q1, q0]])
    J[3:7, 3:7] = 0.5 * np.array([[0, -w1, -w2, -w3],
                                  [w1, 0, w3, -w2],
                                  [w2, -w3, 0, w1],
                                  [w3, w2, -w1, 0]])
    return J


def make_rhs(I):
    """f(u, t) del sólido de momentos I, listo para los integradores (I se valida una vez)"""
    I_mat = inertia_matrix(I)

    def f(u, t):                      # t: lo exige el motor; el sólido libre no lo usa
        return rigid_body_rhs(u, I_mat)
    return f


def make_jacobian(I):
    """J(u, t) = df/du del sólido de momentos I, pareja exacta de make_rhs"""
    I_mat = inertia_matrix(I)

    def J(u, t):
        return rigid_body_jacobian(u, I_mat)
    return J


def initial_state(w0):
    """Estado inicial (w0, q = 1, 0, 0, 0): el sólido empieza sin girar"""
    return np.array([*np.asarray(w0, dtype=float), 1.0, 0.0, 0.0, 0.0])
