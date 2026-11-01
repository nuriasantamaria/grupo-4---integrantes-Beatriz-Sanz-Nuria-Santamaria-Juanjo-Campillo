"""
Ecuaciones que rigen la física. Momentos de inercia, Euler, invariantes T y |L|^2 y el campo f(u, t) del sistema.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from collections.abc import Callable

Array = NDArray[np.float64]

# ================ [1] INERCIA ================

def inertia_matrix(I: ArrayLike) -> Array:
    """Comprueba que se tengan 3 momentos de inercia, que sean positivos y finitos, y devuelve el vector de floats
    de momentos de inercia."""

    I = np.asarray(I, dtype=float)
    valid = I.shape == (3,) and np.all(np.isfinite(I)) and np.all(I > 0)
    if not valid:
        raise ValueError(f"I debe tener tres momentos principales positivos (recibido {I.tolist()})")
    return I


def sorted_axes(I: ArrayLike) -> tuple[int, int, int]:
    """Índices de los ejes de menor, intermedio y mayor momento de inercia, en ese orden."""
    i_min, i_mid, i_max = np.argsort(I, kind="stable").tolist() # I = (4.2, 7.5, 2.8) -> (2, 0, 1)
    return i_min, i_mid, i_max


def is_degenerate(I: ArrayLike, tol: float=1e-6) -> bool:
    """Comprueba si el sistema es degenerado, esto es, al menos dos momentos de inercia son iguales. En ese caso
    no hay eje intermedio y, por ende, el análisis asociado no aplica."""
    return bool(np.diff(np.sort(I)).min() < tol)

# ================ [2] DINÁMICA ================

def euler_equations(omega: Array, I: Array) -> Array:
    """Ecuaciones dinámicas de Euler para un sólido rígido girando alrededor de su centro de masas en ausencia
    de pares de fuerzas externos. Esta función admite el omega de entrada como (3,) o (3, nx, ny) para evaluar
    múltiples estados simultáneamente (campo de vectores)."""
    I1, I2, I3 = I
    w1, w2, w3 = omega
    return np.array([(I2 - I3) / I1 * w2 * w3, (I3 - I1) / I2 * w3 * w1, (I1 - I2) / I3 * w1 * w2])


def jacobian_matrix(omega: Array, I: Array) -> Array:
    """Matriz jacobiana analítica del sistema dinámico."""
    I1, I2, I3 = I
    w1, w2, w3 = omega
    c1 = (I2 - I3) / I1
    c2 = (I3 - I1) / I2
    c3 = (I1 - I2) / I3
    return np.array([[0.0, c1 * w3, c1 * w2],
                     [c2 * w3, 0.0, c2 * w1],
                     [c3 * w2, c3 * w1, 0.0]])

# ================ [3] INVARIANTES ================

def kinetic_energy(omega: ArrayLike, I: ArrayLike) -> Array | float:
    """Energía cinética rotacional instantánea del sólido rígido. Admite omega como (3,) o (n,3)."""
    omega = np.asarray(omega, dtype=float)
    I = np.asarray(I, dtype=float)
    return 0.5 * np.sum(I * omega ** 2, axis=-1) # Suma sobre el último eje (componentes de omega para cada instante).


def angular_momentum_sq(omega: ArrayLike, I: ArrayLike) -> Array | float:
    """Módulo al cuadrado del momento angular instantáneo. Admite omega como (3,) o (n,3)."""
    omega = np.asarray(omega, dtype=float)
    I = np.asarray(I, dtype=float)
    return np.sum((I * omega) ** 2, axis=-1) # Suma sobre el último eje (componentes de omega para cada instante).


def conserved_quantities(w0: ArrayLike, I: ArrayLike) -> tuple[float, float]:
    """Invariantes T y |L| correspondientes a la condición inicial."""
    T = float(kinetic_energy(w0, I))
    L = float(np.sqrt(angular_momentum_sq(w0, I)))
    return T, L


def omega_bound(I: ArrayLike, w0: ArrayLike) -> float:
    """Cota de |omega| en la órbita exacta: |w| <= sqrt(2T/I_min). Es el semieje mayor del elipsoide de energía.
    Proviene de la desigualdad siguiente: 2T = sum(I_i * w_i^2) >= I_min * |w|^2 """
    I = np.asarray(I, dtype=float)
    twoT = 2.0 * kinetic_energy(w0, I)
    return float(np.sqrt(twoT / np.min(I)))

def conservation_drift(W: ArrayLike, I: ArrayLike, T0: float, L0: float) -> tuple[Array | float, Array | float]:
    """Error de T y de |L| de la trayectoria W (n, 3) respecto a T0 y L0 (de conserved_quantities)."""
    dT = kinetic_energy(W, I) - T0
    dL = np.sqrt(angular_momentum_sq(W, I)) - L0
    return dT, dL

# ================ [4] SÓLIDO RÍGIDO ================

def rigid_body_rhs(u: Array, I: Array) -> Array:
    """Para el sólido rígido, el vector de estado es u = (w, q). La evolución q' completa du/dt = f(u,t).
    (q0, q1, q2, q3) = [cos(theta/2), nx*sin(theta/2), ny*sin(theta/2), nz*sin(theta/2)], con n el eje de giro."""
    w1, w2, w3, q0, q1, q2, q3 = u
    dw = euler_equations(u[:3], I)
    dq0 = 0.5 * (-q1 * w1 - q2 * w2 - q3 * w3) # -1/2 * (q_vec dot omega)
    dq1 = 0.5 * (q0 * w1 + q2 * w3 - q3 * w2) # 1/2 * (q0 * omega + q_vec x omega), componente 1
    dq2 = 0.5 * (q0 * w2 + q3 * w1 - q1 * w3) # 1/2 * (q0 * omega + q_vec x omega), componente 2
    dq3 = 0.5 * (q0 * w3 + q1 * w2 - q2 * w1) # 1/2 * (q0 * omega + q_vec x omega), componente 3
    return np.array([*dw, dq0, dq1, dq2, dq3])


def rigid_body_jacobian(u: Array, I: Array) -> Array:
    """Jacobiana 7x7 del sistema rigid_body_rhs, con la misma dimensión y orden de u."""
    w1, w2, w3, q0, q1, q2, q3 = u
    J = np.zeros((7, 7)) # El bloque superior derecho es nulo (omega no depende de q, pero sí al revés).
    J[0:3, 0:3] = jacobian_matrix(u[:3], I)
    J[3:7, 0:3] = 0.5 * np.array([[-q1, -q2, -q3],
                                  [q0, -q3, q2],
                                  [q3, q0, -q1],
                                  [-q2, q1, q0]])
    J[3:7, 3:7] = 0.5 * np.array([[0, -w1, -w2, -w3],
                                  [w1, 0, w3, -w2],
                                  [w2, -w3, 0, w1],
                                  [w3, w2, -w1, 0]])
    return J


def make_rhs(I: ArrayLike) -> Callable[[Array, float], Array]:
    """Lado derecho f(u, t) del sólido rígido de momentos I, listo para los integradores. I se valida una vez. Esto
    es necesario porque el motor numérico es genérico; llama a f(u, t) sin conocer I."""
    inertia = inertia_matrix(I)

    def f(u: Array, t: float) -> Array: # Genérico, aunque los SDA no dependen explícitamente de t.                 
        return rigid_body_rhs(u, inertia) # Llama con un valor ya fijado de I.
    return f


def make_jacobian(I: ArrayLike) -> Callable[[Array, float], Array]:
    """J(u, t) = df/du del sólido de momentos I. Adapta el formato al esperado por los integradores, validando
    I una vez."""
    inertia = inertia_matrix(I)

    def J(u: Array, t:float) -> Array: # Genérico, aunque los SDA no dependen explícitamente de t.  
        return rigid_body_jacobian(u, inertia) # Llama con un valor ya fijado de I.
    return J


def initial_state(w0: ArrayLike) -> Array:
    """Estado inicial (w0, q). El sólido aún no ha rotado, pero sí tiene velocidad de giro inicial."""
    q_identity = [1.0, 0.0, 0.0, 0.0] # Cuaternión identidad (orientación antes de rotar).
    return np.concatenate([np.asarray(w0, dtype=float), q_identity])
