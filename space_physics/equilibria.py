"""¿Qué ejes son estables? Equilibrios, autovalores, estabilidad lineal y no lineal, y lado de la separatriz"""

from typing import TypedDict

import numpy as np

from space_physics.model import Array, ComplexArray, angular_momentum_sq, is_degenerate, kinetic_energy, sorted_axes



class Equilibrium(TypedDict):
    """Equilibrio w = Omega*e_k y su estabilidad, resultado de equilibrium_analysis"""
    omega: Array           # rotación pura, con Omega^2 = 2T/I_k
    eigvals: ComplexArray  # autovalores de la Jacobiana
    type: str              # clasificación lineal
    stability: str         # 'estable' | 'marginal' | 'inestable' (lineal)
    nonlinear: str         # 'estable' | 'inestable' | 'neutro' (con T y |L|^2)


# ============================================================== EQUILIBRIA

def principal_equilibria(I, twoT):
    """Fila k = rotación pura sobre el eje k con energía 2T: omega = sqrt(2T/I_k) * e_k"""
    I = np.asarray(I, dtype=float)
    eq = np.zeros((3, 3))
    for k in range(3):
        eq[k, k] = np.sqrt(twoT / I[k])
    return eq


def analytic_eigenvalues(I, twoT):
    """Fila k = autovalores (0, mu, -mu) de la Jacobiana en el eje k; mu es imaginario en los ejes extremos"""
    I = np.asarray(I, dtype=float)
    lam = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        mu = np.sqrt(complex(twoT / I[k] * (I[k] - I[i]) * (I[j] - I[k]) / (I[i] * I[j])))
        lam[k] = [0, mu, -mu]
    return lam


def equilibrium_analysis(I, twoT):
    """Los tres equilibrios w = Omega_k*e_k con autovalores, tipo, estabilidad lineal y no lineal"""
    I = np.asarray(I, dtype=float)
    lam = analytic_eigenvalues(I, twoT)
    result = []
    for k, omega in enumerate(principal_equilibria(I, twoT)):
        # lineal: autovalores (0, mu, -mu); traza nula, así que no hay nodos ni focos
        small = 1e-9 * max(1.0, np.max(np.abs(lam[k])))   # por debajo de esto se considera cero
        if np.all(np.abs(lam[k]) <= small):
            kind, stability = "degenerado (todos los autovalores nulos)", "marginal"
        elif np.any(np.abs(lam[k].real) > small):     # mu real: una dirección crece y otra decae
            kind, stability = "punto de silla", "inestable"
        else:                                         # mu imaginario: solo gira, la linealización no decide
            kind, stability = "centro", "marginal"
        # no lineal (Lyapunov con T y L^2): eje extremo estable, intermedio inestable, empate neutro
        others = np.delete(I, k)
        if twoT <= 0:                                 # sin rotación no hay nada que analizar
            nonlinear = "neutro"
        elif I[k] > others.max() or I[k] < others.min():
            nonlinear = "estable"
        elif others.min() < I[k] < others.max():
            nonlinear = "inestable"
        else:
            nonlinear = "neutro"                      # sólido simétrico
        result.append({"omega": omega, "eigvals": lam[k], "type": kind,
                       "stability": stability, "nonlinear": nonlinear})
    return result


# ============================================================== SEPARATRIX

SEPARATRIX_TOL = 1e-3            # si |side| queda por debajo, la órbita se da por sobre la separatriz


def separatrix_side(I, w0):
    """(|L|^2 - 2T*I_med) / (2T*I_med): > 0 la órbita rodea el eje mayor, < 0 el menor, ~ 0 separatriz"""
    I = np.asarray(I, dtype=float)
    _, b, _ = sorted_axes(I)
    twoT_Ib = 2 * kinetic_energy(w0, I) * I[b]
    return float((angular_momentum_sq(w0, I) - twoT_Ib) / twoT_Ib)


def surrounded_axis(I, w0, tol=SEPARATRIX_TOL):
    """Eje (mayor o menor inercia) que rodea la órbita; None en separatriz, simétrico o reposo"""
    twoT = 2 * kinetic_energy(w0, I)
    if is_degenerate(I) or twoT <= 0:
        return None
    side = separatrix_side(I, w0)
    if abs(side) < tol:
        return None
    smallest, _, largest = sorted_axes(I)
    return largest if side > 0 else smallest
