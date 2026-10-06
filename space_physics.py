"""Definición analítica de las ecuaciones del problema,
cómputo de la matriz Jacobiana exacta y
verificación de los invariantes 2T y L2."""

import sympy as sp
import numpy as np


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
