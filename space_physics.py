"Definición analítica de las ecuaciones del problema, "
"cómputo de la matriz Jacobiana exacta,"
"verificación de los invariantes 2T y L2."

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

def kinetic_energy(omega, I):
    return 0.5 * np.sum(np.array(I) * np.asarray(omega)**2, axis=-1)

def angular_momentum_sq(omega, I):
    return np.sum((np.array(I) * np.asarray(omega)) ** 2, axis=-1)
