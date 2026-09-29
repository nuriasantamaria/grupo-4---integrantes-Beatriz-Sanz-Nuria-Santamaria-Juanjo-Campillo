"Definición analítica de las ecuaciones del problema, "
"cómputo de la matriz Jacobiana exacta,"
"verificación de los invariantes 2T y L2."

import sympy as sp
import numpy as np

# Había empezado definiendo en este script el tensor de inercia
# y la velocidad angular, pero creo que no hace falta porque este módulo 
# es solo de operaciones, los valores se determinan desde la GUI y ahí podemos 
# hacer las verificaciones. 

#I1, I2, I3 = sp.symbols("I1 I2 I3", positive=True)
#I = sp.diag(I1, I2, I3)

#orden_inercias = sp.And(I1 < I2, I2 < I3)

#omega=np.array(3)
#omega1, omega2, omega3 = sp.symbols("omega1 omega2 omega3")
#omega[:]=(omega1, omega2, omega3)

#momento_cinetico=I*omega

def euler_equations(omega, I):
    return np.array([
        (I[0,0] - I[2,2]) / I[0,0] * omega[1] * omega[2],
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
    return 0.5 * np.dot(omega, np.array(I) * omega)

def angular_momentum_sq(omega, I):  
    return np.sum((np.array(I) * omega) ** 2)
