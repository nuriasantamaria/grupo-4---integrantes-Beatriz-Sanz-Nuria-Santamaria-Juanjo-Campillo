"definición analítica de las ecuaciones del problema, "
"el cómputo de la matriz Jacobiana exacta, la verificación de los invariantes 2T y L2."

import sympy as sp
import numpy as np

I1, I2, I3 = sp.symbols("I1 I2 I3", positive=True)
I = sp.diag(I1, I2, I3)

orden_inercias = sp.And(I1 < I2, I2 < I3)

omega=np.array(3)
omega1, omega2, omega3 = sp.symbols("omega1 omega2 omega3")
omega[:]=(omega1, omega2, omega3)