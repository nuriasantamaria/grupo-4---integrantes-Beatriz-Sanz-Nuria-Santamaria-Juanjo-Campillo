"""Amplificación y estabilidad: R(z) de cada esquema sacada de su propio paso"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from numerical_engine.numerical_engine import ComplexArray, Step

# Paradigma declarativo: R(z) no se escribe a mano, cada esquema la descubre dando un paso del problema test u' = z*u.
# En ese problema, un paso de tamaño dt multiplica u por R(z*dt). Partiendo de u = 1 con dt = 1, el paso
# devuelve directamente R(z). Así R sale del mismo código que integra, y un esquema nuevo no necesita fórmula.
# Región de estabilidad absoluta: los z con |R(z)| <= 1 (la raíz r = R(z) de pi(r, z) = 0 dentro del círculo
# unidad). Ahí un modo de autovalor lambda no crece por efecto del esquema con el que se integra.

def amplification_factor(scheme: Step, z: ArrayLike, implicit: bool = False, chunk: int = 32) -> ComplexArray:
    """R(z) de un esquema, obtenida de su propio paso. z puede ser un número, un vector o una malla 2D, y R sale con
    su misma forma. Los explícitos procesan todos los puntos a la vez; los implícitos usan Newton por bloques de
    chunk puntos (inf en el polo de R)."""
    z = np.asarray(z, dtype=complex)
    if not implicit:                           # Explícito: solo operaciones punto a punto, toda la malla de golpe
        return np.asarray(scheme(lambda U, t: z * U, np.ones_like(z), 1.0, 0.0), dtype=complex)
    flat = z.ravel()                           # Implícito: la malla aplanada, troceada en bloques
    blocks = [_implicit_block(scheme, flat[i:i + chunk]) for i in range(0, flat.size, chunk)]
    return np.concatenate(blocks).reshape(z.shape)


def _implicit_block(scheme: Step, z: ComplexArray) -> ComplexArray:
    """R en los puntos z (vector) de un esquema implícito, con un solo paso del problema test; inf en el polo de R.
    Los puntos son independientes, así que la jacobiana de u' = z*u es diagonal, diag(z), y se pasa exacta: la
    numérica (my_jacobian) trabaja con reales y perdería la parte imaginaria de z."""
    try:
        return np.asarray(scheme(lambda U, t: z * U, np.ones_like(z), 1.0, 0.0, jac=lambda U, t: np.diag(z)),
                          dtype=complex)
    except (np.linalg.LinAlgError, ArithmeticError, RuntimeError):   # algún punto es el polo: Newton no puede
        if z.size == 1:
            return np.array([np.inf])
        return np.concatenate([_implicit_block(scheme, z[j:j + 1]) for j in range(z.size)])   # punto a punto
