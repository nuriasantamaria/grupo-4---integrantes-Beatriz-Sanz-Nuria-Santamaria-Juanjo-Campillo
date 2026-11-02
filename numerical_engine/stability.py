"""Amplificación y estabilidad: R(z) de cada esquema sacada de su propio paso, y registro de esquemas SCHEMES"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from numerical_engine.numerical_engine import ComplexArray, Step, crank_nicolson, euler, implicit_euler, runge_kutta_4

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


class Scheme(NamedTuple):
    """Esquema de un paso: su función de paso y sus propiedades teóricas. R(z) no se guarda: se descubre del paso."""
    step: Step                            # función de paso scheme(f, u, dt, t)
    order: int                            # orden global del error en dt
    module_order: int | None = None       # orden global del error de |R| en un centro; None si |R| = 1 exacto
    phase_order: int = 1                  # orden global del error de fase (arg R - beta*dt) en un centro
    label: str = ""                       # nombre para mostrar
    short: str = ""                       # nombre corto para tablas y leyendas
    implicit: bool = False                # usa Newton (admite jac=)

    def amplification(self, z: ArrayLike) -> ComplexArray:
        """R(z) de este esquema, sacada de su propio paso (ver amplification_factor)"""
        return amplification_factor(self.step, z, self.implicit)

#Registro de esquemas: clave → Scheme(paso, orden, orden de módulo, orden de fase, nombre, nombre corto, implícito)
SCHEMES: dict[str, Scheme] = {
    "euler": Scheme(euler,           1, 1, 2, "Euler explícito", "Euler"),
    "be":    Scheme(implicit_euler,  1, 1, 2, "Euler implícito (inverso)", "Euler imp.", True),
    "cn":    Scheme(crank_nicolson,  2, None, 2, "Crank-Nicolson (implícito)", "C-N", True),
    "rk4":   Scheme(runge_kutta_4,   4, 5, 4, "Runge-Kutta 4", "RK4"),
}
