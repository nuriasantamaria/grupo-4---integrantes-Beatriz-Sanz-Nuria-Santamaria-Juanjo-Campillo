"""Registro de esquemas: cada esquema de un paso con sus propiedades teóricas, y la clave con la que se elige"""

from __future__ import annotations

from typing import NamedTuple

from numpy.typing import ArrayLike

from numerical_engine.numerical_engine import ComplexArray, Step, crank_nicolson, euler, implicit_euler, runge_kutta_4
from numerical_engine.stability import amplification_factor


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
