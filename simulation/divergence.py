"""Detección de divergencia numérica: vigila el módulo de ω para cortar las trayectorias que explotan.

Dos criterios:
    · durante la integración, `stop_criterion(limit)` para el esquema si Σ|ω| deja de ser finita o llega a limit;
    · sobre un bloque ya integrado, `first_divergence(Y, bound)` localiza la primera fila no finita o con ‖ω‖ > bound.
La cota física la da `omega_threshold`: DIVERGENCE_FACTOR veces el máximo de ‖ω‖ posible en la órbita exacta."""

from __future__ import annotations

from typing import Callable

import numpy as np

from numerical_engine.numerical_engine import Array, Component
from space_physics import ArrayLike, omega_bound

OMEGA = slice(0, 3)              # columnas de ω en el estado (ω, q)
DIVERGENCE_FACTOR = 1e3          # ‖ω‖ > 1e3 · (máximo posible en la órbita exacta) ⇒ la simulación diverge
DIVERGENCE_LIMIT = 1e6           # Σ|ω| a partir de la cual una trayectoria se da por perdida (NaN)


def omega_threshold(I: ArrayLike, w0: ArrayLike) -> float:
    """Cota de ‖ω‖ por encima de la cual la solución numérica se da por divergida"""
    return DIVERGENCE_FACTOR * omega_bound(I, w0)


def stop_criterion(limit: float = DIVERGENCE_LIMIT, components: Component = OMEGA) -> Callable[[Array], bool]:
    """Criterio de parada paso a paso: True si la suma de |u[components]| no es finita o llega a limit"""
    def blown(u: Array) -> bool:
        with np.errstate(all="ignore"):
            return not (np.abs(u[components]).sum() < limit)
    return blown


def first_divergence(Y: Array, bound: float, components: Component = OMEGA) -> int | None:
    """Índice de la primera fila de Y no finita o con norma de Y[components] > bound; None si no hay"""
    with np.errstate(all="ignore"):
        bad = ~np.isfinite(Y).all(axis=1) | (np.linalg.norm(Y[:, components], axis=1) > bound)
    return int(np.argmax(bad)) if bad.any() else None
