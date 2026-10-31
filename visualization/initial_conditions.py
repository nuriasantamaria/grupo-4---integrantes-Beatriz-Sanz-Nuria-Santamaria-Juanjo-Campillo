"""Edición de las condiciones iniciales por ejes y comparación de problemas (sin Qt)"""

from __future__ import annotations

import math
from typing import Sequence, TypedDict

import numpy as np

from space_physics import angular_momentum_sq, ArrayLike, kinetic_energy


# ---- edicion de las condiciones iniciales y comparacion de problemas

def axis_decomposition(w: Sequence[float], keep: tuple[int, ...] | None = None) -> tuple[int, int, int]:
    """Terna (principal, perturbación 1, perturbación 2) ordenada por |omega|; con omega nulo mantiene el principal de keep"""
    size = [abs(x) for x in w]
    if keep is not None and max(size) < 1e-12:
        a = keep[0]
    else:
        a = int(np.argmax(size))
    others = [k for k in range(3) if k != a]
    # sorted es estable: en empate gana el índice menor
    b, c = sorted(others, key=lambda k: -size[k])
    return a, b, c


def omega_from_axis(axes: tuple[int, int, int], main: float, p1: float, p2: float = 0.0) -> list[float]:
    """Construye omega0 con el valor principal y las dos perturbaciones sobre sus ejes"""
    a, b, c = axes
    w = [0.0] * 3
    w[a], w[b], w[c] = main, p1, p2
    return w


def reassign_axes(axes: tuple[int, int, int], new_main: int | None = None,
                  new_p1: int | None = None) -> tuple[int, int, int]:
    """Nueva terna de ejes al elegir otro eje principal o otra primera perturbación"""
    a, b, c = axes
    if new_main is not None and new_main != a:
        rest = [k for k in range(3) if k != new_main]
        b2 = b if b in rest else a
        return new_main, b2, next(k for k in rest if k != b2)
    if new_p1 is not None and new_p1 != b:
        return a, new_p1, next(k for k in range(3) if k not in (a, new_p1))
    return a, b, c


def same_problem(I_a: ArrayLike, w_a: ArrayLike, I_b: ArrayLike, w_b: ArrayLike) -> bool:
    """Indica si coinciden I y omega0, es decir, si la solución exacta es la misma"""
    return bool(np.allclose(I_a, I_b, rtol=1e-12, atol=0) and np.allclose(w_a, w_b, rtol=1e-12, atol=1e-15))


SURFACE_TOL = 1e-3


class SurfaceMismatch(TypedDict):
    """Diferencia entre las superficies de dos problemas"""
    same_I: bool
    dT: float              # diferencia relativa de energía
    dL: float              # diferencia relativa del momento angular al cuadrado


def surface_mismatch(I_ref: ArrayLike, w_ref: ArrayLike, I: ArrayLike, w: ArrayLike,
                     tol: float = SURFACE_TOL) -> SurfaceMismatch | None:
    """None si la trayectoria de (I, w) está sobre las superficies de referencia (a tol); si no, las diferencias relativas"""
    Ir, Ic = np.asarray(I_ref, dtype=float), np.asarray(I, dtype=float)
    Tr, Tc = float(kinetic_energy(w_ref, Ir)), float(kinetic_energy(w, Ic))
    Lr, Lc = float(angular_momentum_sq(w_ref, Ir)), float(angular_momentum_sq(w, Ic))

    def rel(a: float, b: float) -> float:
        return (b / a - 1.0) if a > 0 else (0.0 if b == 0 else math.inf)

    same_I = bool(np.allclose(Ir, Ic, rtol=1e-9, atol=1e-12))
    dT, dL = rel(Tr, Tc), rel(Lr, Lc)
    if same_I and abs(dT) <= tol and abs(dL) <= tol:
        return None
    return {"same_I": same_I, "dT": dT, "dL": dL}
