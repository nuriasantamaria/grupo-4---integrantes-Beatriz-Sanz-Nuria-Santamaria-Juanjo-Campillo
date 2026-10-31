"""Utilidades de los gráficos (sin Qt): escalas, máscaras, flechas y contorno de la región de estabilidad"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

import numerical_engine as ne
from space_physics import Array


Range = tuple[float, float]               # intervalo (mínimo, máximo)


# ---- escalas y flechas de los graficos

def stability_window(z: Array, zoom: bool) -> tuple[tuple[Range, Range], Range, Range]:
    """Devuelve (vista, re, im) de las regiones de estabilidad; la malla es mayor que la vista y con zoom se ajusta a los puntos z"""
    if zoom and len(z):
        # se redondea a 2 cifras para que la caché de contornos acierte más
        r = float(f"{max(3.0 * float(np.max(np.abs(z))), 0.02):.2g}")
        return ((-r, r), (-r, r)), (-3 * r, 3 * r), (-3 * r, 3 * r)
    return ((-4.0, 2.0), (-3.5, 3.5)), (-7.0, 5.0), (-4.5, 4.5)


def log_safe(d: Array, floor: float = 1e-18) -> Array:
    """Valor absoluto más un suelo, para dibujar errores en escala logarítmica sin ceros"""
    return np.abs(d) + floor


def mask_below(d: Array, floor: float) -> Array:
    """Valor absoluto donde supera floor y NaN donde no (indistinguible del ruido)"""
    a = np.abs(d)
    return np.where(a > floor, a, np.nan)


def arrow_segments(X: Array, Y: Array, U: Array, V: Array, length: float, head: float = 0.38,
                   angle: float = 26.0) -> Array:
    """Segmentos (tallo y dos aletas) de flechas del campo (U, V), todas de longitud length, listos para dibujar por pares"""
    X, Y, U, V = (np.ravel(z) for z in (X, Y, U, V))
    speed = np.hypot(U, V)
    ok = speed > 1e-9 * max(speed.max(initial=0.0), 1e-300)
    if not ok.any():
        return np.zeros((0, 2))
    d = np.column_stack([U[ok], V[ok]]) / speed[ok, None]     # dirección unitaria de cada flecha
    P = np.column_stack([X[ok], Y[ok]])
    tail, tip = P - 0.5 * length * d, P + 0.5 * length * d
    ca, sa = np.cos(np.radians(angle)), np.sin(np.radians(angle))
    left = tip - head * length * np.column_stack([ca * d[:, 0] - sa * d[:, 1], sa * d[:, 0] + ca * d[:, 1]])
    right = tip - head * length * np.column_stack([ca * d[:, 0] + sa * d[:, 1], -sa * d[:, 0] + ca * d[:, 1]])
    return np.stack([tail, tip, tip, left, tip, right], axis=1).reshape(-1, 2)


# ---- región de estabilidad: contorno

# aristas de una celda como pares de esquinas (di, dj): 0 abajo, 1 derecha, 2 arriba, 3 izquierda
_CELL_EDGES = (((0, 0), (1, 0)), ((1, 0), (1, 1)), ((0, 1), (1, 1)), ((0, 0), (0, 1)))


# Segmentos de la curva en una celda según el código de esquinas (1..7), como pares de aristas
_CELL_SEGMENTS = {1: [(3, 0)], 2: [(0, 1)], 3: [(3, 1)], 4: [(1, 2)], 5: [(3, 0), (1, 2)], 6: [(0, 2)], 7: [(3, 2)]}


def level_curve(x: Array, y: Array, A: Array, level: float = 1.0) -> tuple[Array, Array]:
    """Curva de nivel A = level por marching squares, con NaN entre segmentos"""
    # bits 1, 2, 4, 8 según supere el nivel la esquina (i,j), (i+1,j), (i+1,j+1) o (i,j+1)
    a, b, c, d = A[:-1, :-1], A[1:, :-1], A[1:, 1:], A[:-1, 1:]
    code = (a > level) * 1 + (b > level) * 2 + (c > level) * 4 + (d > level) * 8
    code = np.where(code > 7, 15 - code, code)     # el complemento da los mismos segmentos
    ii, jj = np.indices(code.shape)

    def crossing(edge: int, i: Array, j: Array) -> tuple[Array, Array]:
        """Punto donde la curva corta la arista edge de las celdas (i, j), por interpolación lineal"""
        (i0, j0), (i1, j1) = _CELL_EDGES[edge]
        p, q = A[i + i0, j + j0], A[i + i1, j + j1]
        with np.errstate(all="ignore"):
            s = np.clip((level - p) / (q - p), 0.0, 1.0)
        return x[i + i0] + s * (x[i + i1] - x[i + i0]), y[j + j0] + s * (y[j + j1] - y[j + j0])

    out = []
    for case, segments in _CELL_SEGMENTS.items():
        sel = code == case
        if not sel.any():
            continue
        i, j = ii[sel], jj[sel]
        for e0, e1 in segments:
            (xa, ya), (xb, yb) = crossing(e0, i, j), crossing(e1, i, j)
            out.append(np.column_stack([xa, xb, np.full_like(xa, np.nan)]).ravel())
            out.append(np.column_stack([ya, yb, np.full_like(ya, np.nan)]).ravel())
    if not out:
        return np.array([]), np.array([])
    return np.concatenate(out[0::2]), np.concatenate(out[1::2])


def stability_contour(amplification: Callable[[Array], Array], re: Range, im: Range,
                      n: int = 301) -> tuple[Array, Array, NDArray[np.bool_], tuple[float, float, float, float]]:
    """Frontera |R(z)| = 1, máscara |R| <= 1 y rectángulo (x0, y0, ancho, alto) de la ventana"""
    x, y, A = ne.stability_grid(amplification, re, im, n)
    X, Y = level_curve(x, y, np.minimum(A, 10.0), 1.0)
    return X, Y, A <= 1.0, (float(x[0]), float(y[0]), float(x[-1] - x[0]), float(y[-1] - y[0]))
