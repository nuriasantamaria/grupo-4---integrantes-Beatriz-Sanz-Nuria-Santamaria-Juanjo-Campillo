"""Cuerpo por defecto: caja con paneles solares"""

from __future__ import annotations

from typing import Any

import numpy as np
import pyqtgraph.opengl as gl

from space_physics import Array, ArrayLike

from visualization.theme import rgba


_BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                       [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                       [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
_BOX_EDGES = np.array([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                       (0, 4), (1, 5), (2, 6), (3, 7)]).ravel()
PANEL_COLOR = "#1f3f8f"
PANEL_LINES = (0.62, 0.72, 1.0, 0.75)
METAL_COLOR = "#cfd4de"


def box_items(center: ArrayLike, half: ArrayLike, color: str, shades: Any = (1.0, 1.0, 1.0),
              edge_color: Any = (1, 1, 1, 0.6)) -> tuple[Any, Any]:
    """Caja con caras sombreadas y aristas"""
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    V = center + V * half
    C = np.array([rgba(color, 1.0, shades[f // 4]) for f in range(12)])
    mesh = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=V, faces=_BOX_FACES, faceColors=C),
                         smooth=False, shader="shaded", glOptions="opaque")
    edges = gl.GLLinePlotItem(pos=V[_BOX_EDGES], mode="lines", color=edge_color,
                              width=1.5, antialias=True)
    return mesh, edges


def satellite_items(half: ArrayLike, color: str) -> tuple[list[Any], Array]:
    """Satélite con cuerpo, paneles solares y mástil, y su extensión"""
    h = np.asarray(half, dtype=float)
    c, b, a = np.argsort(h)

    def vec(va: float, vb: float, vc: float) -> Array:
        v = np.zeros(3)
        v[a], v[b], v[c] = va, vb, vc
        return v

    items = []
    bus = 0.2 + 0.2 * h
    items += box_items(np.zeros(3), bus, color, shades=(0.7, 1.1, 0.9))

    gap, pt = 0.12, 0.015
    Lp = max(1.25 * h[a] - bus[a] - gap, 0.45)
    pw = max(0.75 * h[b], 0.28)
    grid, struts = [], []
    for s in (-1, 1):
        inner = bus[a] + gap
        items += box_items(vec(s * (inner + Lp / 2), 0, 0), vec(Lp / 2, pw, pt),
                           PANEL_COLOR, shades=(0.9, 0.9, 1.2), edge_color=PANEL_LINES)
        for z in (-pt - 0.003, pt + 0.003):
            for i in range(1, 4):
                x = s * (inner + i * Lp / 4)
                grid += [vec(x, -pw, z), vec(x, pw, z)]
            grid += [vec(s * inner, 0, z), vec(s * (inner + Lp), 0, z)]
        struts += [vec(s * bus[a], 0, 0), vec(s * inner, 0, 0)]
    items.append(gl.GLLinePlotItem(pos=np.array(grid), mode="lines", color=PANEL_LINES,
                                   width=1, antialias=True))

    mast_top = bus[c] + 0.2
    struts += [vec(0, 0, bus[c]), vec(0, 0, mast_top)]
    items.append(gl.GLLinePlotItem(pos=np.array(struts), mode="lines",
                                   color=rgba(METAL_COLOR), width=3, antialias=True))
    md = gl.MeshData.cylinder(rows=1, cols=28, radius=[0.03, 0.26], length=0.12)
    v = md.vertexes()
    P = np.zeros_like(v)
    P[:, a], P[:, b], P[:, c] = v[:, 0], v[:, 1], v[:, 2] + mast_top - 0.04
    md.setVertexes(P)
    items.append(gl.GLMeshItem(meshdata=md, smooth=False, color=rgba(METAL_COLOR, 1.0, 0.8),
                               drawEdges=True, edgeColor=rgba(METAL_COLOR), glOptions="opaque"))

    extent = vec(bus[a] + gap + Lp, max(bus[b], pw), mast_top + 0.1)
    return items, extent
