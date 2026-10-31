"""Cuerpo alternativo: caza TIE"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from space_physics import Array, ArrayLike

from visualization.skins.meshes import blob, lathe, merge, ngon, Rig, slab, tube


_CELL_BLUE = np.array([0.10, 0.17, 0.38])
_CELL_SHADE = (1.0, 0.78, 1.22, 0.9, 1.12, 0.84)


def _rect(p0: Array, p1: Array, w: ArrayLike) -> Array:
    """Rectángulo de ancho w alrededor del segmento de p0 a p1"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * w / 2
    return np.array([p0 - n, p1 - n, p1 + n, p0 + n])


def _tie_panel(rig: Rig, sgn: int, accent: str) -> None:
    """Ala hexagonal a un lado de la cabina, en el plano y z del marco canónico"""
    xc = sgn * 0.66
    dt = 0.032 / rig.s[0]
    Ro, Ri = 1.0, 0.905
    out, inn = ngon(6, Ro, 90), ngon(6, Ri, 90)
    ALL = (0, 1, 2)

    frame = merge(*[slab([out[k], out[(k + 1) % 6], inn[(k + 1) % 6], inn[k]],
                         xc - dt, xc + dt, axis=0) for k in range(6)])
    rig.mesh(frame, "#4b5260", smooth=False, shade=2.0, stretch=ALL)

    parts, cols = [], []                                       # seis cuñas de celdas, cada una con su tono
    for k in range(6):
        V, F = slab([(0, 0), inn[k] * 0.985, inn[(k + 1) % 6] * 0.985], xc - dt * 0.55, xc + dt * 0.55, axis=0)
        parts.append((V, F))
        cols.append(np.tile(np.append(_CELL_BLUE * _CELL_SHADE[k] * 2.3, 1.0), (len(V), 1)))
    rig.mesh(merge(*parts), vc=np.vstack(cols), smooth=False, stretch=ALL)

    ribs = merge(*[slab(_rect(ngon(6, 0.1, 90)[k], ngon(6, Ro * 0.99, 90)[k], 0.045),
                        xc - dt * 1.35, xc + dt * 1.35, axis=0) for k in range(6)])
    rig.mesh(ribs, "#6d7586", smooth=False, shade=2.1, stretch=ALL)
    rig.mesh(slab(ngon(6, 0.17, 90), xc - dt * 2.2, xc + dt * 2.2, axis=0), "#59616f",
             smooth=False, shade=2.0, stretch=ALL)
    rig.mesh(slab(ngon(6, 0.105, 90), xc - dt * 2.7, xc + dt * 2.7, axis=0), accent,
             smooth=False, shade=1.3, stretch=ALL)

    for face in (-1, 1):                                       # contorno del color de la sim y malla
        x = xc + face * (dt * 1.02)
        hexo = np.vstack([out, out[:1]])
        rig.line(np.column_stack([np.full(7, x), hexo]), accent, width=3)
        for R in (0.36, 0.64):
            h = ngon(6, R, 90)
            h = np.vstack([h, h[:1]])
            rig.line(np.column_stack([np.full(7, x - face * dt * 0.45), h]), "#8aa0d6",
                     alpha=0.75, width=1)


def fighter_items(half: ArrayLike, sim_color: str) -> tuple[list[Any], Array]:
    """Caza TIE con cabina esférica, pilones y alas hexagonales"""
    rig = Rig(half, roles=(1, 2, 0), ref=(0.74, 0.87, 1.0), floor=0.42)
    ALL = (0, 1, 2)
    k = rig.k
    O = (0.0, 0.0, 0.0)
    K = dict(local=O)

    # cabina esférica con escotillas
    Rb = 0.31
    rig.mesh(blob(O, (Rb, Rb, Rb), 22, 40), "#b4bccb", shade=2.0, **K)
    rig.mesh(blob((0, 0, Rb * 0.97), (0.095, 0.095, 0.022), 8, 20), "#4d5563", shade=2.0, **K)
    rig.mesh(blob((0, -Rb * 0.93, 0), (0.12, 0.03, 0.12), 8, 20), "#4d5563", shade=2.0, **K)
    rig.mesh(blob((0, -Rb, 0), (0.045, 0.02, 0.045), 8, 14), "#ff6a3d", emissive=True, **K)

    # ventanal frontal octogonal con ocho radios
    yw = Rb * 0.80
    rig.mesh(slab(ngon(8, 0.150, 22.5), yw - 0.012, yw + 0.012, axis=1), "#0c1320", smooth=False, **K)
    out, inn = ngon(8, 0.185, 22.5), ngon(8, 0.145, 22.5)
    rig.mesh(merge(*[slab([out[i], out[(i + 1) % 8], inn[(i + 1) % 8], inn[i]], yw - 0.035, yw + 0.03, axis=1)
                     for i in range(8)]), "#6b7484", smooth=False, shade=2.1, **K)
    yl = yw + 0.0135
    a8 = np.radians(22.5 + 45 * np.arange(8))

    def wpt(R: float, ang: float) -> tuple[float, float, float]:
        return (R * math.sin(ang), yl, R * math.cos(ang))

    spokes = []
    for a in a8:
        spokes += [wpt(0.0, a), wpt(0.15, a)]
    rig.line(np.array(spokes), "#c5ccd8", width=2, mode="lines", **K)
    for R in (0.055, 0.1):
        rig.line(np.array([wpt(R, a) for a in list(a8) + [a8[0]]]), "#c5ccd8", width=1.5, **K)

    # cañones láser bajo el ventanal
    for s in (-1, 1):
        P = [(s * 0.065, 0.2, -0.12), (s * 0.065, 0.33, -0.13), (s * 0.065, 0.47, -0.135)]
        rig.mesh(tube(P, rw=[0.026, 0.02, 0.016], n=10, per=4), "#3a404c", shade=2.0, **K)
        rig.mesh(blob((s * 0.065, 0.485, -0.135), (0.026, 0.03, 0.026), 8, 12), "#5dff7a",
                 emissive=True, **K)

    # collares octogonales y pilones
    x_col = 0.33 * k / rig.s[0]
    for s in (-1, 1):
        collar = lathe([(0.0, 0.0), (0.0, 0.2), (s * 0.075, 0.2), (s * 0.075, 0.0)], n=8, axis=0,
                       center=(s * 0.22, 0, 0))
        rig.mesh(collar, "#59616f", smooth=False, shade=2.0, **K)
        P = [(s * x_col * 0.9, 0, 0), (s * (x_col + 0.66 - x_col) * 0.5, 0, 0), (s * 0.63, 0, 0)]
        r = 0.095 * k
        pyl = tube(P, rw=[r / rig.s[1], 0.8 * r / rig.s[1], 0.65 * r / rig.s[1]],
                   rh=[r / rig.s[2], 0.8 * r / rig.s[2], 0.65 * r / rig.s[2]], n=14, side=(0, 1, 0), per=4)
        rig.mesh(pyl, "#7a8392", shade=2.0, stretch=ALL)

    for s in (-1, 1):
        _tie_panel(rig, s, sim_color)

    return rig.finish(), rig.extent
