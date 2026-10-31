"""Cuerpo alternativo: vaca Holstein"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from space_physics import Array, ArrayLike

from visualization.skins.meshes import blob, IntArray, lathe, Rig, rot_axis, smoothstep, tube


_COW_WHITE = np.array([0.96, 0.95, 0.92])
_COW_BLACK = np.array([0.07, 0.07, 0.08])
# (centro, semiejes) de las manchas, en el marco canónico de la vaca
_SPOTS = [
    ((0.20, 0.22, 0.30), (0.20, 0.13, 0.17)),
    ((-0.30, 0.24, 0.14), (0.24, 0.13, 0.22)),
    ((-0.08, -0.24, 0.30), (0.26, 0.13, 0.15)),
    ((0.30, -0.25, 0.04), (0.15, 0.11, 0.17)),
    ((-0.55, -0.12, 0.36), (0.15, 0.15, 0.13)),
    ((0.05, 0.02, 0.52), (0.16, 0.13, 0.08)),
    ((0.74, 0.17, 0.30), (0.11, 0.09, 0.11)),     # parche en la cabeza
    ((0.62, -0.10, 0.34), (0.13, 0.12, 0.10)),    # cuello
    ((-0.62, 0.13, 0.00), (0.10, 0.10, 0.16)),    # muslo
]


def _coat(V: Array) -> Array:
    """Color del pelaje por vértice: manchas negras sobre fondo blanco"""
    f = np.full(len(V), 9.0)
    for c, r in _SPOTS:
        d = np.linalg.norm((V - np.array(c)) / np.array(r), axis=1)
        f = np.minimum(f, d)
    wob = 0.13 * np.sin(9.0 * V[:, 0] + 4.0 * V[:, 1]) * np.sin(7.0 * V[:, 2] + 3.0 * V[:, 1] + 1.0)
    m = 1.0 - smoothstep(0.90, 1.06, f + wob)
    rgb = _COW_WHITE[None, :] * (1 - m[:, None]) + _COW_BLACK[None, :] * m[:, None]
    return np.column_stack([rgb * 2.0, np.ones(len(V))])


def cow_items(half: ArrayLike, sim_color: str) -> tuple[list[Any], Array]:
    """Vaca: el cuerpo largo sigue al eje de menor inercia; el collar lleva el color de la simulación"""
    rig = Rig(half, roles=(0, 2, 1), ref=(1.0, 0.30, 0.78), floor=0.24)
    ALL = (0, 1, 2)

    def coat(vf: tuple[Array, IntArray], **kw) -> None:
        rig.mesh(vf, vc=_coat(vf[0]), **kw)

    # tronco: tubo elíptico con perfil de barril
    t = np.cos(np.linspace(math.pi, 0, 45))
    e = np.maximum(np.maximum(1 - np.abs(t) ** 2.6, 0) ** (1 / 2.6), 0.004)
    xs = -0.07 + 0.63 * t
    path = np.column_stack([xs, np.zeros_like(xs), 0.14 - 0.03 * t])
    coat(tube(path, rw=0.285 * e * (1 + 0.05 * t), rh=0.36 * e, n=44, side=(0, 1, 0), per=1,
              caps=False), stretch=ALL)

    # cuello y cabeza
    coat(tube([(0.34, 0, 0.23), (0.55, 0, 0.255), (0.71, 0, 0.23)], rw=[0.23, 0.2, 0.17],
              rh=[0.21, 0.2, 0.18], n=28, side=(0, 1, 0), per=6), stretch=ALL)
    coat(tube([(0.68, 0, 0.27), (0.79, 0, 0.19), (0.90, 0, -0.01)], rw=[0.165, 0.15, 0.11],
              rh=[0.15, 0.135, 0.095], n=28, side=(0, 1, 0), per=6), stretch=ALL)
    mz = (0.915, 0, -0.065)
    rig.mesh(blob(mz, (0.07, 0.115, 0.08), 12, 22), "#f4aab5", shade=1.5, local=mz, stretch=ALL)
    for s in (-1, 1):
        nc = (0.982, s * 0.045, -0.045)
        rig.mesh(blob(nc, (0.014, 0.02, 0.016), 8, 12), "#3a1e26", local=nc)
        ec = (0.775, s * 0.15, 0.245)                                    # ojo con brillo
        rig.mesh(blob(ec, (0.026, 0.024, 0.03), 10, 16), "#0c0c0e", local=ec)
        gc = (0.79, s * 0.165, 0.258)
        rig.mesh(blob(gc, (0.008, 0.008, 0.008), 6, 8), "#ffffff", emissive=True, local=gc)
        oc = (0.665, s * 0.19, 0.285)                                    # oreja caída
        R = rot_axis((1, 0, 0), -s * 28)
        rig.mesh(blob(oc, (0.035, 0.115, 0.05), 10, 18, rot=R), "#25252a", shade=1.5, local=oc)
        ic = (0.675, s * 0.185, 0.28)
        rig.mesh(blob(ic, (0.02, 0.08, 0.03), 8, 14, rot=R), "#f1a6b2", shade=1.5, local=ic)
        hb = (0.70, s * 0.10, 0.37)                                      # cuerno curvo
        horn = tube([hb, (0.70, s * 0.19, 0.43), (0.71, s * 0.26, 0.53), (0.74, s * 0.25, 0.63)],
                    rw=[0.042, 0.034, 0.022, 0.004], n=12, per=6)
        rig.mesh(horn, "#efe3bf", shade=1.6, local=hb)

    # patas (blancas, con corvejón en las traseras) y pezuñas
    legs = {
        "front": ([(0.34, 0, 0.05), (0.35, 0, -0.25), (0.355, 0, -0.48), (0.36, 0, -0.66)],
                  [0.115, 0.085, 0.055, 0.05], 0.34),
        "rear": ([(-0.46, 0, 0.05), (-0.50, 0, -0.22), (-0.56, 0, -0.42), (-0.50, 0, -0.58),
                  (-0.49, 0, -0.66)], [0.15, 0.11, 0.06, 0.05, 0.05], -0.49),
    }
    for kind, (pts, rad, hx) in legs.items():
        for s in (-1, 1):
            P = [(x, s * 0.15, z) for x, _, z in pts]
            coat(tube(P, rw=rad, n=16, per=6), stretch=ALL)
            hc = (hx + 0.012, s * 0.15, -0.735)
            rig.mesh(blob(hc, (0.062, 0.052, 0.04), 10, 16), "#16161a", local=hc)

    # ubre con pezones
    uc = (-0.30, 0, -0.22)
    rig.mesh(blob(uc, (0.15, 0.11, 0.09), 12, 20), "#f2a9b4", shade=1.5, local=uc, stretch=(0, 1))
    for dx in (-0.055, 0.055):
        for dy in (-0.04, 0.04):
            tc = (-0.30 + dx, dy, -0.29)
            rig.mesh(tube([tc, (tc[0], tc[1], tc[2] - 0.055)], rw=[0.024, 0.017], n=10, per=1),
                     "#e58b9c", shade=1.5, local=tc)

    # cola con borla negra
    rig.mesh(tube([(-0.66, 0, 0.30), (-0.77, 0, 0.25), (-0.80, 0, 0.05), (-0.77, 0, -0.22),
                   (-0.76, 0, -0.42)], rw=[0.036, 0.03, 0.025, 0.02, 0.018], n=10, per=6),
             "#f1eee6", shade=1.8, stretch=ALL)
    tc = (-0.76, 0, -0.50)
    rig.mesh(blob(tc, (0.04, 0.04, 0.095), 10, 14), "#16161a", local=tc)

    # collar con el color de la simulación y cencerro dorado
    ph = np.linspace(0, 2 * math.pi, 40)
    ring = np.column_stack([np.full_like(ph, 0.555), 0.215 * np.cos(ph), 0.255 + 0.235 * np.sin(ph)])
    rig.mesh(tube(ring, rw=0.03, n=10, per=1, caps=False), sim_color, shade=1.5, stretch=ALL)
    bc = (0.555, 0, 0.03)
    bell = lathe([(0.045, 0.022), (0.03, 0.045), (0.0, 0.058), (-0.035, 0.07), (-0.05, 0.072)],
                 n=20, axis=2, center=bc, caps=True)
    rig.mesh(bell, "#f0bc3a", shade=1.6, local=bc, stretch=(0, 1))
    cl = (0.555, 0, -0.02)
    rig.mesh(blob(cl, (0.016, 0.016, 0.016), 6, 10), "#5a4210", local=cl)

    return rig.finish(), rig.extent
