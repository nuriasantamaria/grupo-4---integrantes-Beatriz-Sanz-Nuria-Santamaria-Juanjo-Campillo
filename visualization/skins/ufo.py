"""Cuerpo alternativo: ovni"""

from __future__ import annotations

import math
from typing import Any, Sequence, Union

import numpy as np
from PySide6 import QtGui

from space_physics import Array, ArrayLike

from visualization.skins.meshes import (blob, IntArray, lathe, orient_out, Rig, rot_axis, slab, smooth_path,
                                        smoothstep, tube)


_METAL_HI = np.array([0.74, 0.78, 0.85])
_METAL_LO = np.array([0.40, 0.44, 0.52])


def _hull_profile() -> Array:
    """Perfil (z, radio) del casco lenticular"""
    pts = [(-0.205, 0.0), (-0.205, 0.22), (-0.19, 0.45), (-0.145, 0.68), (-0.075, 0.88),
           (-0.015, 0.985), (0.02, 1.0), (0.055, 0.965), (0.095, 0.86), (0.135, 0.66),
           (0.17, 0.46), (0.19, 0.38), (0.19, 0.0)]
    P = smooth_path([(z, r, 0.0) for z, r in pts], per=6)
    return P[:, :2]


def _hull_colors(V: Array, rim_rgb: Array) -> Array:
    """Colores del casco con surcos y una banda del color de la simulación"""
    rho = np.hypot(V[:, 0], V[:, 1])
    z = V[:, 2]
    base = np.where((z > 0.02)[:, None], _METAL_HI[None, :], _METAL_LO[None, :])
    groove = np.zeros(len(V))                                  # surcos concéntricos
    for r0 in (0.48, 0.74):
        groove = np.maximum(groove, 1 - smoothstep(0.004, 0.02, np.abs(rho - r0)))
    base = base * (1 - 0.42 * groove[:, None])
    band = (1 - smoothstep(0.03, 0.055, np.abs(z - 0.018))) * smoothstep(0.88, 0.95, rho)
    rgb = base * (1 - band[:, None]) + rim_rgb[None, :] * band[:, None]   # banda del color de la sim
    return np.column_stack([rgb * 2.1, np.ones(len(V))])


def _mini_cow(rig: Rig, center: ArrayLike, tilt: float = -22, spin: float = 35) -> None:
    """Vaquita abducida flotando en el haz"""
    R = rot_axis((0, 1, 0), tilt) @ rot_axis((0, 0, 1), spin)
    ctr = np.asarray(center, float)

    def part(off: ArrayLike, radii: ArrayLike, color: str) -> None:
        c = ctr + R @ np.asarray(off)
        rig.mesh(blob(c, radii, 8, 12, rot=R), color, shade=2.0, local=center)

    def limb(a: ArrayLike, b: ArrayLike, r: Union[float, Sequence[float]], color: str = "#f4f1ea") -> None:
        rig.mesh(tube([ctr + R @ np.array(a), ctr + R @ np.array(b)], rw=r, n=8, per=1),
                 color, shade=2.0, local=center)

    part((0, 0, 0), (0.095, 0.05, 0.055), "#f4f1ea")
    part((-0.02, 0.022, 0.012), (0.04, 0.035, 0.04), "#18181c")
    part((0.12, 0, 0.02), (0.042, 0.03, 0.035), "#f4f1ea")
    part((0.158, 0, 0.012), (0.018, 0.026, 0.02), "#f4aab5")
    for s in (-1, 1):
        part((0.115, s * 0.03, 0.055), (0.012, 0.012, 0.025), "#efe3bf")
        for dx in (-0.06, 0.06):
            limb((dx, s * 0.028, -0.03), (dx, s * 0.03, -0.1), [0.016, 0.012])
    limb((-0.095, 0, 0.01), (-0.14, 0, -0.06), [0.01, 0.008])


def ufo_items(half: ArrayLike, sim_color: str) -> tuple[list[Any], Array]:
    """Ovni con alienígena, cúpula, haz tractor y vaca abducida"""
    rig = Rig(half, roles=(0, 1, 2), ref=(1.0, 1.0, 0.58), floor=0.30)
    ALL = (0, 1, 2)
    qc = QtGui.QColor(sim_color)
    sim_rgb = np.array([qc.redF(), qc.greenF(), qc.blueF()])

    # casco lenticular
    V, F = lathe(_hull_profile(), n=96, axis=2)
    rig.mesh((V, F), vc=_hull_colors(V, sim_rgb), stretch=ALL)

    # luces del borde (alternan color de la simulación / ámbar)
    for i in range(24):
        a = 2 * math.pi * i / 24
        c = (0.985 * math.cos(a), 0.985 * math.sin(a), 0.022)
        rig.mesh(blob(c, (0.036, 0.036, 0.036), 6, 10), sim_color if i % 2 == 0 else "#fff0a8",
                 shade=1.3, emissive=True, local=c)
    for i in range(10):                                        # luces de la panza
        a = 2 * math.pi * (i + 0.5) / 10
        c = (0.56 * math.cos(a), 0.56 * math.sin(a), -0.195)
        rig.mesh(blob(c, (0.03, 0.03, 0.016), 6, 10), "#bff6ff", emissive=True, local=c)

    # emisor del haz
    ph = np.linspace(0, 2 * math.pi, 48)
    ring = np.column_stack([0.19 * np.cos(ph), 0.19 * np.sin(ph), np.full_like(ph, -0.205)])
    rig.mesh(tube(ring, rw=0.04, n=10, per=1, caps=False), "#30384a", shade=2.0, stretch=ALL)
    e = (0, 0, -0.225)
    rig.mesh(blob(e, (0.17, 0.17, 0.035), 8, 24), "#d8fbff", emissive=True, local=e, stretch=(0, 1))

    # tren de aterrizaje: tres patas con patín
    for a in (math.pi / 2, math.pi / 2 + 2 * math.pi / 3, math.pi / 2 + 4 * math.pi / 3):
        ca, sa = math.cos(a), math.sin(a)
        P = [(0.62 * ca, 0.62 * sa, -0.14), (0.73 * ca, 0.73 * sa, -0.32), (0.77 * ca, 0.77 * sa, -0.42)]
        rig.mesh(tube(P, rw=[0.036, 0.026, 0.022], n=10, per=4), "#6a7283", shade=2.0, stretch=ALL)
        fc = (0.78 * ca, 0.78 * sa, -0.455)
        foot = lathe([(0.02, 0.0), (0.02, 0.06), (0.0, 0.095), (-0.02, 0.095), (-0.02, 0.0)],
                     n=18, axis=2, center=fc)
        rig.mesh(foot, "#2d3340", shade=1.6, local=fc)

    # alienígena en la cabina (escala local isótropa: no se deforma con la inercia)
    AC = (0.0, 0.0, 0.34)
    A = dict(local=AC, k="min")

    def al(vf: tuple[Array, IntArray], color: str, **kw) -> None:
        kw.setdefault("shade", 2.0)
        rig.mesh(vf, color, **A, **kw)

    al(blob((0, 0, -0.06 + 0.34), (0.075, 0.07, 0.09), 10, 16), "#6fcf52")
    al(blob((0, 0, 0.055 + 0.34), (0.108, 0.098, 0.122), 14, 22), "#86e060")
    for s in (-1, 1):
        R = rot_axis((1, 0, 0), -s * 28)
        ec = (0.088, s * 0.052, 0.068 + 0.34 - 0.04)
        al(blob(ec, (0.02, 0.034, 0.06), 10, 14, rot=R), "#08090c", shade=1.0)
        gc = (0.104, s * 0.046, 0.082 + 0.34 - 0.04)
        al(blob(gc, (0.008, 0.008, 0.008), 6, 8), "#ffffff", emissive=True)
        base = (0.0, s * 0.045, 0.34 + 0.165)                      # antena con bulbo luminoso
        top = (0.0, s * 0.085, 0.34 + 0.255)
        al(tube([base, (0.0, s * 0.06, 0.34 + 0.21), top], rw=[0.009, 0.007], n=8, per=4), "#6fcf52")
        al(blob(top, (0.022, 0.022, 0.022), 8, 12), sim_color, emissive=True, shade=1.3)
        al(tube([(0.01, s * 0.07, 0.31), (0.075, s * 0.09, 0.285), (0.15, s * 0.07, 0.265)],
                rw=[0.017, 0.014, 0.013], n=8, per=4), "#6fcf52")   # brazo hacia la consola
    ys = np.linspace(-0.04, 0.04, 9)                               # sonrisa
    zs = 0.34 + 0.012 - 0.018 * (1 - (ys / 0.04) ** 2)
    xs = 0.098 * np.sqrt(np.maximum(1 - (ys / 0.098) ** 2 - ((zs - 0.395) / 0.122) ** 2, 0.01)) + 0.003
    rig.line(np.column_stack([xs, ys, zs]), "#14361a", width=2, **A)
    cb = slab([(-0.1, -0.035), (0.1, -0.035), (0.1, 0.035), (-0.1, 0.035)], 0.15, 0.21, axis=0,
              center=(0, 0.255))
    al(cb, "#2a2f3b", shade=1.6)                                   # consola de mandos
    for j, col in enumerate(("#ff4d4d", "#5dff8a", "#ffe45c")):
        al(blob((0.2125, (j - 1) * 0.05, 0.268), (0.008, 0.012, 0.012), 6, 8), col, emissive=True)

    # vaquita abducida
    _mini_cow(rig, (0.04, 0.0, -0.66))

    # translúcidos (se dibujan al final): haz tractor y cúpula
    beam = orient_out(*lathe([(-0.235, 0.15), (-0.52, 0.29), (-0.86, 0.45)], n=48, axis=2),
                      (0, 0, -0.55))
    rig.mesh(beam, "#9ff3ff", alpha=0.16, glass=True, emissive=True)
    dome = blob((0, 0, 0.19), (0.405, 0.405, 0.44), 20, 40, theta=(0.0, math.pi / 2))
    rig.mesh(dome, sim_color, alpha=0.42, glass=True, shade=1.3)
    dring = np.column_stack([0.41 * np.cos(ph), 0.41 * np.sin(ph), np.full_like(ph, 0.19)])
    rig.mesh(tube(dring, rw=0.026, n=10, per=1, caps=False), "#e0b84e", shade=1.6, stretch=ALL)

    return rig.finish(), rig.extent
