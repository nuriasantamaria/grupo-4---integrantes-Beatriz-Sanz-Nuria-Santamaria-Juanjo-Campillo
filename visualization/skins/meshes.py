"""Primitivas de malla 3D (tubos, tornos, losas…) y el ensamblador Rig"""

from __future__ import annotations

import math
from typing import Any, Sequence, Union

import numpy as np
from numpy.typing import NDArray
from PySide6 import QtGui
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_CULL_FACE
from pyqtgraph.opengl.GLGraphicsItem import GLOptions

from space_physics import Array, ArrayLike


IntArray = NDArray[np.intp]


Mesh = tuple[Array, IntArray]            # (vértices, caras triangulares)


# Cada skin se modela en un marco canónico y un Rig lo estira sobre los ejes principales

# translúcido con descarte de caras traseras: una sola capa de vidrio, sin rayas
_GLASS = dict(GLOptions["translucent"])
_GLASS[GL_CULL_FACE] = True


def orient_out(V: Array, F: IntArray, center: ArrayLike) -> Mesh:
    """Superficies abiertas: gira las caras para que las normales miren fuera de center"""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    d = V[F].mean(1) - np.asarray(center, float)
    return (V, F[:, ::-1].copy()) if np.mean(np.einsum("ij,ij->i", n, d)) < 0 else (V, F)


def _volume(V: Array, F: IntArray) -> float:
    return np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6.0


def _closed(V: Array, F: IntArray) -> Mesh:
    """Orienta las caras hacia fuera (volumen con signo positivo)"""
    return (V, F[:, ::-1].copy()) if _volume(V, F) < 0 else (V, F)


def _grid(X: Array, wrap: bool = True) -> Mesh:
    """Triangula una rejilla de puntos"""
    nr, nc = X.shape[:2]
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc if wrap else nc - 1), indexing="ij")
    j1 = (j + 1) % nc
    a, b, c, d = i * nc + j, i * nc + j1, (i + 1) * nc + j1, (i + 1) * nc + j
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    return X.reshape(-1, 3), F


def _capped(X: Array, cap0: bool = True, cap1: bool = True) -> Mesh:
    """Malla a partir de una rejilla de anillos (nr, nc, 3), con tapas opcionales"""
    V, F = _grid(X)
    nr, nc = X.shape[:2]
    parts_v, parts_f = [V], [F]
    n = len(V)
    j = np.arange(nc)
    if cap0:
        parts_v.append(X[0].mean(0, keepdims=True))
        parts_f.append(np.stack([np.full(nc, n), (j + 1) % nc, j], -1))
        n += 1
    if cap1:
        parts_v.append(X[-1].mean(0, keepdims=True))
        parts_f.append(np.stack([np.full(nc, n), (nr - 1) * nc + j, (nr - 1) * nc + (j + 1) % nc], -1))
    return _closed(np.vstack(parts_v), np.vstack(parts_f))


def merge(*parts) -> Mesh:
    """Une varias mallas en una sola reindexando las caras"""
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def blob(center: ArrayLike, radii: ArrayLike, rows: int = 14, cols: int = 24, rot: Array | None = None,
         theta: tuple[float, float] = (0.0, math.pi)) -> Mesh:
    """Elipsoide, o casquete si theta no es completo"""
    th = np.linspace(theta[0], theta[1], rows + 1)
    ph = np.linspace(0, 2 * math.pi, cols, endpoint=False)
    T, P = np.meshgrid(th, ph, indexing="ij")
    s = np.maximum(np.sin(T), 1e-3)
    X = np.stack([s * np.cos(P), s * np.sin(P), np.cos(T)], -1) * np.asarray(radii, float)
    if rot is not None:
        X = X @ np.asarray(rot, float).T
    X = X + np.asarray(center, float)
    V, F = _grid(X)
    if theta == (0.0, math.pi):
        return _closed(V, F)
    return orient_out(V, F, center)


def rot_axis(axis: ArrayLike, deg: float) -> Array:
    """Matriz de rotación de deg grados alrededor de un eje"""
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def smooth_path(pts: ArrayLike, per: int = 8) -> Array:
    """Suaviza una polilínea con Catmull-Rom"""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return P
    Q = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]])
    out = []
    for i in range(1, len(Q) - 2):
        p0, p1, p2, p3 = Q[i - 1], Q[i], Q[i + 1], Q[i + 2]
        for t in np.linspace(0, 1, per, endpoint=False):
            out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append(P[-1])
    return np.array(out)


def _interp_radii(r: ArrayLike, n: int) -> Array:
    r = np.asarray(r, float)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r)), r)


def tube(path: ArrayLike, rw: ArrayLike, rh: ArrayLike | None = None, n: int = 16,
         side: Sequence[int] | None = None, per: int = 8, caps: bool = True) -> Mesh:
    """Barrido de una elipse a lo largo de un camino, con dirección fija del semieje ancho si se da side"""
    P = smooth_path(path, per)
    m = len(P)
    rw = _interp_radii(np.atleast_1d(rw), m)
    rh = rw if rh is None else _interp_radii(np.atleast_1d(rh), m)
    T = np.gradient(P, axis=0)
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    A = np.zeros_like(P)
    if side is not None:
        s = np.asarray(side, float)
        for i in range(m):
            a = s - s.dot(T[i]) * T[i]
            A[i] = a / np.linalg.norm(a)
    else:
        a = np.eye(3)[np.argmin(np.abs(T[0]))]
        a = a - a.dot(T[0]) * T[0]
        A[0] = a / np.linalg.norm(a)
        for i in range(1, m):
            a = A[i - 1] - A[i - 1].dot(T[i]) * T[i]
            A[i] = a / np.linalg.norm(a)
    B = np.cross(T, A)
    ph = np.linspace(0, 2 * math.pi, n, endpoint=False)
    c, s_ = np.cos(ph), np.sin(ph)
    X = (P[:, None, :] + (rw[:, None, None] * c[None, :, None]) * A[:, None, :]
         + (rh[:, None, None] * s_[None, :, None]) * B[:, None, :])
    return _capped(X, caps, caps)


def lathe(profile: ArrayLike, n: int = 32, axis: int = 2, center: ArrayLike = (0, 0, 0), caps: bool = False) -> Mesh:
    """Superficie de revolución de un perfil (t, r) con t a lo largo del eje"""
    prof = np.asarray(profile, float)
    u, v = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    ph = np.linspace(0, 2 * math.pi, n, endpoint=False)
    X = np.zeros((len(prof), n, 3))
    X[:, :, axis] = prof[:, 0:1]
    r = np.maximum(prof[:, 1:2], 1e-4)
    X[:, :, u] = r * np.cos(ph)
    X[:, :, v] = r * np.sin(ph)
    X += np.asarray(center, float)
    return _capped(X, caps, caps)


def slab(poly: Union[Array, Sequence[Sequence[float]]], lo: float, hi: float, axis: int = 0,
         center: ArrayLike = (0, 0)) -> Mesh:
    """Prisma convexo: polígono (u, v) extruido entre lo y hi a lo largo de axis"""
    poly = np.asarray(poly, float) + np.asarray(center, float)
    u, v = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    k = len(poly)

    def pts(z: float) -> Array:
        Q = np.zeros((k, 3))
        Q[:, axis], Q[:, u], Q[:, v] = z, poly[:, 0], poly[:, 1]
        return Q

    V = np.vstack([pts(lo), pts(hi), pts(lo).mean(0, keepdims=True), pts(hi).mean(0, keepdims=True)])
    j = np.arange(k)
    j1 = (j + 1) % k
    F = [np.stack([j, j1, k + j1], -1), np.stack([j, k + j1, k + j], -1),
         np.stack([np.full(k, 2 * k), j1, j], -1), np.stack([np.full(k, 2 * k + 1), k + j, k + j1], -1)]
    return _closed(V, np.vstack(F))


def ngon(n: int, R: float, phase_deg: float = 90.0) -> Array:
    """Vértices de un polígono regular de n lados y radio R"""
    a = np.radians(phase_deg + 360.0 / n * np.arange(n))
    return np.column_stack([R * np.cos(a), R * np.sin(a)])


def smoothstep(e0: float, e1: float, x: Any) -> Any:
    """Transición suave de 0 a 1 entre e0 y e1"""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class Rig:
    """Lleva el modelo canónico a los ejes principales; las piezas se estiran y los detalles conservan su forma"""
    SIZE = 1.25

    def __init__(self, half: ArrayLike, roles: Sequence[int], ref: Sequence[float], floor: float) -> None:
        h = np.asarray(half, float)
        order = np.argsort(h)
        rank_axis = [order[2], order[1], order[0]]
        hh = np.maximum(h / h.max(), floor)
        self.axis = [rank_axis[r] for r in roles]
        self.s = np.array([self.SIZE * hh[self.axis[i]] / ref[i] for i in range(3)])
        self.k = float(np.prod(self.s) ** (1 / 3))
        self.kmin = float(self.s.min())
        P = np.zeros((3, 3))
        for i in range(3):
            P[self.axis[i], i] = 1.0
        self.P = P
        self.flip = np.linalg.det(P) < 0
        ext = np.zeros(3)
        for i in range(3):
            ext[self.axis[i]] = self.SIZE * hh[self.axis[i]]
        self.extent = ext
        self._groups, self._lines = {}, {}

    def _xf(self, V: Array, local: ArrayLike | None = None, stretch: Sequence[int] = (),
            k: Union[int, str, None] = None) -> Array:
        """Pasa vértices del marco canónico a los ejes del cuerpo, con escala local opcional"""
        V = np.asarray(V, float)
        if local is None:
            return (V * self.s) @ self.P.T
        c = np.asarray(local, float)
        kk = self.k if k is None else (self.kmin if k == "min" else k)
        sv = np.array([self.s[i] if i in stretch else kk for i in range(3)])
        return ((c * self.s) + (V - c) * sv) @ self.P.T

    # acumulación por material
    def mesh(self, vf: tuple[Array, IntArray], color: str = "#ffffff", alpha: float = 1.0, shade: float = 1.0,
             vc: Array | None = None, smooth: bool = True, emissive: bool = False, glass: bool = False,
             local: ArrayLike | None = None, stretch: Sequence[int] = (),
             k: Union[int, str, None] = None) -> None:
        """Acumula una malla en el grupo de su material"""
        V, F = vf
        Vb = self._xf(V, local, stretch, k)
        if self.flip:
            F = F[:, ::-1]
        if vc is None:
            c = QtGui.QColor(color)
            vc = np.tile([c.redF() * shade, c.greenF() * shade, c.blueF() * shade, alpha], (len(Vb), 1))
        key = ("glass", emissive) if glass else (("emis",) if emissive else (("smooth",) if smooth else ("flat",)))
        self._groups.setdefault(key, []).append((Vb, F, np.asarray(vc, float)))

    def line(self, pts: Array, color: str = "#ffffff", alpha: float = 1.0, width: float = 1.5, mode: str = "line_strip",
             local: ArrayLike | None = None, stretch: Sequence[int] = (), k: Union[int, str, None] = None) -> None:
        """Acumula una polilínea o segmentos con su color"""
        P = self._xf(pts, local, stretch, k)
        if mode == "line_strip":
            P = np.repeat(P, 2, axis=0)[1:-1]
        c = QtGui.QColor(color)
        self._lines.setdefault(float(width), []).append(
            (P, np.tile([c.redF(), c.greenF(), c.blueF(), alpha], (len(P), 1))))

    def finish(self) -> list[Any]:
        """Fusiona la geometría acumulada: opacos, líneas, emisivos y vidrio, en ese orden"""
        items = [self._merged(key) for key in (("smooth",), ("flat",)) if key in self._groups]
        for width, segs in self._lines.items():
            items.append(gl.GLLinePlotItem(pos=np.vstack([p for p, _ in segs]), mode="lines",
                                           color=np.vstack([c for _, c in segs]), width=width,
                                           antialias=True))
        for key in (("emis",), ("glass", False), ("glass", True)):
            if key in self._groups:
                items.append(self._merged(key))
        return items

    def _merged(self, key: Any) -> gl.GLMeshItem:
        """Funde en un solo GLMeshItem las mallas de un grupo"""
        Vs, Fs, Cs, off = [], [], [], 0
        for V, F, C in self._groups[key]:
            Vs.append(V)
            Fs.append(F + off)
            Cs.append(C)
            off += len(V)
        md = gl.MeshData(vertexes=np.vstack(Vs), faces=np.ascontiguousarray(np.vstack(Fs)),
                         vertexColors=np.vstack(Cs))
        kind = key[0]
        return gl.GLMeshItem(meshdata=md, smooth=(kind != "flat"), color=(1, 1, 1, 1),
                             shader=None if (kind == "emis" or (kind == "glass" and key[1])) else "shaded",
                             glOptions=_GLASS if kind == "glass" else "opaque")
