"""Interfaz gráfica interactiva (PySide6 + pyqtgraph) para simular la rotación libre
de un sólido rígido: integra las ecuaciones de Euler con distintos esquemas numéricos,
muestra las órbitas en el espacio de fases de Poinsot, la deriva de la energía y del
momento angular, la orientación del satélite en 3D y el plano de fases alrededor de
cada punto de equilibrio para analizar su estabilidad."""

import math
import sys
import time
from dataclasses import dataclass, replace
from functools import partial

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from OpenGL.GL import GL_CULL_FACE
from pyqtgraph.opengl.GLGraphicsItem import GLOptions

import numerical_engine as ne
from space_physics import (euler_equations, kinetic_energy, angular_momentum_sq,
                           equilibrium_analysis)


def quat_to_matrix(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def sorted_axes(I):
    a, b, c = np.argsort(np.asarray(I, dtype=float))
    return int(a), int(b), int(c)


def is_degenerate(I, tol=1e-6):
    Is = np.sort(np.asarray(I, dtype=float))
    return (Is[1] - Is[0] < tol) or (Is[2] - Is[1] < tol)


def poinsot_semiaxes(I, w0):
    I = np.asarray(I, dtype=float)
    twoT = 2 * kinetic_energy(w0, I)
    L = np.sqrt(angular_momentum_sq(w0, I))
    return np.sqrt(twoT / I), L / I


def regime(I, w0):
    if is_degenerate(I):
        return "trompo simétrico"
    I = np.asarray(I, dtype=float)
    _, b, _ = sorted_axes(I)
    twoT = 2 * kinetic_energy(w0, I)
    L2 = angular_momentum_sq(w0, I)
    D = L2 - twoT * I[b]
    if abs(D) < 1e-3 * twoT * I[b]:
        return "≈ separatriz (inestable)"
    return "precesa eje mayor (estable)" if D > 0 else "precesa eje menor (estable)"


def separatrix_curves(I, twoT, n_pts=300):
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    k = math.sqrt(Ia * (Ib - Ia) / (Ic * (Ic - Ib)))
    s = np.linspace(0, 2 * np.pi, n_pts)
    A = math.sqrt(twoT / (Ia + Ic * k * k))
    B = math.sqrt(twoT / Ib)
    curves = []
    for sign in (+1, -1):
        P = np.zeros((n_pts, 3))
        P[:, a] = A * np.cos(s)
        P[:, b] = B * np.sin(s)
        P[:, c] = sign * k * A * np.cos(s)
        curves.append(P)
    return curves


def polhode_family(I, twoT, n_levels=5, n_pts=200):
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    s = np.linspace(0, 2 * np.pi, n_pts)
    curves = []
    levels = np.linspace(0.15, 0.9, n_levels)

    for f in levels:
        L2 = twoT * (Ia + f * (Ib - Ia))
        wc = math.sqrt((L2 - twoT * Ia) / (Ic * (Ic - Ia))) * np.cos(s)
        wb = math.sqrt((L2 - twoT * Ia) / (Ib * (Ib - Ia))) * np.sin(s)
        wa = np.sqrt(np.maximum(twoT - Ib * wb ** 2 - Ic * wc ** 2, 0) / Ia)
        for sign in (+1, -1):
            P = np.zeros((n_pts, 3))
            P[:, a], P[:, b], P[:, c] = sign * wa, wb, wc
            curves.append(P)

    for f in levels:
        L2 = twoT * (Ic - f * (Ic - Ib))
        wa = math.sqrt((twoT * Ic - L2) / (Ia * (Ic - Ia))) * np.cos(s)
        wb = math.sqrt((twoT * Ic - L2) / (Ib * (Ic - Ib))) * np.sin(s)
        wc = np.sqrt(np.maximum(twoT - Ia * wa ** 2 - Ib * wb ** 2, 0) / Ic)
        for sign in (+1, -1):
            P = np.zeros((n_pts, 3))
            P[:, a], P[:, b], P[:, c] = wa, wb, sign * wc
            curves.append(P)
    return curves


def body_half_sizes(I, min_rel=0.15):
    I1, I2, I3 = (float(x) for x in I)
    s = np.sqrt(np.maximum([I2 + I3 - I1, I1 + I3 - I2, I1 + I2 - I3], 0.0))
    s = s / s.max()
    return np.maximum(s, min_rel)

def rigid_body_rhs(u, I_mat):
    w1, w2, w3, q0, q1, q2, q3 = u
    dw = euler_equations(u[:3], I_mat)
    return np.array([dw[0], dw[1], dw[2],
                     0.5 * (-q1 * w1 - q2 * w2 - q3 * w3),
                     0.5 * (q0 * w1 + q2 * w3 - q3 * w2),
                     0.5 * (q0 * w2 + q3 * w1 - q1 * w3),
                     0.5 * (q0 * w3 + q1 * w2 - q2 * w1)])


def make_rhs(I):
    return partial(rigid_body_rhs, I_mat=np.diag(np.asarray(I, dtype=float)))


M_EULER = "Euler explícito"
M_BE = "Euler implícito (inverso)"
M_CN = "Crank-Nicolson (implícito)"
M_RK4 = "Runge-Kutta 4"

CN_TOL = 1e-12

SCHEMES = {
    M_EULER: ne.euler,
    M_BE: partial(ne.implicit_euler, tol=CN_TOL),
    M_CN: partial(ne.crank_nicolson, tol=CN_TOL),
    M_RK4: ne.runge_kutta_4,
}

ORDERS = {M_EULER: 1, M_BE: 1, M_CN: 2, M_RK4: 4}

AMPLIFICATION = {
    M_EULER: ne.amplification_euler,
    M_BE: ne.amplification_implicit_euler,
    M_CN: ne.amplification_crank_nicolson,
    M_RK4: ne.amplification_runge_kutta_4,
}

METHOD_SHORT = {M_EULER: "Euler", M_BE: "Euler imp.", M_CN: "C-N", M_RK4: "RK4"}

DIVERGENCE_LIMIT = 1e6


def integrate(method, y0, dt, I, n):
    with np.errstate(all="ignore"):
        out = ne.integrate(SCHEMES[method], make_rhs(I), np.asarray(y0, dtype=float), dt, n)
        bad = ~(np.abs(out[:, :3]).sum(axis=1) < DIVERGENCE_LIMIT)
    if bad.any():
        out[np.argmax(bad):] = np.nan
    return out


def richardson_analysis(cfg, T):
    scheme = SCHEMES[cfg.method]
    f = make_rhs(cfg.I)
    u0 = np.array([*cfg.w0, 1.0, 0.0, 0.0, 0.0])
    n = max(1, int(round(T / cfg.dt)))
    p = ORDERS[cfg.method]
    with np.errstate(all="ignore"):
        p_est = ne.scheme_order(scheme, f, u0, T, n)
        err = ne.richardson_error(scheme, f, u0, T, n, p)
        u_extr = ne.richardson_extrapolation(scheme, f, u0, T, n, p)
    return {"n": n, "p": p, "p_est": float(p_est), "err": float(err), "u_extr": u_extr}


def surrounded_axis(I, w0):
    if is_degenerate(I):
        return None
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    twoT = 2 * kinetic_energy(w0, I)
    D = angular_momentum_sq(w0, I) - twoT * I[b]
    if abs(D) < 1e-3 * twoT * I[b]:
        return None
    return c if D > 0 else a


def equilibrium_diagnosis(cfg):
    I, w0 = np.asarray(cfg.I, dtype=float), np.asarray(cfg.w0, dtype=float)
    twoT = 2 * float(kinetic_energy(w0, I))
    eqs = equilibrium_analysis(I, twoT)
    axis = surrounded_axis(I, w0)
    amp = None
    if axis is not None and eqs[axis]["type"] == "centro":
        beta = float(np.max(np.abs(eqs[axis]["eigvals"].imag)))
        amp = abs(AMPLIFICATION[cfg.method](1j * beta * cfg.dt))
    return {"eqs": eqs, "axis": axis, "amp": amp}


SURFACE_TOL = 1e-3


def surface_mismatch(ref_cfg, cfg, tol=SURFACE_TOL):
    """None si la trayectoria de cfg yace (a tol) sobre las superficies de ref_cfg
    (elipsoides de energía y momento); si no, dict(same_I, dT, dL) con las diferencias
    relativas de energía y de ‖L‖²."""
    Ir, Ic = np.asarray(ref_cfg.I, dtype=float), np.asarray(cfg.I, dtype=float)
    Tr, Tc = float(kinetic_energy(ref_cfg.w0, Ir)), float(kinetic_energy(cfg.w0, Ic))
    Lr, Lc = (float(angular_momentum_sq(ref_cfg.w0, Ir)),
              float(angular_momentum_sq(cfg.w0, Ic)))

    def rel(a, b):
        return (b / a - 1.0) if a > 0 else (0.0 if b == 0 else math.inf)

    same_I = bool(np.allclose(Ir, Ic, rtol=1e-9, atol=1e-12))
    dT, dL = rel(Tr, Tc), rel(Lr, Lc)
    if same_I and abs(dT) <= tol and abs(dL) <= tol:
        return None
    return {"same_I": same_I, "dT": dT, "dL": dL}


STABILITY_COLORS = {"estable": "#69F0AE", "marginal": "#80D8FF", "inestable": "#FF6B6B"}


def regime_color(r):
    if "separatriz" in r:
        return STABILITY_COLORS["inestable"]
    if "estable" in r:
        return STABILITY_COLORS["estable"]
    return STABILITY_COLORS["marginal"]


def chart_axes(k):
    return (k + 1) % 3, (k + 2) % 3


def reduced_flow(I, twoT, k, sign, X, Y):
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    W = np.zeros((3,) + X.shape)
    W[i], W[j] = X, Y
    W[k] = sign * np.sqrt(np.maximum(twoT - I[i] * X ** 2 - I[j] * Y ** 2, 0) / I[k])
    dW = euler_equations(W, np.diag(I))
    return dW[i], dW[j]


def phase_plane_orbits(I, twoT, k, n_levels=9, n_pts=400):
    if is_degenerate(I) or twoT <= 0:
        return [], []
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    fr = np.linspace(0.12, 1.0, n_levels) ** 2
    qa, qb = a * twoT / I[i], b * twoT / I[j]
    if a * b > 0:
        levels = fr * (qa if abs(qa) > abs(qb) else qb)
        seps = []
    else:
        levels = np.concatenate([fr * qa, fr * qb])
        seps = chart_level_curve(I, twoT, k, 0.0, n_pts)
    orbits = [P for c in levels for P in chart_level_curve(I, twoT, k, c, n_pts)]
    return orbits, seps


def chart_level_curve(I, twoT, k, c, n_pts=400):
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    xm, ym = math.sqrt(twoT / I[i]), math.sqrt(twoT / I[j])

    def clip(x, y):
        P = np.column_stack([x, y]).astype(float)
        P[I[i] * P[:, 0] ** 2 + I[j] * P[:, 1] ** 2 > twoT * (1 + 1e-9)] = np.nan
        return P

    if a * b > 0:
        if c / a <= 0:
            return []
        s = np.linspace(0, 2 * np.pi, n_pts)
        return [clip(math.sqrt(c / a) * np.cos(s), math.sqrt(c / b) * np.sin(s))]

    swap = a < 0
    A, B, XM, YM = (b, a, ym, xm) if swap else (a, b, xm, ym)
    if c > 0:
        r, q = math.sqrt(c / A), math.sqrt(c / -B)
        s = np.linspace(-1, 1, n_pts) * math.asinh(YM / q)
        raw = [(sg * r * np.cosh(s), q * np.sinh(s)) for sg in (1, -1)]
    elif c < 0:
        r, q = math.sqrt(c / B), math.sqrt(-c / A)
        s = np.linspace(-1, 1, n_pts) * math.asinh(XM / q)
        raw = [(q * np.sinh(s), sg * r * np.cosh(s)) for sg in (1, -1)]
    else:
        m = math.sqrt(A / -B)
        u = np.linspace(-XM, XM, n_pts)
        raw = [(u, m * u), (u, -m * u)]
    return [clip(v, u) if swap else clip(u, v) for u, v in raw]


def quiver_segments(X, Y, U, V, length, head=0.38, angle=26.0):
    X, Y, U, V = (np.ravel(z) for z in (X, Y, U, V))
    sp = np.hypot(U, V)
    ok = sp > 1e-9 * max(sp.max(initial=0.0), 1e-300)
    if not ok.any():
        return np.zeros((0, 2))
    d = np.column_stack([U[ok], V[ok]]) / sp[ok, None]
    P = np.column_stack([X[ok], Y[ok]])
    tail, tip = P - 0.5 * length * d, P + 0.5 * length * d
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    left = tip - head * length * np.column_stack([ca * d[:, 0] - sa * d[:, 1], sa * d[:, 0] + ca * d[:, 1]])
    right = tip - head * length * np.column_stack([ca * d[:, 0] + sa * d[:, 1], -sa * d[:, 0] + ca * d[:, 1]])
    return np.stack([tail, tip, tip, left, tip, right], axis=1).reshape(-1, 2)


@dataclass
class SimConfig:
    name: str
    color: str
    enabled: bool = True
    method: str = M_RK4
    I: tuple = (1.0, 2.0, 3.0)
    w0: tuple = (0.01, 2.0, 0.01)
    dt: float = 0.02


class Simulation:

    CHUNK = 8192
    MAX_STEPS_PER_CALL = 20000

    def __init__(self, cfg: SimConfig):
        self.cfg = replace(cfg)
        self.I = np.array(cfg.I, dtype=float)
        self.reset()

    def reset(self):
        w0 = np.array(self.cfg.w0, dtype=float)
        self.y = (*w0, 1.0, 0.0, 0.0, 0.0)
        self.n = 1
        self.diverged = False
        self.T0 = float(kinetic_energy(w0, self.I))
        self.L0 = float(np.sqrt(angular_momentum_sq(w0, self.I)))
        self.Ls0 = self.I * w0
        self._Y = np.empty((self.CHUNK, 7))
        self._dT = np.empty(self.CHUNK)
        self._dL = np.empty(self.CHUNK)
        self._Y[0] = self.y
        self._dT[0] = 0.0
        self._dL[0] = 0.0

    @property
    def t(self):
        return self.n_steps * self.cfg.dt

    @property
    def n_steps(self):
        return self.n - 1

    @property
    def times(self):
        return np.arange(self.n) * self.cfg.dt

    @property
    def Y(self):
        return self._Y[:self.n]

    @property
    def dT(self):
        return self._dT[:self.n]

    @property
    def dL(self):
        return self._dL[:self.n]

    def _grow(self, extra):
        need = self.n + extra
        if need <= len(self._Y):
            return
        cap = max(need, 2 * len(self._Y))
        for name in ("_Y", "_dT", "_dL"):
            old = getattr(self, name)
            new = np.empty((cap,) + old.shape[1:])
            new[:self.n] = old[:self.n]
            setattr(self, name, new)

    def advance_to(self, t_target):
        n = int(math.ceil(t_target / self.cfg.dt - 1e-9)) - self.n_steps
        if n > 0:
            self.advance(min(n, self.MAX_STEPS_PER_CALL))

    def advance(self, n):
        if self.diverged or n <= 0:
            return
        Y = integrate(self.cfg.method, self.y, self.cfg.dt, self.I, n)[1:]
        ok = np.isfinite(Y).all(axis=1)
        if not ok.all():
            Y = Y[:np.argmin(ok)]
            self.diverged = True
        if len(Y) == 0:
            return
        self._grow(len(Y))
        sl = slice(self.n, self.n + len(Y))
        self._Y[sl] = Y
        self._dT[sl] = kinetic_energy(Y[:, :3], self.I) - self.T0
        self._dL[sl] = np.sqrt(angular_momentum_sq(Y[:, :3], self.I)) - self.L0
        self.n += len(Y)
        self.y = tuple(Y[-1])

    def quat_norm_error(self):
        return float(np.linalg.norm(self.y[3:]) - 1.0)


MAX_PREVIEW_STEPS = 300_000


def run_preview(cfg: SimConfig, T_total, should_cancel=None):
    """Trayectoria completa hasta T_total. Se integra por bloques para poder cancelarla;
    devuelve None si should_cancel() pasa a True."""
    sim = Simulation(cfg)
    n = min(int(math.ceil(T_total / cfg.dt)), MAX_PREVIEW_STEPS)
    left = n
    while left > 0 and not sim.diverged:
        if should_cancel is not None and should_cancel():
            return None
        m = min(PREVIEW_CHUNK, left)
        sim.advance(m)
        left -= m
        time.sleep(0.001)          # cede el GIL: la interfaz sigue fluida mientras se calcula
    return {"t": sim.times.copy(), "W": sim.Y[:, :3].copy(),
            "dT": sim.dT.copy(), "dL": sim.dL.copy(),
            "diverged": sim.diverged, "truncated": n == MAX_PREVIEW_STEPS}


PREVIEW_CHUNK = 2000


class PreviewWorker(QtCore.QThread):
    """Calcula las previsualizaciones fuera del hilo de la interfaz."""

    done = QtCore.Signal(int, object, float)       # token, {k: preview}, segundos

    def __init__(self, token, jobs, T):
        super().__init__()
        self.token, self.jobs, self.T = token, jobs, T
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        t0 = time.perf_counter()
        out = {}
        for k, cfg in self.jobs:
            r = run_preview(cfg, self.T, lambda: self._cancel)
            if r is None:
                return
            out[k] = r
        self.done.emit(self.token, out, time.perf_counter() - t0)


SIM_COLORS = ["#7FE7FF", "#FFA07A", "#C6FF7A"]
AXIS_COLORS = ["#FF6B6B", "#69F0AE", "#D1A8FF"]
BG = "#3a3d42"
PANEL = "#2f3236"
VIEW_BG = "#26282c"
FIELD = "#222427"
BORDER = "#5a5f66"
BUTTON = "#4a4e55"
ACCENT = "#3b82f6"
TEXT = "#f2f2f2"
MUTED = "#b8bcc2"

UI_FONT_CANDIDATES = ["Bahnschrift", "Segoe UI Variable Display", "Segoe UI", "Arial"]
MONO_FONT_CANDIDATES = ["Cascadia Mono", "Consolas", "Courier New"]
UI_FONT = "Segoe UI"
MONO = "Consolas, monospace"


def pick_font(candidates):
    families = set(QtGui.QFontDatabase.families())
    return next((f for f in candidates if f in families), QtGui.QFont().defaultFamily())


def axis_label_style():
    return {"font-family": UI_FONT, "font-size": "11pt", "color": TEXT}


def pill(text, color):
    return (f"<span style='background-color:{color}; color:#04123a; font-weight:700'>"
            f"&nbsp;{text}&nbsp;</span>")


def badge_css(color, size=13):
    c = QtGui.QColor(color)
    return (f"QLabel {{ background: rgba({c.red()}, {c.green()}, {c.blue()}, 55); color: #ffffff;"
            f" border: 2px solid {color}; border-radius: 14px; padding: 5px 14px;"
            f" font-weight: 700; font-size: {size}px; }}")


def rgba(hex_color, alpha=1.0, factor=1.0):
    c = QtGui.QColor(hex_color)
    return (min(c.redF() * factor, 1.0), min(c.greenF() * factor, 1.0),
            min(c.blueF() * factor, 1.0), alpha)


_BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
                       [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
                       [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
_BOX_EDGES = np.array([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                       (0, 4), (1, 5), (2, 6), (3, 7)]).ravel()
PANEL_COLOR = "#1f3f8f"
PANEL_LINES = (0.62, 0.72, 1.0, 0.75)
METAL_COLOR = "#cfd4de"


def box_items(center, half, color, shades=(1.0, 1.0, 1.0), edge_color=(1, 1, 1, 0.6)):
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    V = center + V * half
    C = np.array([rgba(color, 1.0, shades[f // 4]) for f in range(12)])
    mesh = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=V, faces=_BOX_FACES, faceColors=C),
                         smooth=False, shader="shaded", glOptions="opaque")
    edges = gl.GLLinePlotItem(pos=V[_BOX_EDGES], mode="lines", color=edge_color,
                              width=1.5, antialias=True)
    return mesh, edges


def satellite_items(half, color):
    h = np.asarray(half, dtype=float)
    c, b, a = np.argsort(h)

    def vec(va, vb, vc):
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

# =====================================================================================
#  SKINS 3D: Vaca, OVNI y Caza TIE
#  Todo se modela en un marco canónico (u, v, w) y un _Rig lo estira sobre los ejes
#  principales del cuerpo, de modo que las proporciones siguen a body_half_sizes(I):
#  el eje más largo del animal/nave es el de menor momento de inercia, etc.
#  Cada skin acumula su geometría por material y la fusiona en pocos GLMeshItem.
# =====================================================================================

# translúcido con descarte de caras traseras: una sola capa de vidrio, sin rayas
_GLASS = dict(GLOptions["translucent"])
_GLASS[GL_CULL_FACE] = True


def orient_out(V, F, center):
    """Superficies abiertas: gira las caras para que las normales miren fuera de center."""
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    d = V[F].mean(1) - np.asarray(center, float)
    return (V, F[:, ::-1].copy()) if np.mean(np.einsum("ij,ij->i", n, d)) < 0 else (V, F)


def _volume(V, F):
    return np.einsum("ij,ij->", V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]])) / 6.0


def _closed(V, F):
    """Orienta las caras hacia fuera (volumen con signo positivo)."""
    return (V, F[:, ::-1].copy()) if _volume(V, F) < 0 else (V, F)


def _grid(X, wrap=True):
    nr, nc = X.shape[:2]
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc if wrap else nc - 1), indexing="ij")
    j1 = (j + 1) % nc
    a, b, c, d = i * nc + j, i * nc + j1, (i + 1) * nc + j1, (i + 1) * nc + j
    F = np.concatenate([np.stack([a, b, c], -1).reshape(-1, 3),
                        np.stack([a, c, d], -1).reshape(-1, 3)])
    return X.reshape(-1, 3), F


def _capped(X, cap0=True, cap1=True):
    """Malla a partir de una rejilla de anillos (nr, nc, 3), con tapas opcionales."""
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


def merge(*parts):
    Vs, Fs, off = [], [], 0
    for V, F in parts:
        Vs.append(V)
        Fs.append(F + off)
        off += len(V)
    return np.vstack(Vs), np.vstack(Fs)


def blob(center, radii, rows=14, cols=24, rot=None, theta=(0.0, math.pi)):
    """Elipsoide (o casquete si theta no es completo)."""
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


def rot_axis(axis, deg):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    t = math.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K


def smooth_path(pts, per=8):
    """Catmull-Rom sobre una polilínea."""
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


def _interp_radii(r, n):
    r = np.asarray(r, float)
    return np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(r)), r)


def tube(path, rw, rh=None, n=16, side=None, per=8, caps=True):
    """Barrido de una elipse a lo largo de un camino (radios escalares o listas).
    side: dirección fija del semieje 'ancho'; si es None, transporte paralelo."""
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


def lathe(profile, n=32, axis=2, center=(0, 0, 0), caps=False):
    """Superficie de revolución. profile: [(t, r), ...] con t a lo largo del eje."""
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


def slab(poly, lo, hi, axis=0, center=(0, 0)):
    """Prisma convexo: polígono (u, v) extruido entre lo y hi a lo largo de axis."""
    poly = np.asarray(poly, float) + np.asarray(center, float)
    u, v = {0: (1, 2), 1: (2, 0), 2: (0, 1)}[axis]
    k = len(poly)

    def pts(z):
        Q = np.zeros((k, 3))
        Q[:, axis], Q[:, u], Q[:, v] = z, poly[:, 0], poly[:, 1]
        return Q

    V = np.vstack([pts(lo), pts(hi), pts(lo).mean(0, keepdims=True), pts(hi).mean(0, keepdims=True)])
    j = np.arange(k)
    j1 = (j + 1) % k
    F = [np.stack([j, j1, k + j1], -1), np.stack([j, k + j1, k + j], -1),
         np.stack([np.full(k, 2 * k), j1, j], -1), np.stack([np.full(k, 2 * k + 1), k + j, k + j1], -1)]
    return _closed(V, np.vstack(F))


def ngon(n, R, phase_deg=90.0):
    a = np.radians(phase_deg + 360.0 / n * np.arange(n))
    return np.column_stack([R * np.cos(a), R * np.sin(a)])


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


class _Rig:
    """Lleva el modelo canónico a los ejes principales (a: mayor semieje, b, c: menor).

    roles[i] = rango (0=a, 1=b, 2=c) del eje canónico i; ref[i] = semitamaño canónico
    del modelo en ese eje. Las piezas estructurales se estiran del todo; los detalles
    (ojos, alienígena, cúpula…) conservan su forma con una escala local isótropa.
    """
    SIZE = 1.25

    def __init__(self, half, roles, ref, floor):
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

    def _xf(self, V, local=None, stretch=(), k=None):
        V = np.asarray(V, float)
        if local is None:
            return (V * self.s) @ self.P.T
        c = np.asarray(local, float)
        kk = self.k if k is None else (self.kmin if k == "min" else k)
        sv = np.array([self.s[i] if i in stretch else kk for i in range(3)])
        return ((c * self.s) + (V - c) * sv) @ self.P.T

    # -- acumulación por material -----------------------------------------------------
    def mesh(self, vf, color="#ffffff", alpha=1.0, shade=1.0, vc=None, smooth=True,
             emissive=False, glass=False, local=None, stretch=(), k=None):
        V, F = vf
        Vb = self._xf(V, local, stretch, k)
        if self.flip:
            F = F[:, ::-1]
        if vc is None:
            c = QtGui.QColor(color)
            vc = np.tile([c.redF() * shade, c.greenF() * shade, c.blueF() * shade, alpha], (len(Vb), 1))
        key = ("glass", emissive) if glass else (("emis",) if emissive else (("smooth",) if smooth else ("flat",)))
        self._groups.setdefault(key, []).append((Vb, F, np.asarray(vc, float)))

    def line(self, pts, color="#ffffff", alpha=1.0, width=1.5, mode="line_strip",
             local=None, stretch=(), k=None):
        P = self._xf(pts, local, stretch, k)
        if mode == "line_strip":
            P = np.repeat(P, 2, axis=0)[1:-1]
        c = QtGui.QColor(color)
        self._lines.setdefault(float(width), []).append(
            (P, np.tile([c.redF(), c.greenF(), c.blueF(), alpha], (len(P), 1))))

    def finish(self):
        """Fusiona lo acumulado: opacos -> líneas -> emisivos -> vidrio (el orden importa)."""
        items = [self._merged(key) for key in (("smooth",), ("flat",)) if key in self._groups]
        for width, segs in self._lines.items():
            items.append(gl.GLLinePlotItem(pos=np.vstack([p for p, _ in segs]), mode="lines",
                                           color=np.vstack([c for _, c in segs]), width=width,
                                           antialias=True))
        for key in (("emis",), ("glass", False), ("glass", True)):
            if key in self._groups:
                items.append(self._merged(key))
        return items

    def _merged(self, key):
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


# =====================================================================================
#  VACA HOLSTEIN
# =====================================================================================
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


def _coat(V):
    f = np.full(len(V), 9.0)
    for c, r in _SPOTS:
        d = np.linalg.norm((V - np.array(c)) / np.array(r), axis=1)
        f = np.minimum(f, d)
    wob = 0.13 * np.sin(9.0 * V[:, 0] + 4.0 * V[:, 1]) * np.sin(7.0 * V[:, 2] + 3.0 * V[:, 1] + 1.0)
    m = 1.0 - smoothstep(0.90, 1.06, f + wob)
    rgb = _COW_WHITE[None, :] * (1 - m[:, None]) + _COW_BLACK[None, :] * m[:, None]
    return np.column_stack([rgb * 2.0, np.ones(len(V))])


def cow_items(half, sim_color):
    """Vaca: el cuerpo largo sigue al eje de menor inercia; el collar lleva el color de la simulación."""
    rig = _Rig(half, roles=(0, 2, 1), ref=(1.0, 0.30, 0.78), floor=0.24)
    ALL = (0, 1, 2)

    def coat(vf, **kw):
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


# =====================================================================================
#  OVNI con alienígena, cúpula de cristal, haz tractor y vaca abducida
# =====================================================================================
_METAL_HI = np.array([0.74, 0.78, 0.85])
_METAL_LO = np.array([0.40, 0.44, 0.52])


def _hull_profile():
    pts = [(-0.205, 0.0), (-0.205, 0.22), (-0.19, 0.45), (-0.145, 0.68), (-0.075, 0.88),
           (-0.015, 0.985), (0.02, 1.0), (0.055, 0.965), (0.095, 0.86), (0.135, 0.66),
           (0.17, 0.46), (0.19, 0.38), (0.19, 0.0)]
    P = smooth_path([(z, r, 0.0) for z, r in pts], per=6)
    return P[:, :2]


def _hull_colors(V, rim_rgb):
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


def _mini_cow(rig, center, tilt=-22, spin=35):
    """Vaquita abducida flotando en el haz."""
    R = rot_axis((0, 1, 0), tilt) @ rot_axis((0, 0, 1), spin)
    ctr = np.asarray(center, float)

    def part(off, radii, color):
        c = ctr + R @ np.asarray(off)
        rig.mesh(blob(c, radii, 8, 12, rot=R), color, shade=2.0, local=center)

    def limb(a, b, r, color="#f4f1ea"):
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


def ufo_items(half, sim_color):
    rig = _Rig(half, roles=(0, 1, 2), ref=(1.0, 1.0, 0.58), floor=0.30)
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

    def al(vf, color, **kw):
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


# =====================================================================================
#  CAZA TIE IMPERIAL: cabina esférica, ventanal octogonal, pilones y alas hexagonales
# =====================================================================================
_CELL_BLUE = np.array([0.10, 0.17, 0.38])
_CELL_SHADE = (1.0, 0.78, 1.22, 0.9, 1.12, 0.84)


def _rect(p0, p1, w):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = (p1 - p0) / np.linalg.norm(p1 - p0)
    n = np.array([-d[1], d[0]]) * w / 2
    return np.array([p0 - n, p1 - n, p1 + n, p0 + n])


def _tie_panel(rig, sgn, accent):
    """Ala hexagonal en x = sgn * 0.66 (marco canónico), plano (y, z)."""
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


def fighter_items(half, sim_color):
    rig = _Rig(half, roles=(1, 2, 0), ref=(0.74, 0.87, 1.0), floor=0.42)
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

    def wpt(R, ang):
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


class FloatSlider(QtWidgets.QWidget):

    valueChanged = QtCore.Signal(float)

    def __init__(self, label, vmin, vmax, step, value, decimals=3, parent=None):
        super().__init__(parent)
        self.vmin, self.vmax, self.step = vmin, vmax, step
        self._n = int(round((vmax - vmin) / step))

        self.label = QtWidgets.QLabel(label)
        self.label.setMinimumWidth(34)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._n)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(vmin, vmax)
        self.spin.setDecimals(decimals)
        self.spin.setSingleStep(step)
        self.spin.setKeyboardTracking(False)
        self.spin.setFixedWidth(78)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addWidget(self.label)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)

        self.slider.valueChanged.connect(self._from_slider)
        self.spin.valueChanged.connect(self._from_spin)
        self.setValue(value, silent=True)

    def value(self):
        return self.spin.value()

    def setValue(self, v, silent=False):
        v = min(max(float(v), self.vmin), self.vmax)
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(int(round((v - self.vmin) / self.step)))
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        if not silent:
            self.valueChanged.emit(v)

    def _from_slider(self, i):
        v = self.vmin + i * self.step
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(int(round((v - self.vmin) / self.step)))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


def _group_box(title, widgets):
    box = QtWidgets.QGroupBox(title)
    v = QtWidgets.QVBoxLayout(box)
    v.setSpacing(6)
    for w in widgets:
        v.addWidget(w)
    return box


class PhysicsSliders(QtWidgets.QWidget):
    """Momentos principales de inercia y velocidad angular inicial Ω₀, con la restricción
    triangular. Se usa para el problema físico compartido y para la física propia de una
    simulación que se desvincula de él."""

    changed = QtCore.Signal()

    def __init__(self, I=(1.0, 2.0, 3.0), w0=(0.0, 0.0, 0.0), parent=None):
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        def heading(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setObjectName("muted")
            return lbl

        lay.addWidget(heading("Momentos principales de inercia"))
        self.I_sliders = [FloatSlider(f"I{k + 1}", 0.1, 6.0, 0.01, 1.0, 2) for k in range(3)]
        for s in self.I_sliders:
            lay.addWidget(s)
        self.tri_warning = QtWidgets.QLabel("")
        self.tri_warning.setObjectName("warning")
        self.tri_warning.setWordWrap(True)
        self.tri_warning.hide()
        lay.addWidget(self.tri_warning)
        lay.addSpacing(4)
        lay.addWidget(heading("Velocidad angular inicial Ω₀ (ejes cuerpo)"))
        self.w_sliders = [FloatSlider(f"ω{k + 1}", -3.0, 3.0, 0.005, 0.0, 3) for k in range(3)]
        for s in self.w_sliders:
            lay.addWidget(s)

        for s in self.I_sliders + self.w_sliders:
            s.label.setMinimumWidth(74)         # columnas alineadas aunque la etiqueta cambie
        self.set(I, w0)
        for k, s in enumerate(self.I_sliders):
            s.valueChanged.connect(lambda v, k=k: self._on_inertia(k))
        for s in self.w_sliders:
            s.valueChanged.connect(self.changed)

    def _on_inertia(self, k):
        I = [s.value() for s in self.I_sliders]
        j, l = (k + 1) % 3, (k + 2) % 3
        lo, hi = abs(I[j] - I[l]), I[j] + I[l]
        if not (lo <= I[k] <= hi):
            self.I_sliders[k].setValue(min(max(I[k], lo), hi), silent=True)
            self.tri_warning.setText(
                f"⚠ Restricción triangular: I{k + 1} acotado a [{lo:.2f}, {hi:.2f}] "
                f"(I{j + 1}+I{l + 1} ≥ I{k + 1} y permutaciones).")
            self.tri_warning.show()
            QtCore.QTimer.singleShot(3500, self.tri_warning.hide)
        self._update_tags()
        self.changed.emit()

    def _update_tags(self):
        """Rotula cada I como mínima / media / máxima: el teorema va del eje intermedio."""
        I = [s.value() for s in self.I_sliders]
        tags = [""] * 3
        for rank, k in enumerate(np.argsort(I, kind="stable")):
            tags[k] = ("mín", "medio", "máx")[rank]
        for k in range(3):
            if any(j != k and abs(I[j] - I[k]) < 1e-9 for j in range(3)):
                tags[k] = "="                   # empate: trompo simétrico
        for k, s in enumerate(self.I_sliders):
            s.label.setText(f"I{k + 1} · {tags[k]}")

    def get(self):
        return (tuple(s.value() for s in self.I_sliders),
                tuple(s.value() for s in self.w_sliders))

    def set(self, I, w0):
        for s, v in zip(self.I_sliders, I):
            s.setValue(v, silent=True)
        for s, v in zip(self.w_sliders, w0):
            s.setValue(v, silent=True)
        self._update_tags()


class SimConfigPanel(QtWidgets.QWidget):
    """Esquema numérico de una simulación (integrador y Δt). Por defecto usa el problema
    físico compartido; puede desvincularse para tener sus propios I y Ω₀."""

    changed = QtCore.Signal()

    def __init__(self, index, cfg: SimConfig, shared: PhysicsSliders, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = cfg.name
        self.color = cfg.color
        self.shared = shared

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(8)

        self.enabled = QtWidgets.QCheckBox(f"Simulación {cfg.name} activa")
        self.enabled.setStyleSheet(f"QCheckBox {{ color: {cfg.color}; font-weight: 600; }}")
        lay.addWidget(self.enabled)

        self.method = QtWidgets.QComboBox()
        self.method.addItems(list(SCHEMES))
        self.method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.method.setMinimumContentsLength(16)
        self.dt = FloatSlider("Δt", 0.001, 0.25, 0.001, 0.02, 3)
        lay.addWidget(_group_box("Esquema numérico", [self.method, self.dt]))

        self.linked = QtWidgets.QCheckBox("Usar el problema físico compartido")
        self.linked.setChecked(True)
        self.linked.setToolTip("Desmárcalo para dar a esta simulación sus propios I y Ω₀ "
                               "(p. ej. para comparar cuerpos distintos).")
        self.phys_summary = QtWidgets.QLabel()
        self.phys_summary.setObjectName("muted")
        self.phys_summary.setWordWrap(True)
        self.own = PhysicsSliders(cfg.I, cfg.w0)
        self.own_box = _group_box(f"Física propia de {cfg.name}", [self.own])
        lay.addWidget(_group_box("Problema físico", [self.linked, self.phys_summary, self.own_box]))
        lay.addStretch(1)

        self.set_config(cfg)
        self._sync_link_ui()

        self.enabled.toggled.connect(self.changed)
        self.method.currentIndexChanged.connect(self.changed)
        self.dt.valueChanged.connect(self.changed)
        self.own.changed.connect(self.changed)
        self.linked.toggled.connect(self._on_link_toggled)
        shared.changed.connect(self._update_summary)

    def is_linked(self):
        return self.linked.isChecked()

    def set_linked(self, flag):
        self.linked.blockSignals(True)
        self.linked.setChecked(bool(flag))
        self.linked.blockSignals(False)
        self._sync_link_ui()

    def _on_link_toggled(self, on):
        if not on:      # parte de los valores compartidos: al desvincular nada salta
            self.own.set(*self.shared.get())
        self._sync_link_ui()
        self.changed.emit()

    def _sync_link_ui(self):
        linked = self.linked.isChecked()
        self.own_box.setVisible(not linked)
        self.phys_summary.setVisible(linked)
        self._update_summary()

    def _update_summary(self):
        if not self.linked.isChecked():
            return
        I, w = self.shared.get()
        self.phys_summary.setText(
            f"I = ({I[0]:.2f}, {I[1]:.2f}, {I[2]:.2f})   "
            f"Ω₀ = ({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})")

    def get_config(self) -> SimConfig:
        I, w0 = self.shared.get() if self.linked.isChecked() else self.own.get()
        return SimConfig(
            name=self.name, color=self.color,
            enabled=self.enabled.isChecked(),
            method=self.method.currentText(),
            I=I, w0=w0, dt=self.dt.value())

    def set_config(self, cfg: SimConfig):
        widgets = [self.enabled, self.method]
        for w in widgets:
            w.blockSignals(True)
        self.enabled.setChecked(cfg.enabled)
        self.method.setCurrentText(cfg.method)
        for w in widgets:
            w.blockSignals(False)
        self.own.set(cfg.I, cfg.w0)
        self.dt.setValue(cfg.dt, silent=True)


class Card(QtWidgets.QFrame):

    def __init__(self, title, widget, subtitle="", parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(4)
        head = QtWidgets.QHBoxLayout()
        t = QtWidgets.QLabel(title)
        t.setObjectName("cardTitle")
        head.addWidget(t)
        head.addStretch(1)
        self.subtitle = QtWidgets.QLabel(subtitle)
        self.subtitle.setObjectName("muted")
        head.addWidget(self.subtitle)
        lay.addLayout(head)
        lay.addWidget(widget, 1)


def build_stylesheet(font):
    return f"""
QWidget {{ background: {BG}; color: {TEXT}; font-family: "{font}"; font-size: 13px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QGroupBox {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 8px;
             margin-top: 16px; padding: 12px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {MUTED}; font-weight: 600; letter-spacing: 0.5px; }}
QGroupBox QWidget {{ background: transparent; }}
QFrame#card {{ background: {PANEL}; border: 1px solid {BORDER}; border-radius: 10px; }}
QFrame#card QLabel {{ background: transparent; }}
QFrame#card QWidget#plain {{ background: transparent; }}
QLabel#cardTitle {{ font-size: 15px; font-weight: 700; color: #ffffff; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#title {{ font-size: 24px; font-weight: 700; color: #ffffff; letter-spacing: 1px; }}
QLabel#warning {{ color: #FFD740; }}
QLabel#notice {{ background: rgba(255, 215, 64, 30); color: #ffe9a0; border: 1px solid #FFD740; border-radius: 6px; padding: 6px 10px; }}
QLabel#bannerTitle {{ color: {MUTED}; font-weight: 600; font-size: 13px; }}
QPushButton {{ background: {BUTTON}; border: 1px solid {BORDER}; border-radius: 6px;
               padding: 6px 10px; }}
QPushButton:hover {{ background: #575c64; }}
QPushButton:pressed {{ background: #33363b; }}
QPushButton#primary {{ background: #00b37e; border-color: #3ddc97; font-weight: 700; color: white; }}
QPushButton#primary:hover {{ background: #00c98d; }}
QPushButton#accent {{ background: #7c3aed; border-color: #a78bfa; font-weight: 700; color: white; }}
QPushButton#accent:hover {{ background: #8b5cf6; }}
QComboBox, QDoubleSpinBox, QSpinBox {{ background: {FIELD}; border: 1px solid {BORDER};
               border-radius: 5px; padding: 3px 6px; }}
QComboBox QAbstractItemView {{ background: {FIELD}; selection-background-color: {ACCENT}; }}
QSlider::groove:horizontal {{ height: 4px; background: #1f2124; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #7FE7FF; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #ffffff; width: 14px; margin: -6px 0; border-radius: 7px; }}
QTabWidget::pane {{ border: 1px solid {BORDER}; border-radius: 8px; background: {BG}; }}
QTabBar::tab {{ background: {PANEL}; padding: 6px 14px; border-top-left-radius: 6px;
                border-top-right-radius: 6px; margin-right: 2px; color: {MUTED}; }}
QTabBar::tab:selected {{ background: {BUTTON}; color: white; }}
QTabWidget#views::pane {{ border: none; background: {BG}; }}
QTabWidget#views QTabBar::tab {{ padding: 9px 22px; font-size: 14px; font-weight: 600;
                                 margin-right: 4px; border: 1px solid {BORDER}; border-bottom: none; }}
QTabWidget#views QTabBar::tab:selected {{ background: {ACCENT}; color: white;
                                          border-bottom: 3px solid #FFD740; }}
QTabWidget#views QTabBar::tab:hover:!selected {{ background: #4a4e55; color: white; }}
QSplitter::handle {{ background: {BG}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {MUTED};
                         border-radius: 3px; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background: #00b37e; border-color: #3ddc97; }}
"""


DEFAULT_CONFIGS = [
    SimConfig("A", SIM_COLORS[0], True, M_EULER),
    SimConfig("B", SIM_COLORS[1], True, M_CN),
    SimConfig("C", SIM_COLORS[2], True, M_RK4),
]

_JANI = dict(I=(1.0, 2.0, 3.0), w0=(0.01, 2.0, 0.01))
PRESETS = {
    "Comparar integradores (mismo Δt)": [
        dict(enabled=True, method=M_EULER, dt=0.02, **_JANI),
        dict(enabled=True, method=M_CN, dt=0.02, **_JANI),
        dict(enabled=True, method=M_RK4, dt=0.02, **_JANI)],
    "Efecto Janibekov (eje intermedio)": [
        dict(enabled=True, method=M_RK4, dt=0.01, **_JANI),
        dict(enabled=False, method=M_CN, dt=0.01, **_JANI),
        dict(enabled=False, method=M_EULER, dt=0.01, **_JANI)],
    "Estable: eje de menor inercia": [
        dict(enabled=True, method=M_RK4, dt=0.01, I=(1.0, 2.0, 3.0), w0=(2.0, 0.08, 0.08)),
        dict(enabled=False), dict(enabled=False)],
    "Estable: eje de mayor inercia": [
        dict(enabled=True, method=M_RK4, dt=0.01, I=(1.0, 2.0, 3.0), w0=(0.08, 0.08, 2.0)),
        dict(enabled=False), dict(enabled=False)],
    "Separatriz: un lado y el otro": [
        dict(enabled=True, method=M_RK4, dt=0.01, I=(1.0, 2.0, 3.0), w0=(0.85, 1.0, 0.5)),
        dict(enabled=True, method=M_RK4, dt=0.01, I=(1.0, 2.0, 3.0), w0=(0.88, 1.0, 0.5)),
        dict(enabled=False)],
    "Comparar inercias (mismo Ω₀)": [
        dict(enabled=True, method=M_RK4, dt=0.01, I=(1.0, 2.0, 3.0), w0=(0.01, 2.0, 0.01)),
        dict(enabled=True, method=M_RK4, dt=0.01, I=(1.0, 2.7, 3.0), w0=(0.01, 2.0, 0.01)),
        dict(enabled=True, method=M_RK4, dt=0.01, I=(1.0, 1.3, 3.0), w0=(0.01, 2.0, 0.01))],
    "Euler vs Crank-Nicolson vs RK4 (Δt grande)": [
        dict(enabled=True, method=M_EULER, dt=0.08, **_JANI),
        dict(enabled=True, method=M_CN, dt=0.08, **_JANI),
        dict(enabled=True, method=M_RK4, dt=0.08, **_JANI)],
    "Euler explícito vs implícito vs C-N (Δt grande)": [
        dict(enabled=True, method=M_EULER, dt=0.08, I=(1.0, 2.0, 3.0), w0=(2.0, 0.3, 0.3)),
        dict(enabled=True, method=M_BE, dt=0.08, I=(1.0, 2.0, 3.0), w0=(2.0, 0.3, 0.3)),
        dict(enabled=True, method=M_CN, dt=0.08, I=(1.0, 2.0, 3.0), w0=(2.0, 0.3, 0.3))],
}
PRESET_NOTES = {
    "Comparar integradores (mismo Δt)":
        "Mismo problema y mismo Δt con Euler, Crank-Nicolson y RK4: compara cómo derivan T y ‖L‖.",
    "Efecto Janibekov (eje intermedio)":
        "Rotación casi pura sobre el eje intermedio con una pequeña perturbación: observa el vuelco.",
    "Estable: eje de menor inercia":
        "Rotación sobre el eje de menor inercia: las órbitas próximas lo rodean.",
    "Estable: eje de mayor inercia":
        "Rotación sobre el eje de mayor inercia: las órbitas próximas lo rodean.",
    "Separatriz: un lado y el otro":
        "Dos Ω₀ casi iguales a ambos lados de la separatriz: una órbita rodea el eje mayor y la otra el menor.",
    "Comparar inercias (mismo Ω₀)":
        "Mismo Ω₀ con tres cuerpos distintos (B y C con física propia): cambia la órbita con la I₂.",
    "Euler vs Crank-Nicolson vs RK4 (Δt grande)":
        "Los tres esquemas con Δt = 0.08 sobre el mismo problema: observa cuál se desvía antes.",
    "Euler explícito vs implícito vs C-N (Δt grande)":
        "Euler explícito, implícito y C-N con Δt = 0.08 girando sobre el eje de menor inercia: "
        "observa si la energía crece, decae o se conserva.",
}
DEFAULT_PRESET ="Comparar integradores (mismo Δt)"


class MainWindow(QtWidgets.QMainWindow):
    FRAME_MS = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Janibekov Lab — Dinámica de rotación del sólido rígido")
        self.resize(1680, 980)

        self.sims: list[Simulation] = []
        self.previews: dict[int, dict] = {}
        self.clock = 0.0
        self.running = False
        self.frame = 0
        self.family_items = []
        self.separatrix_items = []
        self.sat_items = []
        self._plane_k, self._plane_sign_val = 1, 1
        self._workers, self._preview_token = [], 0

        self._build_ui()
        self._build_phase_items()
        self._build_plot_items()

        self._param_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._param_timer.timeout.connect(self.apply_params)

        self.apply_preset(DEFAULT_PRESET)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.FRAME_MS)

        for key, slot in (("Space", self.toggle_run), ("R", self.reset),
                          ("P", self.compute_preview)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)
        for k in range(self.view_tabs.count()):
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{k + 1}"), self,
                            activated=partial(self.view_tabs.setCurrentIndex, k))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_controls())
        left_w = QtWidgets.QWidget()
        left_w.setFixedWidth(400)
        left = QtWidgets.QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(8)
        left.addWidget(self._build_header())     # fija: Iniciar/Reiniciar siempre a la vista
        left.addWidget(scroll, 1)
        root.addWidget(left_w)

        main = QtWidgets.QVBoxLayout()
        main.setSpacing(8)
        main.addLayout(self._build_banner())

        # ---- Dinámica: órbita en el espacio de fases + movimiento del satélite, a la vez
        self.phase_view = gl.GLViewWidget()
        self.phase_view.setBackgroundColor(VIEW_BG)
        self.sat_view = gl.GLViewWidget()
        self.sat_view.setBackgroundColor(VIEW_BG)

        def surface_check(text, color, tip):
            c = QtWidgets.QCheckBox(text)
            c.setChecked(True)
            c.setToolTip(tip)
            c.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            c.toggled.connect(self._update_static_visibility)
            return c

        self.chk_E = surface_check("Energía", "#40D98C", "Elipsoide de energía (2T constante)")
        self.chk_L = surface_check("Momento", "#F259BF", "Elipsoide de momento angular (‖L‖ constante)")
        self.chk_sep = surface_check("Separatriz", "#FFFFFF", "Curvas que separan los dos tipos de órbita")
        self.chk_fam = surface_check("Polodias", "#8C9EC7", "Órbitas posibles con esta energía y este momento")
        self.trail_spin = QtWidgets.QDoubleSpinBox()
        self.trail_spin.setRange(1, 5000)
        self.trail_spin.setDecimals(0)
        self.trail_spin.setSingleStep(10)
        self.trail_spin.setValue(60)
        self.trail_spin.setSuffix(" s")
        self.trail_spin.setKeyboardTracking(False)
        self.trail_spin.setToolTip("Duración visible de la estela, igual para todas las simulaciones "
                                   "(órbita 3D y plano de fases)")
        bar = QtWidgets.QHBoxLayout()
        for c in (self.chk_E, self.chk_L, self.chk_sep, self.chk_fam):
            bar.addWidget(c)
        bar.addStretch(1)
        bar.addWidget(QtWidgets.QLabel("Estela"))
        bar.addWidget(self.trail_spin)

        self.surface_notice = self._notice_label()
        phase_body = QtWidgets.QWidget()
        phase_body.setObjectName("plain")
        pv = QtWidgets.QVBoxLayout(phase_body)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.setSpacing(6)
        pv.addLayout(bar)
        pv.addWidget(self.surface_notice)
        pv.addWidget(self.phase_view, 1)
        self.phase_card = Card("Órbita en el espacio de fases (ω₁, ω₂, ω₃)", phase_body,
                               "elipsoide T · elipsoide ‖L‖ · separatriz")

        self.shape_combo = QtWidgets.QComboBox()
        self.shape_combo.addItems(["Satélite", "Paralelepípedo", "Elipsoide", "Vaca", "Caza Estelar", "OVNI"])
        self.shape_combo.currentIndexChanged.connect(self._rebuild_satellites)
        sat_bar = QtWidgets.QHBoxLayout()
        sat_bar.addWidget(QtWidgets.QLabel("Forma"))
        sat_bar.addWidget(self.shape_combo)
        sat_bar.addStretch(1)
        sat_body = QtWidgets.QWidget()
        sat_body.setObjectName("plain")
        sv = QtWidgets.QVBoxLayout(sat_body)
        sv.setContentsMargins(0, 0, 0, 0)
        sv.setSpacing(6)
        sv.addLayout(sat_bar)
        sv.addWidget(self.sat_view, 1)
        self.sat_card = Card("Orientación del cuerpo (marco inercial)", sat_body)
        self.sat_card.subtitle.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.sat_card.subtitle.setText(
            "<span style='color:%s'>e₁</span> <span style='color:%s'>e₂</span> "
            "<span style='color:%s'>e₃</span> ejes · "
            "<span style='color:#FFD933'>L</span> momento · <span style='color:#FFFFFF'>ω</span> vel. angular"
            % tuple(AXIS_COLORS))

        self.dyn_tab = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        self.dyn_tab.setChildrenCollapsible(False)
        self.dyn_tab.addWidget(self.phase_card)
        self.dyn_tab.addWidget(self.sat_card)
        self.dyn_tab.setSizes([1, 1])

        # ---- Derivas de T, L y componentes de ω
        self.plot_T = pg.PlotWidget()
        self.plot_L = pg.PlotWidget()
        self.plot_w = pg.PlotWidget()
        for p, ylab in ((self.plot_T, "ΔT"), (self.plot_L, "Δ‖L‖"), (self.plot_w, "ω")):
            self._style_plot(p, "t", ylab)
            p.getPlotItem().setDownsampling(auto=True, mode="peak")
            p.getPlotItem().setClipToView(True)
            p.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.card_T = Card("Deriva de energía  ΔT(t) = T(t) − T₀", self.plot_T)
        self.card_L = Card("Deriva del momento  Δ‖L‖(t) = ‖L(t)‖ − ‖L₀‖", self.plot_L)
        self.card_w = Card("Componentes ω(t) — simulación de referencia", self.plot_w)
        drift = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        for c in (self.card_T, self.card_L, self.card_w):
            drift.addWidget(c)
        drift.setSizes([360, 360, 300])
        self.chk_log = QtWidgets.QCheckBox("Escala logarítmica |Δ|")
        self.chk_log.toggled.connect(self._on_log_toggled)
        log_row = QtWidgets.QHBoxLayout()
        log_row.addStretch(1)
        log_row.addWidget(self.chk_log)
        self.drift_tab = QtWidgets.QWidget()
        dv = QtWidgets.QVBoxLayout(self.drift_tab)
        dv.setContentsMargins(0, 0, 0, 0)
        dv.setSpacing(6)
        dv.addLayout(log_row)
        dv.addWidget(drift, 1)

        self.view_tabs = QtWidgets.QTabWidget()
        self.view_tabs.setObjectName("views")
        self.view_tabs.setDocumentMode(True)
        self.plane_tab = self._build_plane_tab()
        self.analysis_tab = self._build_analysis_tab()
        for widget, label in ((self.dyn_tab, "🌐  Dinámica"),
                              (self.drift_tab, "📈  Derivas de T y L"),
                              (self.plane_tab, "🧭  Plano de fases"),
                              (self.analysis_tab, "📋  Análisis")):
            self.view_tabs.addTab(widget, label)
        for i in range(self.view_tabs.count()):
            self.view_tabs.setTabToolTip(i, f"Ctrl+{i + 1}")
        self.view_tabs.currentChanged.connect(self._on_tab_changed)
        corner = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 0, 4)
        cl.setSpacing(8)
        self.clock_label = QtWidgets.QLabel("t = 0.00 s")
        self.clock_label.setObjectName("muted")
        self.clock_label.setMinimumWidth(80)
        ref_lbl = QtWidgets.QLabel("Referencia")
        ref_lbl.setObjectName("muted")
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems([c.name for c in DEFAULT_CONFIGS])
        self.ref_combo.setToolTip("Simulación cuyas superficies, carta de fases y ω(t) se muestran")
        self.ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        for w_ in (self.clock_label, ref_lbl, self.ref_combo):
            cl.addWidget(w_)
        self.view_tabs.setCornerWidget(corner, QtCore.Qt.Corner.TopRightCorner)
        main.addWidget(self.view_tabs, 1)
        root.addLayout(main, 1)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def _on_tab_changed(self, _=None):
        self._refresh_views()
        self._update_status()

    @staticmethod
    def _style_plot(p, xlab, ylab):
        p.showGrid(x=True, y=True, alpha=0.18)
        p.setLabel("bottom", xlab, **axis_label_style())
        p.setLabel("left", ylab, **axis_label_style())
        tick_font = QtGui.QFont(UI_FONT, 9)
        for ax in ("left", "bottom"):
            p.getAxis(ax).setTickFont(tick_font)
            p.getAxis(ax).setTextPen(TEXT)
        p.getAxis("left").enableAutoSIPrefix(False)

    def _build_banner(self):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)
        title = QtWidgets.QLabel("ESTABILIDAD DE LA ÓRBITA")
        title.setObjectName("bannerTitle")
        row.addWidget(title)
        self.badges, self._badge_state = [], [None] * len(DEFAULT_CONFIGS)
        for _ in DEFAULT_CONFIGS:
            b = QtWidgets.QLabel()
            b.setTextFormat(QtCore.Qt.TextFormat.RichText)
            b.hide()
            row.addWidget(b)
            self.badges.append(b)
        row.addStretch(1)
        return row

    def _build_plane_tab(self):
        self.plot_plane = pg.PlotWidget()
        self._style_plot(self.plot_plane, "ω_i", "ω_j")
        self.plot_plane.setAspectLocked(True)
        self.plot_plane.addLegend(offset=(8, 8), labelTextSize="9pt")

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("Equilibrio"))
        self.plane_axis = QtWidgets.QComboBox()
        self.plane_axis.addItems(["Automático (eje de la órbita de ref.)",
                                  "e₁  (ω₁ = ±Ω)", "e₂  (ω₂ = ±Ω)", "e₃  (ω₃ = ±Ω)"])
        self.plane_sign = QtWidgets.QComboBox()
        self.plane_sign.addItems(["+Ω", "−Ω"])
        self.plane_sign.setEnabled(False)
        self.chk_plane_field = QtWidgets.QCheckBox("Campo de direcciones")
        self.chk_plane_field.setChecked(True)
        self.plane_sign.setToolTip("Solo se elige con un equilibrio fijo; en automático lo decide "
                                   "la órbita de la simulación de referencia")
        for w in (self.plane_axis, self.plane_sign):
            w.currentIndexChanged.connect(self._update_phase_plane)
        controls.addWidget(self.plane_axis)
        controls.addWidget(QtWidgets.QLabel("Signo"))
        controls.addWidget(self.plane_sign)
        self.chk_plane_field.toggled.connect(self._update_phase_plane)
        controls.addWidget(self.chk_plane_field)
        controls.addStretch(1)

        body = QtWidgets.QWidget()
        body.setObjectName("plain")
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(controls)
        self.plane_notice = self._notice_label()
        v.addWidget(self.plane_notice)
        v.addWidget(self.plot_plane, 1)
        self.plane_card = Card("Plano de fases sobre el elipsoide de energía", body)

        info = QtWidgets.QFrame()
        info.setObjectName("card")
        info.setFixedWidth(360)
        iv = QtWidgets.QVBoxLayout(info)
        iv.setContentsMargins(14, 12, 14, 12)
        iv.setSpacing(10)
        t = QtWidgets.QLabel("Estabilidad del equilibrio")
        t.setObjectName("cardTitle")
        iv.addWidget(t)
        self.plane_badge = QtWidgets.QLabel()
        self.plane_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        iv.addWidget(self.plane_badge)
        self.plane_info = QtWidgets.QLabel()
        self.plane_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.plane_info.setWordWrap(True)
        self.plane_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        info_scroll = QtWidgets.QScrollArea()
        info_scroll.setWidgetResizable(True)
        info_scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        info_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        info_scroll.viewport().setAutoFillBackground(False)
        info_scroll.setWidget(self.plane_info)
        iv.addWidget(info_scroll, 1)

        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addWidget(self.plane_card, 1)
        h.addWidget(info)
        return w

    def _build_analysis_tab(self):
        def scrolled(inner):
            inner.setObjectName("plain")
            sa = QtWidgets.QScrollArea()
            sa.setWidgetResizable(True)
            sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            sa.setStyleSheet("QScrollArea { background: transparent; }")
            sa.viewport().setAutoFillBackground(False)
            sa.setWidget(inner)
            return sa

        def rich_label():
            lbl = QtWidgets.QLabel()
            lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            lbl.setWordWrap(True)
            lbl.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
            return lbl

        # estado y conservación (en vivo)
        box = QtWidgets.QWidget()
        v = QtWidgets.QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(12)
        self.status_labels = []
        for _ in DEFAULT_CONFIGS:
            lbl = rich_label()
            v.addWidget(lbl)
            self.status_labels.append(lbl)
        v.addStretch(1)
        status_card = Card("Estado y conservación", scrolled(box), "en vivo")

        # extrapolación de Richardson
        rb = QtWidgets.QWidget()
        g = QtWidgets.QGridLayout(rb)
        g.setContentsMargins(0, 0, 0, 0)
        g.addWidget(QtWidgets.QLabel("Instante final T"), 0, 0)
        self.rich_T = QtWidgets.QDoubleSpinBox()
        self.rich_T.setRange(0.1, 200)
        self.rich_T.setValue(10)
        self.rich_T.setSuffix(" s")
        g.addWidget(self.rich_T, 0, 1)
        btn_rich = QtWidgets.QPushButton("Estimar orden y error")
        btn_rich.setToolTip("Integra con n, 2n y 4n pasos hasta T (n = T/Δt) para cada simulación activa")
        btn_rich.clicked.connect(self.run_richardson)
        g.addWidget(btn_rich, 1, 0, 1, 2)
        self.rich_label = rich_label()
        self.rich_label.setText(
            f"<span style='color:{MUTED}'>Orden estimado p ≈ log₂(‖u_n − u_2n‖ / ‖u_2n − u_4n‖) "
            f"y error ‖u_2n − u_n‖ / (2ᵖ − 1) del estado final.</span>")
        g.addWidget(self.rich_label, 2, 0, 1, 2)
        g.setRowStretch(3, 1)
        rich_card = Card("Extrapolación de Richardson", scrolled(rb))

        # equilibrios
        self.eq_label = rich_label()
        eq_card = Card("Puntos de equilibrio (autovalores del Jacobiano)", scrolled(self.eq_label))

        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(status_card, 1)
        left.addWidget(rich_card, 1)
        h.addLayout(left, 1)
        h.addWidget(eq_card, 1)
        return w

    def _build_header(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 8, 0)
        lay.setSpacing(8)
        title = QtWidgets.QLabel("Janibekov Lab")
        title.setObjectName("title")
        sub = QtWidgets.QLabel("Ecuaciones de Euler · teorema del eje intermedio")
        sub.setObjectName("muted")
        lay.addWidget(title)
        lay.addWidget(sub)

        box = QtWidgets.QGroupBox("Simulación")
        g = QtWidgets.QGridLayout(box)
        self.btn_run = QtWidgets.QPushButton("▶  Iniciar")
        self.btn_run.setObjectName("primary")
        self.btn_run.setToolTip("Iniciar / pausar  [Espacio]")
        self.btn_run.clicked.connect(self.toggle_run)
        btn_reset = QtWidgets.QPushButton("⟲  Reiniciar")
        btn_reset.setToolTip("Volver a t = 0  [R]")
        btn_reset.clicked.connect(self.reset)
        g.addWidget(self.btn_run, 0, 0)
        g.addWidget(btn_reset, 0, 1)
        self.speed = FloatSlider("Vel.", 0.2, 30.0, 0.1, 3.0, 1)
        self.speed.spin.setPrefix("×")
        self.speed.setToolTip("Segundos simulados por cada segundo real (aprox.)")
        g.addWidget(self.speed, 1, 0, 1, 2)
        lay.addWidget(box)
        return w

    def _build_controls(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 8, 4)
        lay.setSpacing(10)

        # ---- escenarios
        box = QtWidgets.QGroupBox("Escenarios predefinidos")
        v = QtWidgets.QVBoxLayout(box)
        h = QtWidgets.QHBoxLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        self.preset_combo.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.preset_combo.setMinimumContentsLength(16)
        btn_apply = QtWidgets.QPushButton("Aplicar")
        btn_apply.clicked.connect(lambda: self.apply_preset(self.preset_combo.currentText()))
        h.addWidget(self.preset_combo, 1)
        h.addWidget(btn_apply)
        v.addLayout(h)
        self.preset_note = QtWidgets.QLabel()
        self.preset_note.setObjectName("muted")
        self.preset_note.setWordWrap(True)
        v.addWidget(self.preset_note)
        self.preset_combo.currentTextChanged.connect(
            lambda t: self.preset_note.setText(PRESET_NOTES.get(t, "")))
        self.preset_note.setText(PRESET_NOTES.get(self.preset_combo.currentText(), ""))
        lay.addWidget(box)

        # ---- problema físico compartido y esquema numérico de cada simulación
        base = DEFAULT_CONFIGS[0]
        self.physics = PhysicsSliders(base.I, base.w0)
        self.physics.changed.connect(self._schedule_apply)
        lay.addWidget(_group_box("Problema físico · común a las simulaciones", [self.physics]))

        caption = QtWidgets.QLabel("Cada simulación elige su esquema; ✎ marca las que usan física propia.")
        caption.setObjectName("muted")
        caption.setWordWrap(True)
        lay.addWidget(caption)
        self.tabs = QtWidgets.QTabWidget()
        self.panels = []
        for k, cfg in enumerate(DEFAULT_CONFIGS):
            p = SimConfigPanel(k, cfg, self.physics)
            p.changed.connect(self._schedule_apply)
            self.panels.append(p)
            self.tabs.addTab(p, f"● {cfg.name}")
            self.tabs.tabBar().setTabTextColor(k, QtGui.QColor(cfg.color))
        lay.addWidget(self.tabs)

        # ---- previsualización (trayectoria completa calculada de antemano)
        box = QtWidgets.QGroupBox("Previsualización")
        g = QtWidgets.QGridLayout(box)
        btn_prev = QtWidgets.QPushButton("↻  Calcular")
        btn_prev.setObjectName("accent")
        btn_prev.setToolTip("Recalcula la trayectoria completa hasta el horizonte  [P]")
        btn_prev.clicked.connect(self.compute_preview)
        g.addWidget(btn_prev, 0, 0)
        hz = QtWidgets.QHBoxLayout()
        hz.addWidget(QtWidgets.QLabel("Horizonte"))
        self.preview_T = QtWidgets.QDoubleSpinBox()
        self.preview_T.setRange(1, 2000)
        self.preview_T.setValue(100)
        self.preview_T.setSuffix(" s")
        self.preview_T.setKeyboardTracking(False)
        self.preview_T.valueChanged.connect(self._on_horizon_changed)
        hz.addWidget(self.preview_T, 1)
        g.addLayout(hz, 0, 1)
        self.chk_show_prev = QtWidgets.QCheckBox("Mostrar")
        self.chk_show_prev.setChecked(True)
        self.chk_show_prev.setToolTip("Dibuja la trayectoria prevista (discontinua) en las vistas")
        self.chk_show_prev.toggled.connect(self._update_preview_items)
        self.chk_auto = QtWidgets.QCheckBox("Automática")
        self.chk_auto.setChecked(True)
        self.chk_auto.setToolTip("Recalcula sola al cambiar cualquier parámetro")
        g.addWidget(self.chk_show_prev, 1, 0)
        g.addWidget(self.chk_auto, 1, 1)
        self.preview_status = QtWidgets.QLabel("")
        self.preview_status.setObjectName("muted")
        self.preview_status.setWordWrap(True)
        g.addWidget(self.preview_status, 2, 0, 1, 2)
        lay.addWidget(box)
        lay.addStretch(1)
        return w

    def _build_phase_items(self):
        v = self.phase_view
        self.phase_axes = []
        self.phase_axis_labels = []
        for k in range(3):
            line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(AXIS_COLORS[k], 0.8),
                                     width=1.5, antialias=True, mode="lines")
            v.addItem(line)
            lbl = gl.GLTextItem(text=f"ω{'₁₂₃'[k]}", color=QtGui.QColor(AXIS_COLORS[k]),
                                font=QtGui.QFont(UI_FONT, 13, QtGui.QFont.Weight.Bold))
            v.addItem(lbl)
            self.phase_axes.append(line)
            self.phase_axis_labels.append(lbl)

        sphere = gl.MeshData.sphere(rows=36, cols=72)
        self.mesh_E = gl.GLMeshItem(meshdata=sphere, smooth=True, color=(0.25, 0.85, 0.55, 0.10),
                                    shader="balloon", glOptions="additive")
        self.mesh_L = gl.GLMeshItem(meshdata=sphere, smooth=True, color=(0.95, 0.35, 0.75, 0.10),
                                    shader="balloon", glOptions="additive")
        v.addItem(self.mesh_E)
        v.addItem(self.mesh_L)

        self.eq_points = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), size=14, pxMode=True)
        self.eq_points.setGLOptions("translucent")
        v.addItem(self.eq_points)

        self.prev_lines, self.trail_lines, self.heads = [], [], []
        for cfg in DEFAULT_CONFIGS:
            pl = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(cfg.color, 0.35),
                                   width=1.0, antialias=True)
            tl = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(cfg.color, 1.0),
                                   width=2.5, antialias=True)
            hd = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=rgba(cfg.color, 1.0, 1.3),
                                      size=12, pxMode=True)
            for it in (pl, tl, hd):
                it.setGLOptions("translucent")
                it.hide()
                v.addItem(it)
            self.prev_lines.append(pl)
            self.trail_lines.append(tl)
            self.heads.append(hd)

        grid = gl.GLGridItem()
        grid.setSize(20, 10)
        grid.setSpacing(1, 1)
        grid.translate(0, 0, -2.0)
        grid.setColor((170, 170, 175, 90))
        self.sat_view.addItem(grid)
        self.sat_view.setCameraPosition(distance=9, elevation=18, azimuth=-60)

    def _build_plot_items(self):
        self.curves_T, self.curves_L = [], []
        self.prev_T, self.prev_L = [], []
        for cfg in DEFAULT_CONFIGS:
            for plot, live, prev in ((self.plot_T, self.curves_T, self.prev_T),
                                     (self.plot_L, self.curves_L, self.prev_L)):
                prev.append(plot.plot(pen=self._dash_pen(cfg.color)))
                live.append(plot.plot(pen=pg.mkPen(cfg.color, width=2), name=cfg.name))
        self.curves_w, self.prev_w = [], []
        for k in range(3):
            self.prev_w.append(self.plot_w.plot(pen=self._dash_pen(AXIS_COLORS[k])))
            self.curves_w.append(self.plot_w.plot(pen=pg.mkPen(AXIS_COLORS[k], width=2),
                                                  name=f"ω{k + 1}"))

        pp = self.plot_plane
        self.plane_domain = pp.plot(pen=pg.mkPen((255, 255, 255, 120), width=1.5,
                                                 style=QtCore.Qt.PenStyle.DotLine))
        self.plane_field = pp.plot(pen=pg.mkPen((200, 200, 205, 120), width=1), connect="pairs")
        self.plane_orbit_items, self.plane_sep_items = [], []
        self.plane_ref_orbit = pp.plot(pen=pg.mkPen("#FFD740", width=2,
                                                    style=QtCore.Qt.PenStyle.DashLine),
                                       name="órbita exacta (ref.)")
        self.plane_prev, self.plane_trail, self.plane_head = [], [], []
        for cfg in DEFAULT_CONFIGS:
            self.plane_prev.append(pp.plot(pen=self._dash_pen(cfg.color), connect="finite"))
            self.plane_trail.append(pp.plot(pen=pg.mkPen(cfg.color, width=2.2), connect="finite",
                                            name=cfg.name))
            self.plane_head.append(pp.plot(pen=None, symbol="o", symbolSize=10,
                                           symbolBrush=cfg.color, symbolPen="w"))
        self.plane_eq = pp.plot(pen=None, symbol="star", symbolSize=22, symbolPen="w")
        self.plane_eq.setZValue(10)

    @staticmethod
    def _dash_pen(color, alpha=150):
        c = QtGui.QColor(color)
        c.setAlpha(alpha)
        return pg.mkPen(c, width=1, style=QtCore.Qt.PenStyle.DashLine)

    def _schedule_apply(self, *_):
        self._param_timer.start()

    def _notice_label(self):
        lbl = QtWidgets.QLabel()
        lbl.setObjectName("notice")
        lbl.setWordWrap(True)
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl.setOpenExternalLinks(False)
        lbl.linkActivated.connect(self._on_notice_link)
        lbl.hide()
        return lbl

    def _on_notice_link(self, href):
        if href.startswith("ref:"):
            self.ref_combo.setCurrentIndex(int(href[4:]))

    def _refresh_tab_titles(self):
        for k, p in enumerate(self.panels):
            own = not p.is_linked()
            self.tabs.setTabText(k, f"● {p.name}" + ("  ✎" if own else ""))
            self.tabs.setTabToolTip(
                k, f"{p.name}: física propia (desvinculada del problema compartido)" if own
                else f"{p.name}: usa el problema físico compartido")

    def _update_surface_notice(self):
        """Avisa cuando alguna simulación no yace sobre las superficies mostradas (las de la
        simulación de referencia): otra I, u otra energía/momento."""
        cfgs = self.configs()
        ref_k = self.ref_index()
        ref = cfgs[ref_k]
        rows = []
        for k, c in enumerate(cfgs):
            if k == ref_k or not c.enabled:
                continue
            m = surface_mismatch(ref, c)
            if m is None:
                continue
            why = ("otra I" if not m["same_I"] else
                   f"otra energía ({m['dT']:+.1%}) y otro momento ({m['dL']:+.1%})")
            rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> {why} "
                        f"(<a href='ref:{k}' style='color:#FFD740'>ver sus superficies</a>)")
        for label, what in ((self.surface_notice, "Las superficies de esta vista"),
                            (self.plane_notice, "La carta y las órbitas exactas")):
            if rows:
                label.setText(f"⚠ {what} son las de la simulación <b>{ref.name}</b> y no "
                              f"contienen exactamente la trayectoria de: " + " · ".join(rows) + ".")
                label.show()
            else:
                label.hide()

    def apply_preset(self, name):
        upd = PRESETS[name]
        I0, w00 = self.physics.get()
        I0, w00 = tuple(upd[0].get("I", I0)), tuple(upd[0].get("w0", w00))
        self.physics.set(I0, w00)           # el problema físico del escenario es el de A
        for panel, u in zip(self.panels, upd):
            panel.set_config(replace(panel.get_config(), **u))
            own = (tuple(u.get("I", I0)), tuple(u.get("w0", w00))) != (I0, w00)
            panel.set_linked(not own)       # solo se desvincula quien trae otra física
        self.preset_combo.setCurrentText(name)
        self.apply_params()

    def configs(self):
        return [p.get_config() for p in self.panels]

    def ref_index(self):
        return self.ref_combo.currentIndex()

    def apply_params(self):
        self._param_timer.stop()
        cfgs = self.configs()
        if not cfgs[self.ref_index()].enabled:
            first = next((k for k, c in enumerate(cfgs) if c.enabled), None)
            if first is not None:
                self.ref_combo.blockSignals(True)
                self.ref_combo.setCurrentIndex(first)
                self.ref_combo.blockSignals(False)
        # todas las simulaciones se reinician juntas: así siguen en el mismo instante
        self.sims = [Simulation(c) for c in cfgs]
        self.clock = 0.0
        self.previews = {}
        self._update_static_phase()
        self._update_equilibria()
        self._update_phase_plane()
        self._rebuild_satellites()
        self._refresh_tab_titles()
        self._update_surface_notice()
        self._badge_state = [None] * len(self.badges)
        self.plot_w.setTitle(None)
        self._update_preview_items()
        if self.chk_auto.isChecked():
            self.request_preview()
        else:
            self._set_preview_status("desactualizada · pulsa ↻ Calcular [P]")
        self.status.showMessage("Parámetros aplicados · simulación reiniciada (t = 0)", 4000)
        self._refresh_views()

    def _on_horizon_changed(self):
        if self.chk_auto.isChecked() or self.previews:
            self.request_preview()

    def reset(self):
        for s in self.sims:
            s.reset()
        self.clock = 0.0
        self._refresh_views()

    def toggle_run(self):
        self.running = not self.running
        self.btn_run.setText("⏸  Pausa" if self.running else "▶  Iniciar")

    def run_richardson(self):
        T = self.rich_T.value()
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        t0 = time.perf_counter()
        rows = []
        try:
            for c in self.configs():
                if not c.enabled:
                    continue
                r = richardson_analysis(c, T)
                if np.isfinite(r["p_est"]) and np.isfinite(r["err"]):
                    w = r["u_extr"][:3]
                    body = (f"p estimado = <b>{r['p_est']:.2f}</b> (nominal {r['p']}) · n = {r['n']}<br>"
                            f"error(T) ≈ {r['err']:.2e}<br>"
                            f"ω extrapolado = ({w[0]:+.4f}, {w[1]:+.4f}, {w[2]:+.4f})")
                else:
                    body = "<span style='color:#FF6B6B'>no estimable (divergencia)</span>"
                rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                            f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g}<br>"
                            f"<span style='font-family:{MONO}'>{body}</span>")
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.rich_label.setText("<br><br>".join(rows) or "No hay simulaciones activas.")
        self.status.showMessage(
            f"Richardson hasta T = {T:g} s calculado en "
            f"{1e3 * (time.perf_counter() - t0):.0f} ms", 6000)

    def compute_preview(self):
        self.request_preview()

    def request_preview(self):
        """Calcula la trayectoria prevista en un hilo aparte; un cálculo nuevo cancela el anterior."""
        for w in self._workers:
            w.cancel()
        self._preview_token += 1
        jobs = [(k, c) for k, c in enumerate(self.configs()) if c.enabled]
        if not jobs:
            self.previews = {}
            self._update_preview_items()
            self._set_preview_status("sin simulaciones activas")
            return
        worker = PreviewWorker(self._preview_token, jobs, self.preview_T.value())
        worker.done.connect(self._on_preview_done)
        worker.finished.connect(partial(self._worker_finished, worker))
        self._workers.append(worker)
        self._set_preview_status("⏳ calculando…")
        worker.start()

    def _worker_finished(self, worker):
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def _on_preview_done(self, token, previews, elapsed):
        if token != self._preview_token:        # resultado de una versión anterior de los parámetros
            return
        self.previews = previews
        note = f"✔ {self.preview_T.value():g} s calculados en {1e3 * elapsed:.0f} ms"
        if any(p["truncated"] for p in previews.values()):
            note += f" · limitada a {MAX_PREVIEW_STEPS:,} pasos"
        self._set_preview_status(note)
        self._update_preview_items()

    def _set_preview_status(self, text):
        self.preview_status.setText(text)

    def closeEvent(self, event):
        for w in list(self._workers):
            w.cancel()
        for w in list(self._workers):
            w.wait(3000)
        super().closeEvent(event)

    def _update_preview_items(self):
        show = self.chk_show_prev.isChecked()
        log = self.chk_log.isChecked()
        for k in range(3):
            p = self.previews.get(k)
            visible = show and p is not None and len(p["t"]) > 1
            self.prev_lines[k].setVisible(visible)
            if visible:
                self.prev_lines[k].setData(pos=p["W"])
                self.prev_T[k].setData(p["t"], self._fmt(p["dT"], log))
                self.prev_L[k].setData(p["t"], self._fmt(p["dL"], log))
            else:
                self.prev_T[k].setData([], [])
                self.prev_L[k].setData([], [])
        p = self.previews.get(self.ref_index())
        for k in range(3):
            if show and p is not None:
                self.prev_w[k].setData(p["t"], p["W"][:, k])
            else:
                self.prev_w[k].setData([], [])
        self._update_plane_preview()

    def _update_plane_preview(self):
        show = self.chk_show_prev.isChecked()
        for k in range(3):
            p = self.previews.get(k)
            if show and p is not None and len(p["t"]) > 1:
                P = self._project_plane(p["W"])
                self.plane_prev[k].setData(P[:, 0], P[:, 1], connect="finite")
            else:
                self.plane_prev[k].setData([], [])

    def _plane_equilibrium(self):
        cfg = self.configs()[self.ref_index()]
        idx = self.plane_axis.currentIndex()
        self.plane_sign.setEnabled(idx != 0)
        if idx == 0:
            k = surrounded_axis(cfg.I, cfg.w0)
            if k is None:
                k = sorted_axes(cfg.I)[1]
            return k, (1 if cfg.w0[k] >= 0 else -1)
        return idx - 1, (1 if self.plane_sign.currentIndex() == 0 else -1)

    def _project_plane(self, W):
        k, sign = self._plane_k, self._plane_sign_val
        i, j = chart_axes(k)
        P = W[:, [i, j]].astype(float)
        P[~(sign * W[:, k] > 0)] = np.nan
        return P

    def _update_phase_plane(self):
        cfg = self.configs()[self.ref_index()]
        I, w0 = np.asarray(cfg.I, dtype=float), np.asarray(cfg.w0, dtype=float)
        twoT = 2 * float(kinetic_energy(w0, I))
        k, sign = self._plane_equilibrium()
        self._plane_k, self._plane_sign_val = k, sign
        i, j = chart_axes(k)
        sub = "₁₂₃"
        pp = self.plot_plane
        pp.setLabel("bottom", f"ω{sub[i]}", **axis_label_style())
        pp.setLabel("left", f"ω{sub[j]}", **axis_label_style())

        for it in self.plane_orbit_items + self.plane_sep_items:
            pp.removeItem(it)
        self.plane_orbit_items, self.plane_sep_items = [], []
        if twoT <= 0:
            for it in (self.plane_domain, self.plane_field, self.plane_ref_orbit, self.plane_eq):
                it.setData([], [])
            self.plane_badge.setText("Ω₀ = 0: sin rotación")
            self.plane_badge.setStyleSheet(badge_css(STABILITY_COLORS["marginal"]))
            self.plane_info.setText("")
            self._update_plane_preview()
            return

        xm, ym = math.sqrt(twoT / I[i]), math.sqrt(twoT / I[j])
        s = np.linspace(0, 2 * np.pi, 300)
        self.plane_domain.setData(xm * np.cos(s), ym * np.sin(s))

        if self.chk_plane_field.isChecked():
            n = 21
            X, Y = np.meshgrid(np.linspace(-xm, xm, n), np.linspace(-ym, ym, n))
            inside = I[i] * X ** 2 + I[j] * Y ** 2 < 0.96 * twoT
            U, V = reduced_flow(I, twoT, k, sign, X[inside], Y[inside])
            seg = quiver_segments(X[inside], Y[inside], U, V,
                                  0.8 * min(2 * xm, 2 * ym) / (n - 1))
            self.plane_field.setData(seg[:, 0], seg[:, 1], connect="pairs")
        else:
            self.plane_field.setData([], [])

        orbits, seps = phase_plane_orbits(I, twoT, k)
        for P in orbits:
            it = pp.plot(P[:, 0], P[:, 1], pen=pg.mkPen((215, 215, 220, 110), width=1),
                         connect="finite")
            self.plane_orbit_items.append(it)
        for P in seps:
            it = pp.plot(P[:, 0], P[:, 1], pen=pg.mkPen((255, 255, 255, 235), width=2.5),
                         connect="finite")
            self.plane_sep_items.append(it)

        a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
        Q0 = a * w0[i] ** 2 + b * w0[j] ** 2
        self.plane_ref_orbit.setData([], [])
        if sign * w0[k] > 0 and not is_degenerate(I):
            curves = chart_level_curve(I, twoT, k, Q0, 600)
            if curves:
                P = np.vstack([np.vstack([C, [[np.nan, np.nan]]]) for C in curves])
                self.plane_ref_orbit.setData(P[:, 0], P[:, 1], connect="finite")

        e = equilibrium_analysis(I, twoT)[k]
        col = STABILITY_COLORS[e["stability"]]
        self.plane_eq.setData([0.0], [0.0], symbolBrush=col)
        pp.setRange(xRange=(-1.08 * xm, 1.08 * xm), yRange=(-1.08 * ym, 1.08 * ym), padding=0)
        self.plane_card.subtitle.setText(
            f"ref. {cfg.name} · carta ω{sub[k]} = {'+' if sign > 0 else '−'}√(…) · "
            f"2T = {twoT:.4g}")

        self.plane_badge.setText(f"{'+' if sign > 0 else '−'}Ω e{sub[k]}  ·  "
                                 f"{e['type'].upper()}  ·  {e['stability']}")
        self.plane_badge.setStyleSheet(badge_css(col, 15))
        self._update_plane_info(cfg, I, twoT, k, sign, e, a, b)
        self._update_plane_preview()

    def _update_plane_info(self, cfg, I, twoT, k, sign, e, a, b):
        sub = "₁₂₃"
        i, j = chart_axes(k)
        Om = math.sqrt(twoT / I[k])
        lam2 = (I[j] - I[k]) * (I[k] - I[i]) / (I[i] * I[j]) * Om ** 2
        lam = ", ".join(f"{z.real:+.3f}{z.imag:+.3f}i" for z in e["eigvals"])
        if e["type"] == "centro":
            shape = "elipses cerradas → las órbitas próximas rodean el equilibrio (estable)"
        elif e["type"] == "punto de silla":
            shape = ("hipérbolas → las órbitas se alejan por la variedad inestable "
                     "(efecto Janibekov); las rectas blancas son las separatrices")
        else:
            shape = "trompo simétrico: caso degenerado"
        rows = [
            f"<b>Ω</b> = {Om:.4f} &nbsp;(I{sub[k]} Ω² = 2T)",
            f"<b>Autovalores del Jacobiano</b><br>"
            f"<span style='font-family:{MONO}'>λ = {lam}</span>",
            f"<b>Linealización en la carta</b><br>"
            f"<span style='font-family:{MONO}'>λ² = (I{sub[j]}−I{sub[k]})(I{sub[k]}−I{sub[i]})"
            f"/(I{sub[i]}I{sub[j]})·Ω² = {lam2:+.4f}</span><br>"
            f"{'λ² &lt; 0 → λ imaginarios puros' if lam2 < 0 else 'λ² &gt; 0 → λ reales de signo opuesto'}",
            f"<b>Órbitas exactas</b> (curvas de nivel de Q = L² − 2T·I{sub[k]})<br>"
            f"<span style='font-family:{MONO}'>Q = {a:+.3f}·ω{sub[i]}² {b:+.3f}·ω{sub[j]}²</span><br>"
            f"{shape}",
            f"<span style='color:{MUTED}'>— trazo continuo: simulación · discontinuo: previsualización · "
            f"amarillo: órbita exacta de ref.</span>",
            "<b>Efecto de cada esquema sobre este equilibrio</b>",
        ]
        for c in self.configs():
            if not c.enabled:
                continue
            head = (f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                    f"{METHOD_SHORT[c.method]} · Δt={c.dt:g}: ")
            Ic = np.asarray(c.I, dtype=float)
            twoTc = 2 * float(kinetic_energy(c.w0, Ic))
            if twoTc <= 0 or is_degenerate(Ic):
                rows.append(head + "—")
                continue
            ec = equilibrium_analysis(Ic, twoTc)[k]
            ev = ec["eigvals"]
            lam = ev[np.argmax(np.abs(ev.imag))] if ec["type"] == "centro" else ev[np.argmax(ev.real)]
            R = abs(AMPLIFICATION[c.method](lam * c.dt))
            if ec["type"] == "centro":
                g = R - 1
                if g > 1e-10:
                    v = pill("foco inestable", STABILITY_COLORS["inestable"])
                elif g < -1e-10:
                    v = pill("foco estable", STABILITY_COLORS["estable"])
                else:
                    v = pill("centro conservado", STABILITY_COLORS["marginal"])
                rows.append(head + f"<span style='font-family:{MONO}'>|R(iβΔt)|−1 = {g:+.2e}</span> {v}")
            else:
                rows.append(head + f"<span style='font-family:{MONO}'>|R(λΔt)| = {R:.4f}</span> "
                            + pill(ec["type"], STABILITY_COLORS[ec["stability"]]))
        self.plane_info.setText("<br><br>".join(rows))

    def _refresh_phase_plane(self):
        for k, sim in enumerate(self.sims):
            if sim.cfg.enabled and sim.n > 1:
                W = sim.Y[max(0, sim.n - self._trail_points(sim)):, :3]
                P = self._project_plane(W)
                self.plane_trail[k].setData(P[:, 0], P[:, 1], connect="finite")
                last = P[-1:]
                if np.isfinite(last).all():
                    self.plane_head[k].setData(last[:, 0], last[:, 1])
                else:
                    self.plane_head[k].setData([], [])
            else:
                self.plane_trail[k].setData([], [])
                self.plane_head[k].setData([], [])

    def _on_ref_changed(self):
        self._update_static_phase()
        self._update_equilibria()
        self._update_phase_plane()
        self._update_surface_notice()
        self._update_preview_items()
        self._refresh_views()

    def _update_static_phase(self):
        cfg = self.configs()[self.ref_index()]
        I, w0 = np.array(cfg.I), np.array(cfg.w0)
        semi_E, semi_L = poinsot_semiaxes(I, w0)
        twoT = 2 * float(kinetic_energy(w0, I))

        for mesh, semi in ((self.mesh_E, semi_E), (self.mesh_L, semi_L)):
            md = gl.MeshData.sphere(rows=36, cols=72)
            md.setVertexes(md.vertexes() * semi)
            mesh.setMeshData(meshdata=md)

        for it in self.separatrix_items + self.family_items:
            self.phase_view.removeItem(it)
        self.separatrix_items, self.family_items = [], []
        for P in separatrix_curves(I, twoT):
            it = gl.GLLinePlotItem(pos=P, color=(1, 1, 1, 0.95), width=2.5, antialias=True)
            self.phase_view.addItem(it)
            self.separatrix_items.append(it)
        for P in polhode_family(I, twoT):
            it = gl.GLLinePlotItem(pos=P, color=(0.55, 0.62, 0.78, 0.35), width=1, antialias=True)
            it.setGLOptions("translucent")
            self.phase_view.addItem(it)
            self.family_items.append(it)

        lim = 1.25 * max(np.max(semi_E), np.max(semi_L), 1e-3)
        for k in range(3):
            e = np.zeros(3)
            e[k] = lim
            self.phase_axes[k].setData(pos=np.array([-e, e]))
            self.phase_axis_labels[k].setData(pos=1.08 * e)
        self.phase_view.setCameraPosition(distance=2.6 * lim)
        self.phase_card.subtitle.setText(
            f"ref. {cfg.name}:  2T = {twoT:.4g}   ‖L‖ = {np.sqrt(angular_momentum_sq(w0, I)):.4g}")
        self._update_static_visibility()

    def _update_equilibria(self):
        cfgs = self.configs()
        ref = equilibrium_diagnosis(cfgs[self.ref_index()])
        pos, cols = [], []
        rows = [f"<span style='color:{MUTED}'>ref. {cfgs[self.ref_index()].name} · "
                f"rotación pura ω = ±Ω e_k con I_k Ω² = 2T</span>"]
        for k, e in enumerate(ref["eqs"]):
            col = STABILITY_COLORS[e["stability"]]
            Om = e["omega"][k]
            lam = ", ".join(f"{z.real:+.3f}{z.imag:+.3f}i" for z in e["eigvals"])
            rows.append(f"<b>ω{'₁₂₃'[k]}</b> (Ω = ±{Om:.3f}): "
                        f"{pill(e['type'] + ' · ' + e['stability'], col)}<br>"
                        f"<span style='font-family:{MONO}'>λ = {lam}</span>")
            pos += [e["omega"], -e["omega"]]
            cols += [rgba(col, 1.0)] * 2
        self.eq_points.setData(pos=np.array(pos), color=np.array(cols))

        rows.append(f"<br><span style='color:{MUTED}'>Centro rodeado por cada órbita y "
                    f"factor de amplificación |R(iβΔt)| del esquema:</span>")
        for c in cfgs:
            if not c.enabled:
                continue
            d = equilibrium_diagnosis(c)
            head = (f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                    f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g}: ")
            if d["amp"] is None:
                rows.append(head + "órbita sobre la separatriz o trompo simétrico")
                continue
            g = d["amp"] - 1
            if g > 1e-10:
                verdict = pill("foco inestable (espiral hacia fuera)", STABILITY_COLORS["inestable"])
            elif g < -1e-10:
                verdict = pill("foco estable (disipación numérica)", STABILITY_COLORS["estable"])
            else:
                verdict = pill("conserva el centro", STABILITY_COLORS["marginal"])
            rows.append(head + f"rodea ω{'₁₂₃'[d['axis']]} · "
                        f"<span style='font-family:{MONO}'>|R|−1 = {g:+.2e}</span> → {verdict}")
        self.eq_label.setText("<br>".join(rows))

    def _update_static_visibility(self):
        self.mesh_E.setVisible(self.chk_E.isChecked())
        self.mesh_L.setVisible(self.chk_L.isChecked())
        for it in self.separatrix_items:
            it.setVisible(self.chk_sep.isChecked())
        for it in self.family_items:
            it.setVisible(self.chk_fam.isChecked())

    def _rebuild_satellites(self):
        for group in self.sat_items:
            for it in group["all"]:
                self.sat_view.removeItem(it)
        self.sat_items = []
        cfgs = [c for c in self.configs() if c.enabled]
        spacing = 3.4
        for i, cfg in enumerate(cfgs):
            k = [c.name for c in DEFAULT_CONFIGS].index(cfg.name)
            offset = np.array([(i - (len(cfgs) - 1) / 2) * spacing, 0.0, 0.0])
            half = body_half_sizes(cfg.I)
            parts, extent = self._make_body(half, cfg.color)
            axes = [gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(AXIS_COLORS[a]),
                                      width=3, antialias=True) for a in range(3)]
            Lvec = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1.0, 0.85, 0.2, 1.0),
                                     width=3, antialias=True)
            wvec = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=(1, 1, 1, 0.9),
                                     width=2, antialias=True)
            label = gl.GLTextItem(pos=offset + np.array([0, 0, 1.9]),
                                  text=f"{cfg.name} · {METHOD_SHORT[cfg.method]}",
                                  color=QtGui.QColor(cfg.color),
                                  font=QtGui.QFont(UI_FONT, 12, QtGui.QFont.Weight.Bold))
            items = parts + axes + [Lvec, wvec, label]
            for it in items:
                self.sat_view.addItem(it)
            self.sat_items.append(dict(k=k, offset=offset, extent=extent, parts=parts,
                                       axes=axes, L=Lvec, w=wvec, all=items))
        self.sat_view.setCameraPosition(distance=4 + 2.6 * len(cfgs))
        self._update_satellites()

    def _make_body(self, half, color):
        shape = self.shape_combo.currentText()
        if shape == "Elipsoide":
            md = gl.MeshData.sphere(rows=24, cols=48)
            md.setVertexes(md.vertexes() * half)
            body = gl.GLMeshItem(meshdata=md, smooth=True, color=rgba(color, 1.0),
                                 shader="shaded", glOptions="opaque")
            return [body], half
        if shape == "Paralelepípedo":
            shades = [0.55, 1.15, 0.85]
            return list(box_items(np.zeros(3), half, color, shades)), half
        if shape == "Vaca":
            return cow_items(half, color)
        if shape == "Caza Estelar":
            return fighter_items(half, color)
        if shape == "OVNI":
            return ufo_items(half, color)
            
        return satellite_items(half, color)

    def _update_satellites(self):
        for g in self.sat_items:
            sim = self.sims[g["k"]] if self.sims else None
            if sim is None:
                continue
            y = np.array(sim.y)
            R = quat_to_matrix(y[3:])
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = g["offset"]
            tr = pg.Transform3D(M)
            for part in g["parts"]:
                part.setTransform(tr)
            o = g["offset"]
            for a in range(3):
                g["axes"][a].setData(pos=np.array([o, o + R[:, a] * (g["extent"][a] + 0.35)]))
            Ls = R @ (sim.I * y[:3])
            Lref = max(np.linalg.norm(sim.Ls0), 1e-12)
            g["L"].setData(pos=np.array([o, o + 1.6 * Ls / Lref]))
            ws = R @ y[:3]
            wref = max(np.linalg.norm(sim.cfg.w0), 1e-12)
            g["w"].setData(pos=np.array([o, o + 1.25 * ws / wref]))

    def _tick(self):
        if not self.running:
            return
        self.clock += self.speed.value() * self.FRAME_MS / 1000.0
        for sim in self.sims:
            if sim.cfg.enabled:
                sim.advance_to(self.clock)
        self._refresh_views()

    def _trail_points(self, sim):
        """Puntos de la estela: la misma duración en segundos para todas, aunque difiera Δt."""
        return max(2, int(round(self.trail_spin.value() / sim.cfg.dt)))

    @staticmethod
    def _fmt(d, log):
        return np.abs(d) + 1e-18 if log else d

    def _on_log_toggled(self, on):
        for p in (self.plot_T, self.plot_L):
            p.setLogMode(False, on)
            p.setLabel("left", ("|ΔT|" if p is self.plot_T else "|Δ‖L‖|") if on
                       else ("ΔT" if p is self.plot_T else "Δ‖L‖"), **axis_label_style())
        self._update_preview_items()
        self._refresh_views()

    def _refresh_views(self):
        """Actualiza solo la pestaña visible; al cambiar de pestaña se vuelve a llamar."""
        cur = self.view_tabs.currentWidget()
        log = self.chk_log.isChecked()
        if cur is self.dyn_tab:
            for k, sim in enumerate(self.sims):
                live = sim.cfg.enabled and sim.n > 1
                self.trail_lines[k].setVisible(live)
                self.heads[k].setVisible(sim.cfg.enabled)
                if sim.cfg.enabled:
                    W = sim.Y[max(0, sim.n - self._trail_points(sim)):, :3]
                    if live:
                        self.trail_lines[k].setData(pos=W)
                    self.heads[k].setData(pos=W[-1:])
            self._update_satellites()
        elif cur is self.drift_tab:
            for k, sim in enumerate(self.sims):
                if sim.cfg.enabled and sim.n > 1:
                    t = sim.times
                    self.curves_T[k].setData(t, self._fmt(sim.dT, log))
                    self.curves_L[k].setData(t, self._fmt(sim.dL, log))
                else:
                    self.curves_T[k].setData([], [])
                    self.curves_L[k].setData([], [])
            ref = self.sims[self.ref_index()] if self.sims else None
            for k in range(3):
                if ref is not None and ref.cfg.enabled and ref.n > 1:
                    self.curves_w[k].setData(ref.times, ref.Y[:, k])
                else:
                    self.curves_w[k].setData([], [])
        elif cur is self.plane_tab:
            self._refresh_phase_plane()
        if self.frame % 4 == 0 or not self.running:
            self._update_status()
        self.frame += 1

    def _update_status(self):
        show_detail = self.view_tabs.currentWidget() is self.analysis_tab
        self.clock_label.setText(f"t = {self.clock:.2f} s")
        for k, lbl in enumerate(self.status_labels):
            if k >= len(self.sims):
                continue
            sim = self.sims[k]
            c = sim.cfg
            if not c.enabled:
                if show_detail:
                    lbl.setText(f"<span style='color:{c.color}'>●</span> "
                                f"<span style='color:{MUTED}'>{c.name} — inactiva</span>")
                self._set_badge(k, None)
                continue
            r = regime(c.I, c.w0)
            if sim.diverged:
                state, col = "DIVERGIÓ", STABILITY_COLORS["inestable"]
            else:
                state, col = r, regime_color(r)
            if show_detail:
                lbl.setText(
                    f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                    f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g} · t={sim.t:.2f} s<br>"
                    f"<span style='font-family:{MONO}'>"
                    f"ΔT={sim.dT[-1]:+.3e} &nbsp; Δ‖L‖={sim.dL[-1]:+.3e}<br>"
                    f"‖q‖−1={sim.quat_norm_error():+.3e} &nbsp; pasos={sim.n_steps}</span><br>"
                    f"{pill(state, col)}")
            self._set_badge(k, (c.name, c.color, METHOD_SHORT[c.method], state, col))
        for k in range(len(self.sims), len(self.badges)):
            self._set_badge(k, None)
        if not self.status.currentMessage().startswith(("Previsualización", "Parámetros", "Richardson")):
            self.status.showMessage(
                f"{'▶ en ejecución' if self.running else '⏸ en pausa'}   ·   reloj = {self.clock:.2f} s"
                f"   ·   [Espacio] iniciar/pausar  [R] reiniciar  [P] recalcular previsualización")

    def _set_badge(self, k, state):
        if state == self._badge_state[k]:
            return
        self._badge_state[k] = state
        b = self.badges[k]
        if state is None:
            b.hide()
            return
        name, color, method, text, col = state
        b.setText(f"<span style='color:{color}'>●</span>&nbsp; {name} · {method} &nbsp;—&nbsp; "
                  f"{text.upper()}")
        b.setStyleSheet(badge_css(col, 14))
        b.show()


def main():
    global UI_FONT, MONO
    pg.setConfigOptions(antialias=True, background=VIEW_BG, foreground=TEXT)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    UI_FONT = pick_font(UI_FONT_CANDIDATES)
    MONO = f'"{pick_font(MONO_FONT_CANDIDATES)}", monospace'
    app.setFont(QtGui.QFont(UI_FONT, 10))
    app.setStyleSheet(build_stylesheet(UI_FONT))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
