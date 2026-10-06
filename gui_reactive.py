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


def run_preview(cfg: SimConfig, T_total):
    sim = Simulation(cfg)
    n = min(int(math.ceil(T_total / cfg.dt)), MAX_PREVIEW_STEPS)
    sim.MAX_STEPS_PER_CALL = n
    sim.advance(n)
    return {"t": sim.times.copy(), "W": sim.Y[:, :3].copy(),
            "dT": sim.dT.copy(), "dL": sim.dL.copy(),
            "diverged": sim.diverged, "truncated": n == MAX_PREVIEW_STEPS}


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


class SimConfigPanel(QtWidgets.QWidget):

    changed = QtCore.Signal()
    copyRequested = QtCore.Signal(int)

    def __init__(self, index, cfg: SimConfig, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = cfg.name
        self.color = cfg.color

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.enabled = QtWidgets.QCheckBox(f"Simulación {cfg.name} activa")
        self.enabled.setStyleSheet(f"QCheckBox {{ color: {cfg.color}; font-weight: 600; }}")
        top.addWidget(self.enabled)
        top.addStretch(1)
        if index > 0:
            btn = QtWidgets.QPushButton("Copiar de A")
            btn.setToolTip("Copia inercias, Ω0 y Δt de la simulación A (mantiene el integrador)")
            btn.clicked.connect(lambda: self.copyRequested.emit(self.index))
            top.addWidget(btn)
        lay.addLayout(top)

        self.method = QtWidgets.QComboBox()
        self.method.addItems(list(SCHEMES))
        lay.addWidget(self._group("Integrador numérico", [self.method]))

        self.I_sliders = [FloatSlider(f"I{k + 1}", 0.1, 6.0, 0.01, 1.0, 2) for k in range(3)]
        self.tri_warning = QtWidgets.QLabel("")
        self.tri_warning.setObjectName("warning")
        self.tri_warning.setWordWrap(True)
        self.tri_warning.hide()
        lay.addWidget(self._group("Momentos principales de inercia",
                                  self.I_sliders + [self.tri_warning]))

        self.w_sliders = [FloatSlider(f"ω{k + 1}", -3.0, 3.0, 0.005, 0.0, 3) for k in range(3)]
        lay.addWidget(self._group("Velocidad angular inicial Ω₀ (ejes cuerpo)", self.w_sliders))

        self.dt = FloatSlider("Δt", 0.001, 0.25, 0.001, 0.02, 3)
        lay.addWidget(self._group("Paso temporal", [self.dt]))
        lay.addStretch(1)

        self.set_config(cfg)

        self.enabled.toggled.connect(self.changed)
        self.method.currentIndexChanged.connect(self.changed)
        for k, s in enumerate(self.I_sliders):
            s.valueChanged.connect(lambda v, k=k: self._on_inertia(k))
        for s in self.w_sliders + [self.dt]:
            s.valueChanged.connect(self.changed)

    @staticmethod
    def _group(title, widgets):
        box = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(box)
        v.setSpacing(6)
        for w in widgets:
            v.addWidget(w)
        return box

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
        self.changed.emit()

    def get_config(self) -> SimConfig:
        return SimConfig(
            name=self.name, color=self.color,
            enabled=self.enabled.isChecked(),
            method=self.method.currentText(),
            I=tuple(s.value() for s in self.I_sliders),
            w0=tuple(s.value() for s in self.w_sliders),
            dt=self.dt.value())

    def set_config(self, cfg: SimConfig):
        widgets = [self.enabled, self.method]
        for w in widgets:
            w.blockSignals(True)
        self.enabled.setChecked(cfg.enabled)
        self.method.setCurrentText(cfg.method)
        for w in widgets:
            w.blockSignals(False)
        for s, v in zip(self.I_sliders, cfg.I):
            s.setValue(v, silent=True)
        for s, v in zip(self.w_sliders, cfg.w0):
            s.setValue(v, silent=True)
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
        scroll.setFixedWidth(400)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_controls())
        root.addWidget(scroll)

        main = QtWidgets.QVBoxLayout()
        main.setSpacing(8)
        main.addLayout(self._build_banner())

        self.phase_view = gl.GLViewWidget()
        self.phase_view.setBackgroundColor(VIEW_BG)
        self.sat_view = gl.GLViewWidget()
        self.sat_view.setBackgroundColor(VIEW_BG)
        self.phase_card = Card("Espacio de fases (ω₁, ω₂, ω₃) — Poinsot", self.phase_view,
                               "elipsoide T · elipsoide ‖L‖ · separatriz")
        self.sat_card = Card("Orientación del satélite (marco inercial)", self.sat_view,
                             "ejes cuerpo e₁ e₂ e₃ · L (amarillo) · ω (blanco)")

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

        self.view_tabs = QtWidgets.QTabWidget()
        self.view_tabs.setObjectName("views")
        self.view_tabs.setDocumentMode(True)
        self.plane_tab = self._build_plane_tab()
        for widget, label in ((self.phase_card, "🌐  Órbitas 3D"),
                              (drift, "📈  Derivas de T y L"),
                              (self.sat_card, "🛰  Movimiento del satélite"),
                              (self.plane_tab, "🧭  Plano de fases")):
            self.view_tabs.addTab(widget, label)
        self.view_tabs.currentChanged.connect(lambda _: self._refresh_views())
        main.addWidget(self.view_tabs, 1)
        root.addLayout(main, 1)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

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
        for w in (self.plane_axis, self.plane_sign):
            w.currentIndexChanged.connect(self._update_phase_plane)
            controls.addWidget(w)
        self.chk_plane_field.toggled.connect(self._update_phase_plane)
        controls.addWidget(self.chk_plane_field)
        controls.addStretch(1)

        body = QtWidgets.QWidget()
        body.setObjectName("plain")
        v = QtWidgets.QVBoxLayout(body)
        v.setContentsMargins(0, 0, 0, 0)
        v.addLayout(controls)
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
        iv.addWidget(self.plane_info, 1)

        w = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(10)
        h.addWidget(self.plane_card, 1)
        h.addWidget(info)
        return w

    def _build_controls(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 8, 4)
        lay.setSpacing(10)

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
        self.btn_run.clicked.connect(self.toggle_run)
        btn_reset = QtWidgets.QPushButton("⟲  Reiniciar")
        btn_reset.clicked.connect(self.reset)
        btn_prev = QtWidgets.QPushButton("⚡  Previsualización rápida")
        btn_prev.setObjectName("accent")
        btn_prev.setToolTip("Calcula la trayectoria completa (estática) antes de animar")
        btn_prev.clicked.connect(self.compute_preview)
        g.addWidget(self.btn_run, 0, 0)
        g.addWidget(btn_reset, 0, 1)
        g.addWidget(btn_prev, 1, 0, 1, 2)

        self.speed = FloatSlider("Vel.", 0.005, 1.0, 0.005, 0.1, 3)
        self.speed.setToolTip("Tiempo simulado que avanza cada fotograma (s)")
        g.addWidget(self.speed, 2, 0, 1, 2)

        g.addWidget(QtWidgets.QLabel("Horizonte de previsualización T"), 3, 0)
        self.preview_T = QtWidgets.QDoubleSpinBox()
        self.preview_T.setRange(1, 2000)
        self.preview_T.setValue(100)
        self.preview_T.setSuffix(" s")
        self.preview_T.setKeyboardTracking(False)
        self.preview_T.valueChanged.connect(self._on_horizon_changed)
        g.addWidget(self.preview_T, 3, 1)

        self.chk_reactive = QtWidgets.QCheckBox("Previsualización reactiva (recalcula al mover sliders)")
        self.chk_reactive.setChecked(False)
        self.chk_show_prev = QtWidgets.QCheckBox("Mostrar trayectoria previsualizada")
        self.chk_show_prev.setChecked(True)
        self.chk_show_prev.toggled.connect(self._update_preview_items)
        g.addWidget(self.chk_reactive, 4, 0, 1, 2)
        g.addWidget(self.chk_show_prev, 5, 0, 1, 2)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Escenarios predefinidos")
        h = QtWidgets.QHBoxLayout(box)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        btn_apply = QtWidgets.QPushButton("Aplicar")
        btn_apply.clicked.connect(lambda: self.apply_preset(self.preset_combo.currentText()))
        h.addWidget(self.preset_combo, 1)
        h.addWidget(btn_apply)
        lay.addWidget(box)

        self.tabs = QtWidgets.QTabWidget()
        self.panels = []
        for k, cfg in enumerate(DEFAULT_CONFIGS):
            p = SimConfigPanel(k, cfg)
            p.changed.connect(self._schedule_apply)
            p.copyRequested.connect(self._copy_from_A)
            self.panels.append(p)
            self.tabs.addTab(p, f"● {cfg.name}")
            self.tabs.tabBar().setTabTextColor(k, QtGui.QColor(cfg.color))
        lay.addWidget(self.tabs)

        box = QtWidgets.QGroupBox("Visualización")
        g = QtWidgets.QGridLayout(box)
        g.addWidget(QtWidgets.QLabel("Superficies de"), 0, 0)
        self.ref_combo = QtWidgets.QComboBox()
        self.ref_combo.addItems([f"Simulación {c.name}" for c in DEFAULT_CONFIGS])
        self.ref_combo.currentIndexChanged.connect(self._on_ref_changed)
        g.addWidget(self.ref_combo, 0, 1)
        self.chk_E = QtWidgets.QCheckBox("Elipsoide de energía")
        self.chk_L = QtWidgets.QCheckBox("Elipsoide de momento")
        self.chk_sep = QtWidgets.QCheckBox("Separatriz")
        self.chk_fam = QtWidgets.QCheckBox("Familia de polodias")
        for i, c in enumerate((self.chk_E, self.chk_L, self.chk_sep, self.chk_fam)):
            c.setChecked(True)
            c.toggled.connect(self._update_static_visibility)
            g.addWidget(c, 1 + i // 2, i % 2)
        g.addWidget(QtWidgets.QLabel("Forma del satélite"), 3, 0)
        self.shape_combo = QtWidgets.QComboBox()
        self.shape_combo.addItems(["Satélite", "Paralelepípedo", "Elipsoide"])
        self.shape_combo.currentIndexChanged.connect(self._rebuild_satellites)
        g.addWidget(self.shape_combo, 3, 1)
        g.addWidget(QtWidgets.QLabel("Estela (puntos)"), 4, 0)
        self.trail_spin = QtWidgets.QSpinBox()
        self.trail_spin.setRange(50, 200000)
        self.trail_spin.setValue(4000)
        self.trail_spin.setSingleStep(500)
        g.addWidget(self.trail_spin, 4, 1)
        self.chk_log = QtWidgets.QCheckBox("Escala logarítmica |Δ| en derivas")
        self.chk_log.toggled.connect(self._on_log_toggled)
        g.addWidget(self.chk_log, 5, 0, 1, 2)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Extrapolación de Richardson")
        g = QtWidgets.QGridLayout(box)
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
        self.rich_label = QtWidgets.QLabel(
            f"<span style='color:{MUTED}'>Orden estimado p ≈ log₂(‖u_n − u_2n‖ / ‖u_2n − u_4n‖) "
            f"y error ‖u_2n − u_n‖ / (2ᵖ − 1) del estado final.</span>")
        self.rich_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.rich_label.setWordWrap(True)
        g.addWidget(self.rich_label, 2, 0, 1, 2)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Puntos de equilibrio (autovalores del Jacobiano)")
        v = QtWidgets.QVBoxLayout(box)
        self.eq_label = QtWidgets.QLabel()
        self.eq_label.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.eq_label.setWordWrap(True)
        v.addWidget(self.eq_label)
        lay.addWidget(box)

        box = QtWidgets.QGroupBox("Estado y conservación")
        v = QtWidgets.QVBoxLayout(box)
        self.status_labels = []
        for _ in DEFAULT_CONFIGS:
            lbl = QtWidgets.QLabel()
            lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
            lbl.setWordWrap(True)
            v.addWidget(lbl)
            self.status_labels.append(lbl)
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

    def _copy_from_A(self, idx):
        src = self.panels[0].get_config()
        dst = self.panels[idx].get_config()
        self.panels[idx].set_config(replace(dst, I=src.I, w0=src.w0, dt=src.dt))
        self._schedule_apply()

    def apply_preset(self, name):
        for panel, upd in zip(self.panels, PRESETS[name]):
            panel.set_config(replace(panel.get_config(), **upd))
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
        self.sims = [Simulation(c) for c in cfgs]
        self.clock = 0.0
        self.previews = {}
        self._update_static_phase()
        self._update_equilibria()
        self._update_phase_plane()
        self._rebuild_satellites()
        self._badge_state = [None] * len(self.badges)
        self.plot_w.setTitle(None)
        if self.chk_reactive.isChecked():
            self.compute_preview()
        else:
            self._update_preview_items()
            self.status.showMessage(
                "Parámetros aplicados · pulsa ⚡ Previsualización rápida [P] para ver la trayectoria completa",
                6000)
        self._refresh_views()

    def _on_horizon_changed(self):
        if self.chk_reactive.isChecked() or self.previews:
            self.compute_preview()

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
        t0 = time.perf_counter()
        T = self.preview_T.value()
        self.previews = {k: run_preview(c, T) for k, c in enumerate(self.configs()) if c.enabled}
        msg = f"Previsualización de {T:g} s calculada en {1e3 * (time.perf_counter() - t0):.0f} ms"
        if any(p["truncated"] for p in self.previews.values()):
            msg += f"  (limitada a {MAX_PREVIEW_STEPS:,} pasos)"
        self.status.showMessage(msg, 6000)
        self._update_preview_items()

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
                W = sim.Y[max(0, sim.n - self.trail_spin.value()):, :3]
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
        self.clock += self.speed.value()
        for sim in self.sims:
            if sim.cfg.enabled:
                sim.advance_to(self.clock)
        self._refresh_views()

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
        log = self.chk_log.isChecked()
        trail = self.trail_spin.value()
        for k, sim in enumerate(self.sims):
            live = sim.cfg.enabled and sim.n > 1
            self.trail_lines[k].setVisible(live)
            self.heads[k].setVisible(sim.cfg.enabled)
            if sim.cfg.enabled:
                W = sim.Y[max(0, sim.n - trail):, :3]
                if live:
                    self.trail_lines[k].setData(pos=W)
                self.heads[k].setData(pos=W[-1:])
            if live:
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

        self._update_satellites()
        if self.view_tabs.currentWidget() is self.plane_tab:
            self._refresh_phase_plane()
        if self.frame % 4 == 0 or not self.running:
            self._update_status()
        self.frame += 1

    def _update_status(self):
        for k, lbl in enumerate(self.status_labels):
            if k >= len(self.sims):
                continue
            sim = self.sims[k]
            c = sim.cfg
            if not c.enabled:
                lbl.setText(f"<span style='color:{c.color}'>●</span> "
                            f"<span style='color:{MUTED}'>{c.name} — inactiva</span>")
                self._set_badge(k, None)
                continue
            r = regime(c.I, c.w0)
            if sim.diverged:
                state, col = "DIVERGIÓ", STABILITY_COLORS["inestable"]
            else:
                state, col = r, regime_color(r)
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
                f"   ·   [Espacio] iniciar/pausar  [R] reiniciar  [P] previsualizar")


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
