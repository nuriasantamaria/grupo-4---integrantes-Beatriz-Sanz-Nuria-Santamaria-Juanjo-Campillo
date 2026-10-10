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
                           equilibrium_analysis, EulerExact, euler_matches_exact)


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
    # error real de ω en T frente a la solución exacta (Richardson lo estima sin conocerla)
    with np.errstate(all="ignore"):
        Y = integrate(cfg.method, u0, T / n, np.asarray(cfg.I, dtype=float), n)
        w_ex = EulerExact(cfg.I, cfg.w0).omega(np.array([float(T)]))[0]
        err_real = float(np.linalg.norm(Y[-1, :3] - w_ex))
    return {"n": n, "p": p, "p_est": float(p_est), "err": float(err), "u_extr": u_extr,
            "err_real": err_real}


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


def linear_step_errors(cfg):
    """Error por paso del esquema frente a la solución exacta e^{z} en el centro que rodea la
    órbita (pequeñas oscilaciones): módulo |R(iβΔt)| − 1 y fase arg R − βΔt. None si no hay centro."""
    I, w0 = np.asarray(cfg.I, dtype=float), np.asarray(cfg.w0, dtype=float)
    axis = surrounded_axis(I, w0)
    if axis is None:
        return None
    twoT = 2 * float(kinetic_energy(w0, I))
    e = equilibrium_analysis(I, twoT)[axis]
    if e["type"] != "centro":
        return None
    beta = float(np.max(np.abs(e["eigvals"].imag)))
    if beta <= 0:
        return None
    R = complex(AMPLIFICATION[cfg.method](1j * beta * cfg.dt))
    return {"axis": axis, "beta": beta, "R": R, "dmod": abs(R) - 1.0,
            "dphase": float(np.angle(R) - beta * cfg.dt), "period": 2 * math.pi / beta}


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
        self.exact = EulerExact(self.I, w0)
        self.view_n = None                       # None = en vivo; k = se muestra el historial hasta la fila k-1
        self._Y = np.empty((self.CHUNK, 7))
        self._dT = np.empty(self.CHUNK)
        self._dL = np.empty(self.CHUNK)
        self._er = np.zeros(self.CHUNK)          # error de módulo
        self._ep = np.zeros(self.CHUNK)          # error de fase (° de ciclo)
        self._el = np.zeros(self.CHUNK)          # error de fase (s)
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

    # ---- vista del historial (rebobinar): todo lo que se dibuja sale de aquí
    @property
    def nv(self):
        return self.n if self.view_n is None else max(1, min(self.n, self.view_n))

    @property
    def vY(self):
        return self._Y[:self.nv]

    @property
    def vdT(self):
        return self._dT[:self.nv]

    @property
    def vdL(self):
        return self._dL[:self.nv]

    @property
    def v_er(self):
        return self._er[:self.nv]

    @property
    def v_ep(self):
        return self._ep[:self.nv]

    @property
    def v_el(self):
        return self._el[:self.nv]

    @property
    def vtimes(self):
        return np.arange(self.nv) * self.cfg.dt

    @property
    def vt(self):
        return (self.nv - 1) * self.cfg.dt

    @property
    def v_steps(self):
        return self.nv - 1

    @property
    def vy(self):
        return tuple(self._Y[self.nv - 1])

    def v_quat_norm_error(self):
        return float(np.linalg.norm(self._Y[self.nv - 1, 3:]) - 1.0)

    @property
    def err_r(self):
        return self._er[:self.n]

    @property
    def err_psi(self):
        return self._ep[:self.n]

    @property
    def err_lag(self):
        return self._el[:self.n]

    def _grow(self, extra):
        need = self.n + extra
        if need <= len(self._Y):
            return
        cap = max(need, 2 * len(self._Y))
        for name in ("_Y", "_dT", "_dL", "_er", "_ep", "_el"):
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
        tt = np.arange(self.n, self.n + len(Y)) * self.cfg.dt
        with np.errstate(all="ignore"):
            self._er[sl], self._ep[sl], self._el[sl] = self.exact.error_parts(tt, Y[:, :3])
        self.n += len(Y)
        self.y = tuple(Y[-1])


MAX_PREVIEW_STEPS = 300_000
MAX_LIVE_STEPS = 1_000_000


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
            "r": sim.err_r.copy(), "psi": sim.err_psi.copy(), "lag": sim.err_lag.copy(),
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
            f" border: 2px solid {color}; border-radius: 12px; padding: 4px 12px;"
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

    def __init__(self, label, vmin, vmax, step, value, decimals=3, parent=None, spin_range=None):
        super().__init__(parent)
        self.vmin, self.vmax, self.step = vmin, vmax, step
        lo, hi = spin_range if spin_range else (vmin, vmax)
        self._lo, self._hi = lo, hi
        self._n = int(round((vmax - vmin) / step))

        self.label = QtWidgets.QLabel(label)
        self.label.setMinimumWidth(34)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._n)
        self.spin = QtWidgets.QDoubleSpinBox()
        self.spin.setRange(lo, hi)
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
        v = min(max(float(v), self._lo), self._hi)
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(self._pos(v))
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

    def _pos(self, v):
        return min(max(int(round((v - self.vmin) / self.step)), 0), self._n)

    def _from_spin(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._pos(v))
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
    """Momentos principales de inercia y velocidad angular inicial Ω₀. Se usa para el problema
    físico compartido y para la física propia de una simulación que se desvincula de él.

    Ω₀ se puede dar de dos maneras equivalentes: «eje + perturbaciones» (rotación Ω sobre un eje
    principal más una o dos componentes pequeñas sobre los otros) o por componentes (ω₁, ω₂, ω₃).
    Al cambiar de modo el vector no cambia; get() devuelve siempre las componentes."""

    changed = QtCore.Signal()
    AXIS_NAMES = ("e₁", "e₂", "e₃")

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

        head = QtWidgets.QHBoxLayout()
        head.addWidget(heading("Velocidad angular Ω₀"), 1)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Eje + perturb.", "Componentes"])
        self.mode.setToolTip("Dos formas de dar el mismo Ω₀. Al cambiar de una a otra el vector no cambia.")
        head.addWidget(self.mode)
        lay.addLayout(head)

        # --- modo componentes
        self.w_sliders = [FloatSlider(f"ω{k + 1}", -3.0, 3.0, 0.005, 0.0, 3) for k in range(3)]
        self.comp_page = QtWidgets.QWidget()
        cl = QtWidgets.QVBoxLayout(self.comp_page)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(6)
        for s in self.w_sliders:
            cl.addWidget(s)

        # --- modo eje + perturbaciones
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.setToolTip("Eje principal de la rotación inicial")
        self.w_main = FloatSlider("Ω", -3.0, 3.0, 0.005, 0.0, 3)
        self.p1_axis = QtWidgets.QComboBox()
        self.p1_axis.setToolTip("Eje sobre el que actúa la primera perturbación")
        self.p1 = FloatSlider("ε₁", -0.5, 0.5, 0.001, 0.0, 3, spin_range=(-3.0, 3.0))
        self.chk_p2 = QtWidgets.QCheckBox("Segunda perturbación")
        self.chk_p2.setToolTip("Añade una componente sobre el tercer eje (la primera actúa sobre otro distinto)")
        self.p2 = FloatSlider("ε₂", -0.5, 0.5, 0.001, 0.0, 3, spin_range=(-3.0, 3.0))
        self.readout = QtWidgets.QLabel()
        self.readout.setObjectName("muted")
        self.axis_page = QtWidgets.QWidget()
        al = QtWidgets.QVBoxLayout(self.axis_page)
        al.setContentsMargins(0, 0, 0, 0)
        al.setSpacing(6)
        r1 = QtWidgets.QHBoxLayout()
        r1.addWidget(QtWidgets.QLabel("Eje principal"))
        r1.addWidget(self.axis_combo, 1)
        al.addLayout(r1)
        al.addWidget(self.w_main)
        r2 = QtWidgets.QHBoxLayout()
        r2.addWidget(QtWidgets.QLabel("Perturbar"))
        r2.addWidget(self.p1_axis, 1)
        al.addLayout(r2)
        al.addWidget(self.p1)
        al.addWidget(self.chk_p2)
        al.addWidget(self.p2)
        al.addWidget(self.readout)
        lay.addWidget(self.axis_page)
        lay.addWidget(self.comp_page)

        for c in (self.mode, self.axis_combo, self.p1_axis):
            c.setSizeAdjustPolicy(QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            c.setMinimumContentsLength(8)
        for s in self.I_sliders + self.w_sliders + [self.w_main, self.p1, self.p2]:
            s.label.setMinimumWidth(74)         # columnas alineadas aunque la etiqueta cambie

        self._axes = (1, 0, 2)                  # (principal, perturbación 1, perturbación 2)
        self._busy = False
        self._tags = ["", "", ""]
        self.set(I, w0)

        for k, s in enumerate(self.I_sliders):
            s.valueChanged.connect(lambda v, k=k: self._on_inertia(k))
        for s in self.w_sliders:
            s.valueChanged.connect(self._on_components)
        for s in (self.w_main, self.p1, self.p2):
            s.valueChanged.connect(self._on_axis_values)
        self.chk_p2.toggled.connect(self._on_axis_values)
        self.axis_combo.currentIndexChanged.connect(self._on_main_axis)
        self.p1_axis.currentIndexChanged.connect(self._on_p1_axis)
        self.mode.currentIndexChanged.connect(self._on_mode)

    # ---- vector Ω₀ <-> (eje, perturbaciones)
    def _vector_from_axis(self):
        a, b, c = self._axes
        w = [0.0] * 3
        w[a] = self.w_main.value()
        w[b] = self.p1.value()
        w[c] = self.p2.value() if self.chk_p2.isChecked() else 0.0
        return w

    def _axis_from_vector(self, w, keep=None):
        """Eje principal = el de mayor |ω|; la primera perturbación, el mayor de los otros dos."""
        if max(abs(x) for x in w) < 1e-12 and keep is not None:
            a = keep[0]
        else:
            a = int(np.argmax([abs(x) for x in w]))
        others = [k for k in range(3) if k != a]
        b, c = sorted(others, key=lambda k: (-abs(w[k]), k))
        return (a, b, c)

    def _fill_axis_widgets(self, w):
        """Rellena el modo eje con el vector w (sin emitir señales)."""
        self._busy = True
        self._axes = self._axis_from_vector(w, self._axes)
        a, b, c = self._axes
        self.w_main.setValue(w[a], silent=True)
        self.p1.setValue(w[b], silent=True)
        self.p2.setValue(w[c], silent=True)
        self.chk_p2.setChecked(abs(w[c]) > 0)
        self._rebuild_axis_combos()
        self._busy = False
        self._update_readout()

    def _axis_text(self, k):
        return f"{self.AXIS_NAMES[k]} · {self._tags[k]}" if self._tags[k] else self.AXIS_NAMES[k]

    def _rebuild_axis_combos(self):
        a, b, c = self._axes
        was = self._busy
        self._busy = True
        self.axis_combo.clear()
        for k in range(3):
            self.axis_combo.addItem(self._axis_text(k), k)
        self.axis_combo.setCurrentIndex(a)
        self.p1_axis.clear()
        for k in (j for j in range(3) if j != a):
            self.p1_axis.addItem(self._axis_text(k), k)
        self.p1_axis.setCurrentIndex(self.p1_axis.findData(b))
        self.p1.label.setText(f"ε₁ ({self.AXIS_NAMES[b]})")
        self.p2.label.setText(f"ε₂ ({self.AXIS_NAMES[c]})")
        self.w_main.label.setText(f"Ω ({self.AXIS_NAMES[a]})")
        self.p2.setEnabled(self.chk_p2.isChecked())
        self._busy = was

    def _update_readout(self):
        w = self._vector_from_axis()
        self.readout.setText(f"Ω₀ = ({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})")

    def _sync_components(self, w):
        for s, v in zip(self.w_sliders, w):
            s.setValue(v, silent=True)

    # ---- reacciones a la interfaz
    def _on_inertia(self, k):
        self._update_tags()
        self.changed.emit()

    def _on_components(self, *_):
        if not self._busy:
            self.changed.emit()

    def _on_axis_values(self, *_):
        if self._busy:
            return
        self.p2.setEnabled(self.chk_p2.isChecked())
        w = self._vector_from_axis()
        self._sync_components(w)
        self._update_readout()
        self.changed.emit()

    def _on_main_axis(self, _=None):
        if self._busy:
            return
        a_new = self.axis_combo.currentData()
        a, b, c = self._axes
        if a_new == a:
            return
        # el eje nuevo deja de ser perturbación: el antiguo principal ocupa su sitio
        rest = [k for k in range(3) if k != a_new]
        b_new = b if b in rest else a
        c_new = next(k for k in rest if k != b_new)
        self._axes = (a_new, b_new, c_new)
        self._rebuild_axis_combos()
        self._on_axis_values()

    def _on_p1_axis(self, _=None):
        if self._busy:
            return
        b_new = self.p1_axis.currentData()
        a = self._axes[0]
        if b_new is None or b_new == self._axes[1]:
            return
        self._axes = (a, b_new, next(k for k in range(3) if k not in (a, b_new)))
        self._rebuild_axis_combos()
        self._on_axis_values()

    def _on_mode(self, i):
        axis_mode = (i == 0)
        if axis_mode:
            self._fill_axis_widgets([s.value() for s in self.w_sliders])
        self.axis_page.setVisible(axis_mode)
        self.comp_page.setVisible(not axis_mode)

    def _update_tags(self):
        """Rotula cada I como mínima / media / máxima: el teorema va del eje intermedio."""
        I = [s.value() for s in self.I_sliders]
        tags = [""] * 3
        for rank, k in enumerate(np.argsort(I, kind="stable")):
            tags[k] = ("mín", "medio", "máx")[rank]
        for k in range(3):
            if any(j != k and abs(I[j] - I[k]) < 1e-9 for j in range(3)):
                tags[k] = "="                   # empate: trompo simétrico
        self._tags = tags
        for k, s in enumerate(self.I_sliders):
            s.label.setText(f"I{k + 1} · {tags[k]}")
        self._rebuild_axis_combos()
        # Las ecuaciones de Euler valen para cualquier I > 0; la desigualdad triangular solo
        # indica si el cuerpo podría existir como sólido real. Se avisa, no se impide.
        bad = [k for k in range(3) if I[k] > I[(k + 1) % 3] + I[(k + 2) % 3] + 1e-9]
        if bad:
            k = bad[0]
            self.tri_warning.setText(
                f"I{k + 1} > I{(k + 1) % 3 + 1} + I{(k + 2) % 3 + 1}: ningún sólido real tiene estos "
                f"momentos de inercia (desigualdad triangular). Las ecuaciones de Euler siguen valiendo.")
            self.tri_warning.show()
        else:
            self.tri_warning.hide()

    def get(self):
        w = self._vector_from_axis() if self.mode.currentIndex() == 0 else [s.value() for s in self.w_sliders]
        return (tuple(s.value() for s in self.I_sliders), tuple(float(x) for x in w))

    def set(self, I, w0):
        for s, v in zip(self.I_sliders, I):
            s.setValue(v, silent=True)
        self._sync_components(w0)
        self._update_tags()
        self._fill_axis_widgets([s.value() for s in self.w_sliders])
        axis_mode = self.mode.currentIndex() == 0
        self.axis_page.setVisible(axis_mode)
        self.comp_page.setVisible(not axis_mode)


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


def mono_font(size=11):
    fam = MONO.split('"')[1] if '"' in MONO else MONO.split(",")[0]
    f = QtGui.QFont(fam, size)
    f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return f


class DataTable(QtWidgets.QTableWidget):
    """Tabla de solo lectura, sin rejilla ni selección, que se ajusta a su contenido."""

    ALERT = "#FF8A80"

    def __init__(self, headers, parent=None):
        super().__init__(0, len(headers), parent)
        self.setHorizontalHeaderLabels(headers)
        self.verticalHeader().setVisible(False)
        self.verticalHeader().setDefaultSectionSize(28)
        self.setShowGrid(False)
        self.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.NoSelection)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.setWordWrap(False)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        hh = self.horizontalHeader()
        hh.setHighlightSections(False)
        hh.setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        hh.setStretchLastSection(False)
        hh.setDefaultAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Fixed)
        self.setFixedHeight(40)

    def set_rows(self, rows):
        """rows: lista de filas; cada celda es str o (str, opciones) con color, mono, right, tip."""
        self.setRowCount(len(rows))
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                text, opt = (cell, {}) if isinstance(cell, str) else cell
                existing = self.item(r, c)
                it = existing or QtWidgets.QTableWidgetItem()
                it.setText(text)
                it.setForeground(QtGui.QColor(opt.get("color", TEXT)))
                if opt.get("mono"):
                    it.setFont(mono_font(11))
                align = (QtCore.Qt.AlignmentFlag.AlignRight if opt.get("right")
                         else QtCore.Qt.AlignmentFlag.AlignLeft)
                it.setTextAlignment(align | QtCore.Qt.AlignmentFlag.AlignVCenter)
                it.setToolTip(opt.get("tip", ""))
                if existing is None:
                    self.setItem(r, c, it)
        self.resizeColumnsToContents()
        h = self.horizontalHeader().height() + sum(self.rowHeight(i) for i in range(self.rowCount())) + 6
        self.setFixedHeight(max(h, 40))


def dot(cfg):
    return ("●", {"color": cfg.color})


def num(x, fmt="{:+.3e}", alert=False):
    return (fmt.format(x), {"mono": True, "right": True, "color": DataTable.ALERT if alert else TEXT})


class LinkedCursor(QtCore.QObject):
    """Línea vertical que acompaña al ratón en varias gráficas a la vez y avisa del instante."""

    moved = QtCore.Signal(float)               # t bajo el ratón; NaN cuando el ratón sale

    def __init__(self, plots, parent=None):
        super().__init__(parent)
        self.plots, self.lines = plots, []
        pen = pg.mkPen((255, 255, 255, 140), width=1, style=QtCore.Qt.PenStyle.DashLine)
        for p in plots:
            ln = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            ln.setZValue(20)
            ln.hide()
            p.addItem(ln, ignoreBounds=True)
            self.lines.append(ln)
            p.scene().sigMouseMoved.connect(partial(self._on_move, p))
            p.viewport().installEventFilter(self)

    def _on_move(self, plot, pos):
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        x = float(vb.mapSceneToView(pos).x())
        for ln in self.lines:
            ln.setPos(x)
            ln.show()
        self.moved.emit(x)

    def eventFilter(self, obj, ev):
        if ev.type() == QtCore.QEvent.Type.Leave:
            for ln in self.lines:
                ln.hide()
            self.moved.emit(float("nan"))
        return False


class FlowLayout(QtWidgets.QLayout):
    """Coloca los widgets en fila y baja a otra línea cuando no caben (nada queda cortado)."""

    def __init__(self, parent=None, hspacing=10, vspacing=6):
        super().__init__(parent)
        self._items, self._h, self._v = [], hspacing, vspacing

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def itemAt(self, i):
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i):
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self):
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, w):
        return self._layout(QtCore.QRect(0, 0, w, 0), True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        s = QtCore.QSize()
        for it in self._items:
            if not it.isEmpty():
                s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect, test_only):
        m = self.contentsMargins()
        r = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x, y, line_h = r.x(), r.y(), 0
        for it in self._items:
            if it.isEmpty():
                continue
            sz = it.sizeHint()
            if x + sz.width() > r.right() + 1 and line_h > 0:
                x, y, line_h = r.x(), y + line_h + self._v, 0
            if not test_only:
                it.setGeometry(QtCore.QRect(QtCore.QPoint(x, y), sz))
            x += sz.width() + self._h
            line_h = max(line_h, sz.height())
        return y + line_h - rect.y() + m.bottom()


class AdaptiveSplitter(QtWidgets.QSplitter):
    """Reparte en horizontal si hay sitio y en vertical si la ventana es estrecha."""

    def __init__(self, threshold=1000, sizes_h=(1000, 1000), sizes_v=(1000, 1000), parent=None):
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.threshold, self.sizes_h, self.sizes_v = threshold, sizes_h, sizes_v
        self.setChildrenCollapsible(False)

    def showEvent(self, e):
        super().showEvent(e)
        if not getattr(self, "_sized", False):
            self._sized = True
            self.setSizes(list(self.sizes_h if self.orientation() == QtCore.Qt.Orientation.Horizontal
                               else self.sizes_v))

    def minimumSizeHint(self):
        # el mínimo es el del modo vertical (el más estrecho); en horizontal ya no se activa
        w = max((self.widget(i).minimumSizeHint().width() for i in range(self.count())), default=0)
        return QtCore.QSize(w, 2 * 220)

    def resizeEvent(self, e):
        want = (QtCore.Qt.Orientation.Horizontal if e.size().width() >= self.threshold
                else QtCore.Qt.Orientation.Vertical)
        if want != self.orientation():
            self.setOrientation(want)
            self.setSizes(list(self.sizes_h if want == QtCore.Qt.Orientation.Horizontal else self.sizes_v))
        super().resizeEvent(e)


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
QTabWidget#views QTabBar::tab {{ padding: 8px 14px; font-size: 14px; font-weight: 600;
                                 margin-right: 4px; border: 1px solid {BORDER}; border-bottom: none; }}
QTabWidget#views QTabBar::tab:selected {{ background: {ACCENT}; color: white;
                                          border-bottom: 3px solid #FFD740; }}
QTabWidget#views QTabBar::tab:hover:!selected {{ background: #4a4e55; color: white; }}
QSplitter::handle {{ background: {BG}; }}
QTableWidget {{ background: transparent; color: {TEXT}; border: none; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: transparent; color: {MUTED}; border: none;
                        border-bottom: 1px solid {BORDER}; padding: 4px 10px; font-weight: 600; }}
QTableWidget::item {{ padding: 0 10px; }}
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
DEFAULT_PRESET = "Comparar integradores (mismo Δt)"


class MainWindow(QtWidgets.QMainWindow):
    FRAME_MS = 30

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Janibekov Lab — Dinámica de rotación del sólido rígido")
        avail = (QtGui.QGuiApplication.primaryScreen().availableGeometry()
                 if QtGui.QGuiApplication.primaryScreen() else QtCore.QRect(0, 0, 1680, 980))
        self.resize(min(1680, int(avail.width() * 0.96)), min(980, int(avail.height() * 0.92)))

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
        self.view_t = None
        self._div_notified = set()
        self._sat_key = None
        self._last_cfgs = None

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
        QtGui.QShortcut(QtGui.QKeySequence("."), self, activated=self.step_forward)
        QtGui.QShortcut(QtGui.QKeySequence(","), self, activated=self.step_back)
        for k in range(self.view_tabs.count()):
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{k + 1}"), self,
                            activated=partial(self.view_tabs.setCurrentIndex, k))

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        root = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.setChildrenCollapsible(False)
        root.setHandleWidth(10)
        outer.addWidget(root)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(self._build_controls())
        left_w = QtWidgets.QWidget()
        left_w.setMinimumWidth(330)
        left_w.setMaximumWidth(560)
        left = QtWidgets.QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(8)
        left.addWidget(self._build_header())     # fija: Iniciar/Reiniciar siempre a la vista
        left.addWidget(scroll, 1)
        root.addWidget(left_w)

        main_w = QtWidgets.QWidget()
        main = QtWidgets.QVBoxLayout(main_w)
        main.setContentsMargins(0, 0, 0, 0)
        main.setSpacing(8)
        main.addLayout(self._build_banner())

        # ---- Dinámica: órbita en el espacio de fases + movimiento del satélite, a la vez
        self.phase_view = gl.GLViewWidget()
        self.phase_view.setBackgroundColor(VIEW_BG)
        self.sat_view = gl.GLViewWidget()
        self.sat_view.setBackgroundColor(VIEW_BG)
        self.phase_view.setMinimumSize(200, 200)
        self.sat_view.setMinimumSize(200, 200)

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
        self.chk_exact = surface_check("Exacta", "#FFD740",
                                       "Órbita exacta (solución analítica) y, en blanco, dónde estaría ω "
                                       "en este instante: lo que se separa de ahí es error de fase")
        self.trail_spin = QtWidgets.QDoubleSpinBox()
        self.trail_spin.setRange(1, 5000)
        self.trail_spin.setDecimals(0)
        self.trail_spin.setSingleStep(10)
        self.trail_spin.setValue(60)
        self.trail_spin.setSuffix(" s")
        self.trail_spin.setKeyboardTracking(False)
        self.trail_spin.setToolTip("Duración visible de la estela, igual para todas las simulaciones "
                                   "(órbita 3D y plano de fases)")
        bar = FlowLayout(hspacing=12, vspacing=4)
        for c in (self.chk_E, self.chk_L, self.chk_sep, self.chk_fam, self.chk_exact):
            bar.addWidget(c)
        trail_box = QtWidgets.QWidget()
        tb = QtWidgets.QHBoxLayout(trail_box)
        tb.setContentsMargins(0, 0, 0, 0)
        tb.addWidget(QtWidgets.QLabel("Estela"))
        tb.addWidget(self.trail_spin)
        bar.addWidget(trail_box)

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
        self.shape_combo.currentIndexChanged.connect(lambda _: self._rebuild_satellites(force=True))
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

        self.dyn_tab = AdaptiveSplitter(1000)
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
        self.plot_L.setXLink(self.plot_T)
        self.plot_w.setXLink(self.plot_T)
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
        self.drift_readout = QtWidgets.QLabel()
        self.drift_readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.drift_readout.setObjectName("muted")
        log_row.addWidget(self.drift_readout, 1)
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
        self.err_tab = self._build_error_tab()
        self.plane_tab = self._build_plane_tab()
        self.analysis_tab = self._build_analysis_tab()
        for widget, label in ((self.dyn_tab, "🌐  Dinámica"),
                              (self.drift_tab, "📈  Derivas"),
                              (self.err_tab, "📐  Error vs exacta"),
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
        main.addLayout(self._build_timeline())
        root.addWidget(main_w)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([380, 1300])

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
        row = FlowLayout(hspacing=10, vspacing=6)
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
        return row

    def _build_error_tab(self):
        self._err_prev_stale = True
        self.err_comp = QtWidgets.QComboBox()
        self.err_comp.addItems(["ω₁", "ω₂", "ω₃"])
        self.err_comp.setCurrentIndex(1)
        self.err_comp.setToolTip("Componente de ω que se dibuja arriba (los errores usan las tres)")
        self.err_unit = QtWidgets.QComboBox()
        self.err_unit.addItems(["° de ciclo", "segundos"])
        self.err_unit.setToolTip("Unidad del error de fase: grados de un ciclo de la órbita, o "
                                 "tiempo de adelanto/retraso en segundos")
        self.chk_log_err = QtWidgets.QCheckBox("Escala logarítmica |error|")
        self.chk_log_err.setChecked(True)
        self.chk_log_err.setToolTip("Los errores de un esquema y otro difieren en muchos órdenes de "
                                    "magnitud: en log se ven todos a la vez (se pierde el signo; "
                                    "el panel de la derecha lo da)")
        for w in (self.err_comp, self.err_unit):
            w.currentIndexChanged.connect(self._on_error_options)
        self.chk_log_err.toggled.connect(self._on_error_options)
        self.err_readout = QtWidgets.QLabel()
        self.err_readout.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.err_readout.setObjectName("muted")

        bar = FlowLayout(hspacing=12, vspacing=4)
        for w in (QtWidgets.QLabel("Componente"), self.err_comp, QtWidgets.QLabel("Fase en"),
                  self.err_unit, self.chk_log_err, self.err_readout):
            bar.addWidget(w)
        self.err_notice = self._notice_label()

        self.plot_ew = pg.PlotWidget()
        self.plot_er = pg.PlotWidget()
        self.plot_ep = pg.PlotWidget()
        for p, ylab in ((self.plot_ew, "ω₂"), (self.plot_er, "r"), (self.plot_ep, "ψ")):
            self._style_plot(p, "t", ylab)
            p.getPlotItem().setDownsampling(auto=True, mode="peak")
            p.getPlotItem().setClipToView(True)
        self.plot_ew.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.plot_er.setXLink(self.plot_ew)
        self.plot_ep.setXLink(self.plot_ew)
        zero = pg.mkPen((255, 255, 255, 70), width=1, style=QtCore.Qt.PenStyle.DotLine)
        self.err_zero = [self.plot_er.addLine(y=0, pen=zero), self.plot_ep.addLine(y=0, pen=zero)]

        card_w = Card("ω(t): numérica frente a la solución exacta", self.plot_ew,
                      "discontinua blanca = exacta")
        card_r = Card("Error de módulo  r(t)", self.plot_er,
                      "0 = sobre la órbita exacta")
        card_p = Card("Error de fase  ψ(t)", self.plot_ep,
                      "+ adelanta · − retrasa")
        split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        for c in (card_w, card_r, card_p):
            split.addWidget(c)
        split.setSizes([360, 250, 250])

        left = QtWidgets.QWidget()
        left.setObjectName("plain")
        lv = QtWidgets.QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        lv.addLayout(bar)
        lv.addWidget(self.err_notice)
        lv.addWidget(split, 1)

        self.err_info = QtWidgets.QLabel()
        self.err_info.setTextFormat(QtCore.Qt.TextFormat.RichText)
        self.err_info.setWordWrap(True)
        self.err_info.setAlignment(QtCore.Qt.AlignmentFlag.AlignTop)
        sa = QtWidgets.QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sa.setStyleSheet("QScrollArea { background: transparent; }")
        sa.viewport().setAutoFillBackground(False)
        sa.setWidget(self.err_info)
        info = Card("Lectura del error", sa)
        info.setMinimumWidth(250)

        w = AdaptiveSplitter(640, sizes_h=(900, 330), sizes_v=(700, 280))
        w.addWidget(left)
        w.addWidget(info)
        return w

    HOVER_HINT = "Pasa el ratón por las gráficas para leer los valores en un instante."

    def _sims_at(self, t):
        """(sim, índice de fila) de cada simulación activa en el instante t."""
        out = []
        for sim in self.sims:
            if sim.cfg.enabled and sim.nv > 1:
                out.append((sim, int(min(max(round(t / sim.cfg.dt), 0), sim.nv - 1))))
        return out

    def _on_drift_cursor(self, t):
        if t != t:
            self.drift_readout.setText(f"<span style='color:{MUTED}'>{self.HOVER_HINT}</span>")
            return
        parts = [f"<span style='color:{s.cfg.color}'>●</span> {s.cfg.name}: ΔT {s.vdT[i]:+.2e} · "
                 f"Δ‖L‖ {s.vdL[i]:+.2e}" for s, i in self._sims_at(t)]
        self.drift_readout.setText(f"<span style='font-family:{MONO}'>t = {max(t, 0):.2f} s</span> &nbsp; "
                                   + " &nbsp; ".join(parts))

    def _on_err_cursor(self, t):
        if t != t:
            self.err_readout.setText(f"<span style='color:{MUTED}'>{self.HOVER_HINT}</span>")
            return
        parts = [f"<span style='color:{s.cfg.color}'>●</span> {s.cfg.name}: r {s.v_er[i]:+.2e} · "
                 f"ψ {s.v_ep[i]:+.3g}°" for s, i in self._sims_at(t)]
        self.err_readout.setText(f"<span style='font-family:{MONO}'>t = {max(t, 0):.2f} s</span> &nbsp; "
                                 + " &nbsp; ".join(parts))

    def _on_error_options(self, *_):
        self._apply_error_axes()
        self._err_prev_stale = True
        self._ew_cache = None
        self._ew_fit = True
        self._refresh_views()

    def _apply_error_axes(self):
        log = self.chk_log_err.isChecked()
        use_s = self.err_unit.currentIndex() == 1
        comp = "ω" + "₁₂₃"[self.err_comp.currentIndex()]
        self.plot_ew.setLabel("left", comp, **axis_label_style())
        self.plot_er.setLogMode(False, log)
        self.plot_ep.setLogMode(False, log)
        self.plot_er.setLabel("left", "|r|" if log else "r", **axis_label_style())
        unit = "s" if use_s else "°"
        self.plot_ep.setLabel("left", f"|ψ| ({unit})" if log else f"ψ ({unit})", **axis_label_style())
        for z in self.err_zero:
            z.setVisible(not log)

    @staticmethod
    def _fmt_err(d, log):
        """En log: |d|, y por debajo del ruido de redondeo (1e-16) se omite el punto."""
        if not log:
            return d
        a = np.abs(d)
        return np.where(a > 1e-16, a, np.nan)

    @staticmethod
    def _same_exact(a, b):
        return (np.allclose(a.I, b.I, rtol=1e-12, atol=0) and np.allclose(a.w0, b.w0, rtol=1e-12, atol=1e-15))

    def _update_error_notice(self):
        cfgs = self.configs()
        ref = cfgs[self.ref_index()]
        diff = [c.name for k, c in enumerate(cfgs)
                if c.enabled and k != self.ref_index() and not self._same_exact(ref, c)]
        bad = [c.name for c in cfgs if c.enabled and not euler_matches_exact(c.I)]
        if bad:
            self.err_notice.setText(
                "⚠ Las ecuaciones de movimiento de <b>space_physics</b> no coinciden con "
                "I<sub>i</sub> ω̇<sub>i</sub> = (I<sub>j</sub> − I<sub>k</sub>) ω<sub>j</sub> ω<sub>k</sub>, "
                "que es la que resuelve la solución exacta: los errores mostrados no son fiables.")
            self.err_notice.show()
        elif diff:
            self.err_notice.setText(
                f"<b>{' y '.join(diff)}</b> {'usa' if len(diff) == 1 else 'usan'} otra I u otro Ω₀: "
                f"en la gráfica de arriba su solución exacta propia va punteada con su color; "
                f"los errores siempre se miden frente a la exacta de cada simulación.")
            self.err_notice.show()
        else:
            self.err_notice.hide()

    def _exact_curve(self, exact, t_end, comp, live):
        """Solución exacta ω_comp(t) en una malla propia (es analítica: no depende de Δt)."""
        if t_end <= 0:
            return np.empty(0), np.empty(0)
        per = exact.period
        n = 1500 if not math.isfinite(per) else int(min(8000 if live else 40000, max(600, 100 * t_end / per)))
        t = np.linspace(0.0, t_end, n)
        return t, exact.omega(t)[:, comp]

    def _build_plane_tab(self):
        self.plot_plane = pg.PlotWidget()
        self._style_plot(self.plot_plane, "ω_i", "ω_j")
        self.plot_plane.setAspectLocked(True)
        self.plot_plane.addLegend(offset=(8, 8), labelTextSize="9pt")

        controls = FlowLayout(hspacing=10, vspacing=4)
        controls.addWidget(QtWidgets.QLabel("Equilibrio"))
        self.plane_axis = QtWidgets.QComboBox()
        self.plane_axis.addItems(["Automático (eje de la órbita de ref.)",
                                  "e₁  (ω₁ = ±Ω)", "e₂  (ω₂ = ±Ω)", "e₃  (ω₃ = ±Ω)"])
        self.plane_axis.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.plane_axis.setMinimumContentsLength(12)
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
        self.plane_sign_lbl = QtWidgets.QLabel("Signo")
        controls.addWidget(self.plane_sign_lbl)
        controls.addWidget(self.plane_sign)
        self.chk_plane_field.toggled.connect(self._update_phase_plane)
        controls.addWidget(self.chk_plane_field)

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
        info.setMinimumWidth(240)
        iv = QtWidgets.QVBoxLayout(info)
        iv.setContentsMargins(14, 12, 14, 12)
        iv.setSpacing(10)
        t = QtWidgets.QLabel("Estabilidad del equilibrio")
        t.setObjectName("cardTitle")
        iv.addWidget(t)
        self.plane_badge = QtWidgets.QLabel()
        self.plane_badge.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.plane_badge.setWordWrap(True)
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

        w = AdaptiveSplitter(900, sizes_h=(720, 330), sizes_v=(620, 300))
        w.addWidget(self.plane_card)
        w.addWidget(info)
        return w

    def _build_analysis_tab(self):
        def body(*widgets):
            w = QtWidgets.QWidget()
            w.setObjectName("plain")
            v = QtWidgets.QVBoxLayout(w)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(8)
            for x in widgets:
                v.addWidget(x)
            v.addStretch(1)
            return w

        def note(text):
            lbl = QtWidgets.QLabel(text)
            lbl.setObjectName("muted")
            lbl.setWordWrap(True)
            return lbl

        # conservación (en vivo)
        self.cons_table = DataTable(["", "Esquema", "Δt", "t (s)", "ΔT", "Δ‖L‖", "‖q‖ − 1", "pasos"])
        cons_card = Card("Conservación", body(self.cons_table), "en vivo")

        # orden del método: Richardson frente al error real
        ctl = QtWidgets.QWidget()
        cl = QtWidgets.QHBoxLayout(ctl)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.addWidget(QtWidgets.QLabel("Instante final T"))
        self.rich_T = QtWidgets.QDoubleSpinBox()
        self.rich_T.setRange(0.1, 200)
        self.rich_T.setValue(10)
        self.rich_T.setSuffix(" s")
        cl.addWidget(self.rich_T)
        btn_rich = QtWidgets.QPushButton("Calcular")
        btn_rich.setToolTip("Integra con n, 2n y 4n pasos hasta T (n = T/Δt) para cada simulación activa")
        btn_rich.clicked.connect(self.run_richardson)
        cl.addWidget(btn_rich)
        cl.addStretch(1)
        self.rich_table = DataTable(["", "Esquema", "Δt", "p nominal", "p estimado",
                                     "error estimado", "error real"])
        self.rich_note = note("Pulsa «Calcular». Estimado: Richardson sobre el estado completo "
                              "(ω y cuaternión). Real: ‖ω − ω exacta‖ en T con la solución analítica.")
        self.rich_table.hide()
        rich_card = Card("Orden y error del método", body(ctl, self.rich_table, self.rich_note))

        # equilibrios de la simulación de referencia y efecto del esquema
        self.eq_table = DataTable(["Eje", "Ω", "Tipo", "λ"])
        self.eff_table = DataTable(["", "Esquema", "Rodea", "|R| − 1", "Efecto"])
        eq_card = Card("Equilibrios", body(
            note("Rotaciones puras ω = ±Ω eₖ de la simulación de referencia (Iₖ Ω² = 2T)."),
            self.eq_table,
            note("Efecto de cada esquema sobre el centro que rodea su órbita: |R(iβΔt)| − 1 "
                 "mide cuánto crece (+) o decae (−) la amplitud en cada paso."),
            self.eff_table), "ref.")
        self.eq_card = eq_card

        left_w = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.addWidget(cons_card)
        left.addWidget(rich_card)
        left.addStretch(1)
        w = AdaptiveSplitter(900, sizes_h=(620, 480))
        w.addWidget(left_w)
        w.addWidget(eq_card)
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
        g.addWidget(QtWidgets.QLabel("Parar en"), 2, 0)
        self.stop_T = QtWidgets.QDoubleSpinBox()
        self.stop_T.setRange(0, 5000)
        self.stop_T.setDecimals(1)
        self.stop_T.setSingleStep(5)
        self.stop_T.setSuffix(" s")
        self.stop_T.setSpecialValueText("sin límite")
        self.stop_T.setKeyboardTracking(False)
        self.stop_T.setToolTip("Pausa automática al llegar a este instante (0 = sin límite)")
        g.addWidget(self.stop_T, 2, 1)
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

        # órbita exacta (una por simulación que la necesite) y marcador «dónde debería estar ahora»
        self.exact_lines, self.exact_heads, self._exact_show = [], [], [False] * len(DEFAULT_CONFIGS)
        for _ in DEFAULT_CONFIGS:
            el = gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=2, cols=3), smooth=True,
                               color=(1.0, 0.84, 0.25, 1.0), shader=None, glOptions="opaque")
            eh = gl.GLScatterPlotItem(pos=np.zeros((1, 3)), color=(1, 1, 1, 1), size=20, pxMode=True)
            eh.setGLOptions("translucent")
            for it in (el, eh):
                it.hide()
                v.addItem(it)
            self.exact_lines.append(el)
            self.exact_heads.append(eh)

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
        for k, cfg in enumerate(DEFAULT_CONFIGS):
            for plot, live, prev in ((self.plot_T, self.curves_T, self.prev_T),
                                     (self.plot_L, self.curves_L, self.prev_L)):
                prev.append(plot.plot(pen=self._dash_pen(cfg.color)))
                live.append(plot.plot(pen=self._live_pen(k, cfg.color), name=cfg.name))
        self.curves_w, self.prev_w = [], []
        for k in range(3):
            self.prev_w.append(self.plot_w.plot(pen=self._dash_pen(AXIS_COLORS[k])))
            self.curves_w.append(self.plot_w.plot(pen=pg.mkPen(AXIS_COLORS[k], width=2),
                                                  name=f"ω{k + 1}"))

        self.ew_live, self.ew_prev, self.ew_own = [], [], []
        self.er_live, self.er_prev, self.ep_live, self.ep_prev = [], [], [], []
        self.ew_exact = self.plot_ew.plot(
            pen=pg.mkPen("#FFFFFF", width=2.2, style=QtCore.Qt.PenStyle.DashLine), name="exacta")
        self.ew_exact.setZValue(5)
        for k_, cfg in enumerate(DEFAULT_CONFIGS):
            self.ew_prev.append(self.plot_ew.plot(pen=self._dash_pen(cfg.color), connect="finite"))
            own = self.plot_ew.plot(pen=pg.mkPen(cfg.color, width=1.5, style=QtCore.Qt.PenStyle.DotLine))
            own.setZValue(4)
            self.ew_own.append(own)
            self.ew_live.append(self.plot_ew.plot(pen=self._live_pen(k_, cfg.color), name=cfg.name,
                                                  connect="finite"))
            for plot, live, prev in ((self.plot_er, self.er_live, self.er_prev),
                                     (self.plot_ep, self.ep_live, self.ep_prev)):
                prev.append(plot.plot(pen=self._dash_pen(cfg.color), connect="finite"))
                live.append(plot.plot(pen=self._live_pen(k_, cfg.color), connect="finite"))
        self._ew_cache = None
        self._ew_fit = True
        self._apply_error_axes()
        self.cursor_drift = LinkedCursor([self.plot_T, self.plot_L, self.plot_w], self)
        self.cursor_drift.moved.connect(self._on_drift_cursor)
        self.cursor_err = LinkedCursor([self.plot_ew, self.plot_er, self.plot_ep], self)
        self.cursor_err.moved.connect(self._on_err_cursor)
        self._on_drift_cursor(float("nan"))
        self._on_err_cursor(float("nan"))

        pp = self.plot_plane
        self.plane_domain = pp.plot(pen=pg.mkPen((255, 255, 255, 120), width=1.5,
                                                 style=QtCore.Qt.PenStyle.DotLine))
        self.plane_field = pp.plot(pen=pg.mkPen((200, 200, 205, 120), width=1), connect="pairs")
        self.plane_orbit_items, self.plane_sep_items = [], []
        self.plane_ref_orbit = pp.plot(pen=pg.mkPen("#FFD740", width=2,
                                                    style=QtCore.Qt.PenStyle.DashLine),
                                       name="órbita exacta (ref.)")
        self.plane_prev, self.plane_trail, self.plane_head = [], [], []
        for k_, cfg in enumerate(DEFAULT_CONFIGS):
            self.plane_prev.append(pp.plot(pen=self._dash_pen(cfg.color), connect="finite"))
            self.plane_trail.append(pp.plot(pen=self._live_pen(k_, cfg.color, 2.2), connect="finite",
                                            name=cfg.name))
            self.plane_head.append(pp.plot(pen=None, symbol="o", symbolSize=10,
                                           symbolBrush=cfg.color, symbolPen="w"))
        self.plane_eq = pp.plot(pen=None, symbol="star", symbolSize=22, symbolPen="w")
        self.plane_eq.setZValue(10)

    LINE_PATTERNS = (None, [5, 2.5], [1.2, 2.4])      # A continuo · B rayas · C puntos

    @classmethod
    def _live_pen(cls, k, color, width=2.0):
        """Trazo de la simulación k: además del color, A es continua, B rayada y C punteada
        (para distinguirlas sin depender del color)."""
        pen = pg.mkPen(color, width=width)
        pat = cls.LINE_PATTERNS[k % len(cls.LINE_PATTERNS)]
        if pat is not None:
            pen.setStyle(QtCore.Qt.PenStyle.CustomDashLine)
            pen.setDashPattern(pat)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        return pen

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
        changed = self._describe_changes(self._last_cfgs, cfgs)
        self._last_cfgs = cfgs
        if not cfgs[self.ref_index()].enabled:
            first = next((k for k, c in enumerate(cfgs) if c.enabled), None)
            if first is not None:
                self.ref_combo.blockSignals(True)
                self.ref_combo.setCurrentIndex(first)
                self.ref_combo.blockSignals(False)
        # todas las simulaciones se reinician juntas: así siguen en el mismo instante
        self.sims = [Simulation(c) for c in cfgs]
        self.clock = 0.0
        self.view_t = None
        self._div_notified = set()
        self._pause()
        self._sync_timeline()
        self.previews = {}
        self._update_static_phase()
        self._update_exact_orbits()
        self._update_equilibria()
        self._update_phase_plane()
        self._rebuild_satellites()
        self._refresh_tab_titles()
        self._update_surface_notice()
        self._update_error_notice()
        self._badge_state = [None] * len(self.badges)
        self.plot_w.setTitle(None)
        self._update_preview_items()
        if self.chk_auto.isChecked():
            self.request_preview()
        else:
            self._set_preview_status("desactualizada · pulsa ↻ Calcular [P]")
        self.status.showMessage(f"Parámetros aplicados · simulación reiniciada (t = 0){changed}", 4000)
        self._refresh_views()

    @staticmethod
    def _describe_changes(old, new):
        """Qué cambió respecto a la última aplicación, para explicar por qué se reinicia."""
        if not old or len(old) != len(new):
            return ""
        fields = (("enabled", "activación"), ("method", "método"), ("dt", "Δt"), ("I", "I"), ("w0", "Ω₀"))
        out = []
        for a, b in zip(old, new):
            ch = [lab for attr, lab in fields if getattr(a, attr) != getattr(b, attr)]
            if ch:
                out.append(f"{b.name}: {', '.join(ch)}")
        if not out:
            return ""
        return " · cambió " + ("; ".join(out) if len(out) <= 3 else "varias simulaciones")

    def _on_horizon_changed(self):
        if self.chk_auto.isChecked() or self.previews:
            self.request_preview()

    def reset(self):
        for s in self.sims:
            s.reset()
        self.clock = 0.0
        self.view_t = None
        self._div_notified = set()
        self._ew_cache = None
        self._sync_timeline()
        self._refresh_views()

    def _build_timeline(self):
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(6)
        self.btn_back = QtWidgets.QPushButton("◀")
        self.btn_fwd = QtWidgets.QPushButton("▶")
        for b, tip, slot in ((self.btn_back, "Un paso atrás  [,]", self.step_back),
                             (self.btn_fwd, "Un paso adelante  [.]", self.step_forward)):
            b.setFixedWidth(38)
            b.setToolTip(tip + "\nUn paso = el Δt más pequeño de las simulaciones activas")
            b.clicked.connect(slot)
            row.addWidget(b)
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.timeline.setRange(0, 10000)
        self.timeline.setValue(10000)
        self.timeline.setToolTip("Arrastra hacia atrás para revisar lo ya simulado; con Espacio se "
                                 "reproduce hasta el presente y sigue en vivo")
        self.timeline.sliderPressed.connect(self._pause)
        self.timeline.valueChanged.connect(self._on_timeline)
        row.addWidget(self.timeline, 1)
        self.timeline_label = QtWidgets.QLabel("t = 0.00 s")
        self.timeline_label.setObjectName("muted")
        self.timeline_label.setMinimumWidth(120)
        self.timeline_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self.timeline_label)
        self.btn_live = QtWidgets.QPushButton("● En vivo")
        self.btn_live.setToolTip("Volver al presente de la simulación")
        self.btn_live.clicked.connect(lambda: self._set_view_time(None))
        row.addWidget(self.btn_live)
        self._syncing = False
        return row

    # ------------------------------------------------------------------ tiempo
    def _pause(self, message=None):
        self.running = False
        self.btn_run.setText("▶  Iniciar")
        if message:
            self.status.showMessage(message, 8000)

    def toggle_run(self):
        self.running = not self.running
        self.btn_run.setText("⏸  Pausa" if self.running else "▶  Iniciar")
        if self.running and self.view_t is None:
            stop = self.stop_T.value()
            if stop > 0 and self.clock >= stop - 1e-12:
                self._pause(f"Pausa automática: ya se alcanzó T = {stop:g} s (cambia «Parar en» para seguir)")

    def _set_view_time(self, t):
        """None = en vivo; un instante = se muestra el historial hasta ahí (rebobinado)."""
        if t is None or t >= self.clock - 1e-12:
            self.view_t = None
            for s in self.sims:
                s.view_n = None
        else:
            self.view_t = max(0.0, float(t))
            for s in self.sims:
                s.view_n = int(round(self.view_t / s.cfg.dt)) + 1
        self._sync_timeline()
        self._ew_cache = None
        self._refresh_views()

    def _sync_timeline(self):
        self._syncing = True
        frac = 1.0 if (self.view_t is None or self.clock <= 0) else self.view_t / self.clock
        self.timeline.setValue(int(round(10000 * min(max(frac, 0.0), 1.0))))
        self._syncing = False
        live = self.view_t is None
        self.btn_live.setEnabled(not live)
        self.timeline.setEnabled(self.clock > 0)
        self.btn_back.setEnabled(self.clock > 0)
        shown = self.clock if live else self.view_t
        self.clock_label.setText(f"t = {self.clock:.2f} s" if live else f"⏪ t = {shown:.2f} s")
        self.timeline_label.setText(f"t = {self.clock:.2f} s" if live
                                    else f"t = {shown:.2f} / {self.clock:.2f} s")

    def _on_timeline(self, v):
        if self._syncing or self.clock <= 0:
            return
        self._pause()
        self._set_view_time(None if v >= 10000 else v / 10000.0 * self.clock)

    def _step_size(self):
        dts = [s.cfg.dt for s in self.sims if s.cfg.enabled]
        return min(dts) if dts else 0.0

    def step_forward(self):
        self._pause()
        h = self._step_size()
        if h <= 0:
            return
        if self.view_t is not None:
            self._set_view_time(self.view_t + h)
            return
        self._advance_live(h)
        self._refresh_views()

    def step_back(self):
        self._pause()
        h = self._step_size()
        if h <= 0 or self.clock <= 0:
            return
        base = self.clock if self.view_t is None else self.view_t
        self._set_view_time(max(0.0, base - h))

    def _advance_live(self, dt_clock):
        """Avanza el reloj y las simulaciones; aplica «Parar en», divergencia y límite de pasos."""
        stop = self.stop_T.value()
        target = self.clock + dt_clock
        if stop > 0:
            if self.clock >= stop - 1e-12:
                self._pause(f"Pausa automática: ya se alcanzó T = {stop:g} s")
                return False
            target = min(target, stop)
        self.clock = target
        for sim in self.sims:
            if sim.cfg.enabled:
                sim.advance_to(self.clock)
        self._sync_timeline()
        for k, sim in enumerate(self.sims):
            if sim.cfg.enabled and sim.diverged and k not in self._div_notified:
                self._div_notified.add(k)
                self._pause(f"Pausa automática: {sim.cfg.name} divergió en t = {sim.t:.2f} s")
            elif sim.cfg.enabled and sim.n_steps >= MAX_LIVE_STEPS:
                self._pause(f"Pausa automática: {sim.cfg.name} alcanzó {MAX_LIVE_STEPS:,} pasos")
        if stop > 0 and self.clock >= stop - 1e-12:
            self._pause(f"Pausa automática: se alcanzó T = {stop:g} s")
        return True


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
                ok_ = np.isfinite(r["p_est"]) and np.isfinite(r["err"])
                w = r["u_extr"][:3]
                tip = (f"n = {r['n']} pasos · ω extrapolada = ({w[0]:+.5f}, {w[1]:+.5f}, {w[2]:+.5f})"
                       if ok_ else "no estimable (divergencia)")
                rows.append([dot(c), (f"{METHOD_SHORT[c.method]}", {"tip": tip}), (f"{c.dt:g}", {"mono": True}),
                             (f"{r['p']}", {"mono": True, "right": True}),
                             (f"{r['p_est']:.2f}", {"mono": True, "right": True}) if ok_
                             else ("—", {"right": True, "color": MUTED}),
                             (f"{r['err']:.2e}", {"mono": True, "right": True}) if ok_
                             else ("divergió", {"right": True, "color": DataTable.ALERT}),
                             (f"{r['err_real']:.2e}", {"mono": True, "right": True})
                             if np.isfinite(r["err_real"]) else ("—", {"right": True, "color": MUTED})])
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        self.rich_table.set_rows(rows)
        self.rich_table.setVisible(bool(rows))
        self.rich_note.setText(f"Hasta T = {T:g} s. Estimado: Richardson sobre el estado completo "
                               f"(ω y cuaternión). Real: ‖ω − ω exacta‖ con la solución analítica."
                               if rows else "No hay simulaciones activas.")
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
        self._err_prev_stale = True
        self._ew_cache = None
        self._ew_fit = True
        if self.view_tabs.currentWidget() is self.err_tab:
            self._update_error_preview()

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
        self.plane_sign_lbl.setText("Signo" if idx != 0 else "Signo (auto)")
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
            f"<b>λ</b> = <span style='font-family:{MONO}'>{lam}</span>",
            f"<b>Qué se ve</b><br>{shape}",
            f"<span style='color:{MUTED}'>Trazo continuo: simulación · discontinuo: previsualización · "
            f"amarillo: órbita exacta de ref.</span>",
            "<b>Efecto de cada esquema aquí</b>",
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
            lam_c = ev[np.argmax(np.abs(ev.imag))] if ec["type"] == "centro" else ev[np.argmax(ev.real)]
            R = abs(AMPLIFICATION[c.method](lam_c * c.dt))
            if ec["type"] == "centro":
                g = R - 1
                what = ("crece" if g > 1e-10 else "decae" if g < -1e-10 else "se conserva")
                rows.append(head + f"<span style='font-family:{MONO}'>|R|−1 = {g:+.2e}</span> · amplitud {what}")
            else:
                rows.append(head + f"<span style='font-family:{MONO}'>|R(λΔt)| = {R:.4f}</span>")
        self.plane_info.setText("<br><br>".join(rows))
    def _refresh_phase_plane(self):
        for k, sim in enumerate(self.sims):
            if sim.cfg.enabled and sim.nv > 1:
                W = sim.vY[max(0, sim.nv - self._trail_points(sim)):, :3]
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
        self._update_exact_orbits()
        self._update_equilibria()
        self._update_phase_plane()
        self._update_surface_notice()
        self._update_error_notice()
        self._update_preview_items()
        self._refresh_views()

    def _update_static_phase(self):
        cfg = self.configs()[self.ref_index()]
        I, w0 = np.array(cfg.I), np.array(cfg.w0)
        semi_E, semi_L = poinsot_semiaxes(I, w0)
        twoT = 2 * float(kinetic_energy(w0, I))

        for mesh, semi in ((self.mesh_E, semi_E), (self.mesh_L, semi_L)):
            md = gl.MeshData.sphere(rows=36, cols=72)
            md.setVertexes(md.vertexes() * np.maximum(semi, 1e-6))
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
        self._phase_lim = lim
        for k in range(3):
            e = np.zeros(3)
            e[k] = lim
            self.phase_axes[k].setData(pos=np.array([-e, e]))
            self.phase_axis_labels[k].setData(pos=1.08 * e)
        self.phase_view.setCameraPosition(distance=1.9 * lim)
        self.phase_card.subtitle.setText(
            f"ref. {cfg.name}:  2T = {twoT:.4g}   ‖L‖ = {np.sqrt(angular_momentum_sq(w0, I)):.4g}")
        self._update_static_visibility()

    def _update_equilibria(self):
        cfgs = self.configs()
        ref_cfg = cfgs[self.ref_index()]
        ref = equilibrium_diagnosis(ref_cfg)
        pos, cols, rows = [], [], []
        for k, e in enumerate(ref["eqs"]):
            col = STABILITY_COLORS[e["stability"]]
            Om = e["omega"][k]
            lam = e["eigvals"][np.argmax(np.abs(e["eigvals"]))]
            lam_txt = "0" if abs(lam) < 1e-12 else (f"±{abs(lam.imag):.4g} i" if abs(lam.imag) > abs(lam.real)
                                                    else f"±{abs(lam.real):.4g}")
            bad = e["stability"] == "inestable"
            rows.append([(f"ω{'₁₂₃'[k]}", {}),
                         (f"±{Om:.4g}", {"mono": True, "right": True}),
                         (f"{e['type']} · {e['stability']}", {"color": DataTable.ALERT if bad else TEXT}),
                         (lam_txt, {"mono": True, "right": True})])
            pos += [e["omega"], -e["omega"]]
            cols += [rgba(col, 1.0)] * 2
        self.eq_table.set_rows(rows)
        self.eq_card.subtitle.setText(f"ref. {ref_cfg.name}")
        self.eq_points.setData(pos=np.array(pos), color=np.array(cols))

        eff = []
        for c in cfgs:
            if not c.enabled:
                continue
            d = equilibrium_diagnosis(c)
            if d["amp"] is None:
                eff.append([dot(c), METHOD_SHORT[c.method], ("—", {"color": MUTED}), ("", {}),
                            ("separatriz o trompo", {"color": MUTED})])
                continue
            g = d["amp"] - 1
            if g > 1e-10:
                txt, alert = "la amplitud crece", True
            elif g < -1e-10:
                txt, alert = "la amplitud decae (disipa)", False
            else:
                txt, alert = "conserva la amplitud", False
            eff.append([dot(c), METHOD_SHORT[c.method], (f"ω{'₁₂₃'[d['axis']]}", {}),
                        num(g, "{:+.2e}", alert), (txt, {"color": DataTable.ALERT if alert else TEXT})])
        self.eff_table.set_rows(eff)

    def _update_exact_orbits(self):
        """Órbita exacta de la referencia y de las simulaciones cuya física difiere (otra I u otro Ω₀)."""
        if not self.sims:
            return
        ref_k = self.ref_index()
        ref = self.sims[ref_k]
        for k, sim in enumerate(self.sims):
            own = sim.cfg.enabled and (k == ref_k or not self._same_exact(ref.cfg, sim.cfg))
            P = sim.exact.orbit(720) if own else None
            self._exact_show[k] = P is not None and len(P) > 1
            if self._exact_show[k]:
                is_ref = k == ref_k
                # tubo (no una línea): el grosor de las líneas GL depende de la tarjeta gráfica
                r = (0.011 if is_ref else 0.0075) * self._phase_lim
                V, F = tube(P[::2], r, n=10, per=1, caps=False)
                self.exact_lines[k].setMeshData(meshdata=gl.MeshData(vertexes=V, faces=F))
                self.exact_lines[k].setColor((1.0, 0.84, 0.25, 1.0) if is_ref else rgba(sim.cfg.color, 1.0))
        self._update_static_visibility()
        self._update_exact_markers()

    def _update_exact_markers(self):
        if not self.sims:
            return
        t_now = self.clock if self.view_t is None else self.view_t
        for k, sim in enumerate(self.sims):
            if self._exact_show[k] and self.chk_exact.isChecked():
                self.exact_heads[k].setData(pos=sim.exact.omega(np.array([t_now])))
            else:
                self.exact_heads[k].setData(pos=np.zeros((1, 3)))

    def _update_static_visibility(self):
        for k in range(len(self.exact_lines)):
            vis = self._exact_show[k] and self.chk_exact.isChecked()
            self.exact_lines[k].setVisible(vis)
            self.exact_heads[k].setVisible(vis)

        self.mesh_E.setVisible(self.chk_E.isChecked())
        self.mesh_L.setVisible(self.chk_L.isChecked())
        for it in self.separatrix_items:
            it.setVisible(self.chk_sep.isChecked())
        for it in self.family_items:
            it.setVisible(self.chk_fam.isChecked())

    def _rebuild_satellites(self, force=False):
        cfgs = [c for c in self.configs() if c.enabled]
        # reconstruir las mallas es lo más caro de aplicar parámetros: solo si algo del cuerpo cambió
        key = (self.shape_combo.currentText(),
               tuple((c.name, c.color, c.method, tuple(c.I)) for c in cfgs))
        if not force and key == self._sat_key and self.sat_items:
            self._update_satellites()
            return
        self._sat_key = key
        for group in self.sat_items:
            for it in group["all"]:
                self.sat_view.removeItem(it)
        self.sat_items = []
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
            y = np.array(sim.vy)
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
        step = self.speed.value() * self.FRAME_MS / 1000.0
        if self.view_t is not None:                 # reproduciendo el historial hasta el presente
            self._set_view_time(self.view_t + step)
            return
        self._advance_live(step)
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

    def _refresh_error_tab(self):
        log = self.chk_log_err.isChecked()
        use_s = self.err_unit.currentIndex() == 1
        comp = self.err_comp.currentIndex()
        for k, sim in enumerate(self.sims):
            if sim.cfg.enabled and sim.nv > 1:
                t = sim.vtimes
                self.ew_live[k].setData(t, sim.vY[:, comp])
                self.er_live[k].setData(t, self._fmt_err(sim.v_er, log))
                self.ep_live[k].setData(t, self._fmt_err(sim.v_el if use_s else sim.v_ep, log))
            else:
                for c in (self.ew_live, self.er_live, self.ep_live):
                    c[k].setData([], [])
        if self._err_prev_stale:
            self._update_error_preview()
        self._update_exact_curves(comp)
        if self.frame % 4 == 0 or not self.running:
            self._update_error_info()

    def _update_exact_curves(self, comp):
        if not self.sims:
            return
        show = self.chk_show_prev.isChecked()
        ref_k = self.ref_index()
        ref = self.sims[ref_k]
        t_cut = {}
        for k, sim in enumerate(self.sims):
            horizon = 0.0
            p = self.previews.get(k)
            if show and p is not None and len(p["t"]) > 1:
                horizon = float(p["t"][-1])
            t_cut[k] = max(sim.vt, horizon)
        key = (comp, id(ref.exact), round(t_cut[ref_k], 6), tuple(
            (k, id(s.exact), round(t_cut[k], 6)) for k, s in enumerate(self.sims)))
        # con la estela en vivo el tiempo cambia en cada fotograma: se recalcula siempre
        if ref.cfg.enabled and t_cut[ref_k] > 0:
            live_only = t_cut[ref_k] == ref.vt
            if self._ew_cache is None or self._ew_cache[0] != key or live_only:
                t, y = self._exact_curve(ref.exact, t_cut[ref_k], comp, live_only)
                self.ew_exact.setData(t, y)
                self._ew_cache = (key,)
                if self._ew_fit and len(y) > 1:
                    # escala según la solución exacta: si la numérica diverge se sale del recuadro
                    lo, hi = float(np.min(y)), float(np.max(y))
                    pad = 0.3 * (hi - lo) if hi - lo > 1e-12 else max(1.0, abs(hi))
                    self.plot_ew.setYRange(lo - pad, hi + pad, padding=0)
                    self._ew_fit = False
        else:
            self.ew_exact.setData([], [])
        for k, sim in enumerate(self.sims):
            own = self.ew_own[k]
            if (sim.cfg.enabled and k != ref_k and ref.cfg.enabled and t_cut[k] > 0
                    and not self._same_exact(ref.cfg, sim.cfg)):
                t, y = self._exact_curve(sim.exact, t_cut[k], comp, True)
                own.setData(t, y)
            else:
                own.setData([], [])

    def _update_error_preview(self):
        self._err_prev_stale = False
        show = self.chk_show_prev.isChecked()
        log = self.chk_log_err.isChecked()
        use_s = self.err_unit.currentIndex() == 1
        comp = self.err_comp.currentIndex()
        for k in range(len(self.ew_prev)):
            p = self.previews.get(k)
            if show and p is not None and len(p["t"]) > 1:
                self.ew_prev[k].setData(p["t"], p["W"][:, comp])
                self.er_prev[k].setData(p["t"], self._fmt_err(p["r"], log))
                self.ep_prev[k].setData(p["t"], self._fmt_err(p["lag"] if use_s else p["psi"], log))
            else:
                for c in (self.ew_prev, self.er_prev, self.ep_prev):
                    c[k].setData([], [])

    def _update_error_info(self):
        rows = [f"<span style='color:{MUTED}'><b>Módulo r</b>: separación radial respecto a la órbita "
                f"exacta (0 si el punto está sobre ella).<br><b>Fase ψ</b>: cuánto adelanta (+) o "
                f"retrasa (−) respecto a la exacta <i>en el mismo instante</i>.<br>"
                f"Los dos se separan en coordenadas polares de la órbita: ‖·‖ = módulo, ángulo = fase."
                f"</span><br>"]
        any_row = False
        for k, sim in enumerate(self.sims):
            c = sim.cfg
            if not c.enabled:
                continue
            any_row = True
            ex = sim.exact
            if ex.kind == "jacobi":
                per = (f"periodo {ex.period:.4g} s" if math.isfinite(ex.period) else "no periódica (separatriz)")
                what = f"Jacobi · m = {ex.m:.5f} · {per}"
            elif ex.kind == "trompo":
                what = f"precesión uniforme · periodo {ex.period:.4g} s"
            else:
                what = "equilibrio: ω constante"
            rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                        f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g}<br>"
                        f"<span style='color:{MUTED}'>exacta: {what}</span>")
            if sim.diverged:
                rows.append(pill("DIVERGIÓ", STABILITY_COLORS["inestable"]))
            elif sim.nv > 1:
                r, psi, lag = sim.v_er[-1], sim.v_ep[-1], sim.v_el[-1]
                rows.append(f"<span style='font-family:{MONO}'>ahora (t={sim.vt:.2f} s)<br>"
                            f"&nbsp;módulo r = {r:+.3e}<br>"
                            f"&nbsp;fase ψ = {psi:+.4g}° &nbsp;({lag:+.3e} s)</span>")
            lin = linear_step_errors(c)
            if lin is None:
                rows.append(f"<span style='color:{MUTED}'>sin centro lineal que analizar "
                            f"(separatriz, trompo o eje intermedio)</span>")
            else:
                dm, dp = lin["dmod"], lin["dphase"]
                nst = sim.v_steps
                pred_r = float(np.expm1(nst * math.log(abs(lin["R"])))) if abs(lin["R"]) > 0 else -1.0
                pred_psi = math.degrees(dp) * nst
                mod_v = (pill("conserva el módulo", STABILITY_COLORS["marginal"]) if abs(dm) < 1e-10 else
                         pill("módulo crece", STABILITY_COLORS["inestable"]) if dm > 0 else
                         pill("módulo decae", STABILITY_COLORS["estable"]))
                pha_v = (pill("sin error de fase", STABILITY_COLORS["marginal"]) if abs(dp) < 1e-10 else
                         pill("va por delante", STABILITY_COLORS["inestable"]) if dp > 0 else
                         pill("va por detrás", STABILITY_COLORS["inestable"]))
                rows.append(f"<span style='font-family:{MONO}'>por paso, esquema vs e^(iβΔt):<br>"
                            f"&nbsp;|R|−1 = {dm:+.2e} &nbsp;{mod_v}<br>"
                            f"&nbsp;arg R − βΔt = {dp:+.2e} rad &nbsp;{pha_v}<br>"
                            f"teoría lineal a t={sim.vt:.2f} s: r ≈ {pred_r:+.2e}, ψ ≈ {pred_psi:+.3g}°</span>")
            rows.append("")
        if not any_row:
            rows.append("Sin simulaciones activas.")
        self.err_info.setText("<br>".join(rows))

    def _refresh_views(self):
        """Actualiza solo la pestaña visible; al cambiar de pestaña se vuelve a llamar."""
        cur = self.view_tabs.currentWidget()
        log = self.chk_log.isChecked()
        if cur is self.dyn_tab:
            for k, sim in enumerate(self.sims):
                live = sim.cfg.enabled and sim.nv > 1
                self.trail_lines[k].setVisible(live)
                self.heads[k].setVisible(sim.cfg.enabled)
                if sim.cfg.enabled:
                    W = sim.vY[max(0, sim.nv - self._trail_points(sim)):, :3]
                    if live:
                        self.trail_lines[k].setData(pos=W)
                    self.heads[k].setData(pos=W[-1:])
            self._update_satellites()
            self._update_exact_markers()
        elif cur is self.drift_tab:
            for k, sim in enumerate(self.sims):
                if sim.cfg.enabled and sim.nv > 1:
                    t = sim.vtimes
                    self.curves_T[k].setData(t, self._fmt(sim.vdT, log))
                    self.curves_L[k].setData(t, self._fmt(sim.vdL, log))
                else:
                    self.curves_T[k].setData([], [])
                    self.curves_L[k].setData([], [])
            ref = self.sims[self.ref_index()] if self.sims else None
            for k in range(3):
                if ref is not None and ref.cfg.enabled and ref.nv > 1:
                    self.curves_w[k].setData(ref.vtimes, ref.vY[:, k])
                else:
                    self.curves_w[k].setData([], [])
        elif cur is self.err_tab:
            self._refresh_error_tab()
        elif cur is self.plane_tab:
            self._refresh_phase_plane()
        if self.frame % 4 == 0 or not self.running:
            self._update_status()
        self.frame += 1

    def _update_status(self):
        show_detail = self.view_tabs.currentWidget() is self.analysis_tab
        rows = []
        for k, sim in enumerate(self.sims):
            c = sim.cfg
            if not c.enabled:
                self._set_badge(k, None)
                rows.append([dot(c), (f"{c.name} — inactiva", {"color": MUTED}), "", "", "", "", "", ""])
                continue
            r = regime(c.I, c.w0)
            diverged = sim.diverged and sim.nv >= sim.n
            if diverged:
                obs, alert = f"divergió en t = {sim.t:.2f} s", True
            elif sim.nv <= 1:
                obs, alert = "aún sin simular", False
            else:
                rel = sim.vdT[-1] / sim.T0 if sim.T0 > 0 else 0.0
                obs, alert = f"ΔT/T₀ = {rel:+.1e} · r = {sim.v_er[-1]:.1e}", False
            self._set_badge(k, (c.name, c.color, METHOD_SHORT[c.method], r, regime_color(r), obs, alert))
            if show_detail:
                rows.append([dot(c), METHOD_SHORT[c.method], (f"{c.dt:g}", {"mono": True, "right": True}),
                             num(sim.vt, "{:.2f}"), num(sim.vdT[-1]), num(sim.vdL[-1]),
                             num(sim.v_quat_norm_error()), (f"{sim.v_steps}", {"mono": True, "right": True})])
        for k in range(len(self.sims), len(self.badges)):
            self._set_badge(k, None)
        if show_detail:
            self.cons_table.set_rows(rows)
        if not self.status.currentMessage().startswith(("Previsualización", "Parámetros", "Richardson", "Pausa automática")):
            self.status.showMessage(
                f"{'▶ en ejecución' if self.running else '⏸ en pausa'}   ·   reloj = {self.clock:.2f} s"
                f"   ·   [Espacio] iniciar/pausar  [R] reiniciar  [P] recalcular previsualización"
                f"  [,] [.] paso")

    def _set_badge(self, k, state):
        if state == self._badge_state[k]:
            return
        prev = self._badge_state[k]
        self._badge_state[k] = state
        b = self.badges[k]
        if state is None:
            b.hide()
            return
        name, color, method, pred, col, obs, alert = state
        obs_html = (f"<span style='color:#FFD740'>{obs}</span>" if alert
                    else f"<span style='color:#d3d6db'>{obs}</span>")
        b.setText(f"<span style='color:{color}'>●</span>&nbsp;<b>{name} · {method}</b><br>"
                  f"<span style='font-weight:400; font-size:12px'>predicho: <b>{pred.upper()}</b></span><br>"
                  f"<span style='font-weight:400; font-size:11px'>observado: {obs_html}</span>")
        if prev is None or prev[4] != col:
            b.setStyleSheet(badge_css(col, 13))
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
