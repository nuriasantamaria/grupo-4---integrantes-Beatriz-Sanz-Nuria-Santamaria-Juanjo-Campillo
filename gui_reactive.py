"""
gui_reactive.py — Janibekov Lab
===============================

Dashboard interactivo multipanel para la dinámica de rotación libre de un sólido
rígido (satélite) descrita por las ecuaciones de Euler:

    I1 dω1/dt = (I2 - I3) ω2 ω3
    I2 dω2/dt = (I3 - I1) ω3 ω1
    I3 dω3/dt = (I1 - I2) ω1 ω2

junto con la cinemática de la orientación en cuaterniones, q' = ½ q ⊗ (0, ω).

Muestra el efecto Janibekov (teorema del eje intermedio): la rotación en torno
al eje de momento de inercia intermedio es inestable y el cuerpo "voltea"
periódicamente.

Estructura del archivo
----------------------
1. Física y geometría       – ecuaciones, invariantes, superficies de Poinsot,
                               separatriz y familia de polodias analíticas.
2. Integradores numéricos    – SOLO los esquemas de numerical_engine.py: Euler,
                               Crank-Nicolson y RK4, y la extrapolación de
                               Richardson para estimar orden y error.
3. Modelo de simulación      – SimConfig / Simulation (historial incremental)
                               y previsualización estática.
4. Widgets de interfaz       – FloatSlider, SimConfigPanel, tarjetas.
5. Ventana principal         – paneles 3D (OpenGL), gráficas de deriva,
                               multisimulación, animación y presets.

Dependencias:  pip install PySide6 pyqtgraph PyOpenGL numpy sympy
               + numerical_engine.py y space_physics.py en la misma carpeta
Ejecución:     python gui_reactive.py

Atajos: [Espacio] iniciar/pausar · [R] reiniciar · [P] previsualizar
"""

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
from space_physics import euler_equations, kinetic_energy, angular_momentum_sq


# =============================================================================
# 1. FÍSICA Y GEOMETRÍA
# =============================================================================

# Las ecuaciones de Euler y los invariantes T y L² vienen de space_physics.py
# (euler_equations, kinetic_energy, angular_momentum_sq); aquí solo hay geometría.

def quat_to_matrix(q):
    """Matriz de rotación cuerpo -> inercial del cuaternión q = (w, x, y, z)."""
    q = np.asarray(q, dtype=float)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def sorted_axes(I):
    """Índices (a, b, c) de los ejes de inercia menor, intermedia y mayor."""
    a, b, c = np.argsort(np.asarray(I, dtype=float))
    return int(a), int(b), int(c)


def is_degenerate(I, tol=1e-6):
    """True si dos momentos coinciden (trompo simétrico: no hay separatriz)."""
    Is = np.sort(np.asarray(I, dtype=float))
    return (Is[1] - Is[0] < tol) or (Is[2] - Is[1] < tol)


def poinsot_semiaxes(I, w0):
    """Semiejes, en el espacio ω, del elipsoide de energía (Σ Ii ωi² = 2T) y del
    elipsoide de momento angular (Σ Ii² ωi² = L²)."""
    I = np.asarray(I, dtype=float)
    twoT = 2 * kinetic_energy(w0, I)
    L = np.sqrt(angular_momentum_sq(w0, I))
    return np.sqrt(twoT / I), L / I


def regime(I, w0):
    """Clasifica la órbita según el signo de D = L² - 2T·I_intermedio."""
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
    """Separatriz: intersección del elipsoide de energía con L² = 2T·Ib.

    Restando ambas ecuaciones:  Ia(Ib-Ia) ωa² = Ic(Ic-Ib) ωc²  ->  ωc = ±k ωa,
    y sustituyendo en la energía queda una elipse en el plano (ωa, ωb)."""
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
    """Familia de polodias analíticas sobre el elipsoide de energía 2T.

    Para cada nivel L² se eliminan variables entre los dos invariantes y se
    obtienen relaciones lineales en los cuadrados, que se parametrizan con
    (cos s, sin s). Devuelve curvas (n_pts, 3) alrededor de los ejes estables."""
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    s = np.linspace(0, 2 * np.pi, n_pts)
    curves = []
    levels = np.linspace(0.15, 0.9, n_levels)

    # Alrededor del eje de inercia menor: L² ∈ (2T·Ia, 2T·Ib)
    for f in levels:
        L2 = twoT * (Ia + f * (Ib - Ia))
        wc = math.sqrt((L2 - twoT * Ia) / (Ic * (Ic - Ia))) * np.cos(s)
        wb = math.sqrt((L2 - twoT * Ia) / (Ib * (Ib - Ia))) * np.sin(s)
        wa = np.sqrt(np.maximum(twoT - Ib * wb ** 2 - Ic * wc ** 2, 0) / Ia)
        for sign in (+1, -1):
            P = np.zeros((n_pts, 3))
            P[:, a], P[:, b], P[:, c] = sign * wa, wb, wc
            curves.append(P)

    # Alrededor del eje de inercia mayor: L² ∈ (2T·Ib, 2T·Ic)
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
    """Semilados de un paralelepípedo homogéneo con momentos principales I.

    Para una caja de lados (lx, ly, lz):  I1 = m(ly² + lz²)/12, etc.  Invirtiendo,
    lx² ∝ I2 + I3 - I1 (positivo gracias a la desigualdad triangular).
    Se normaliza para que el semilado mayor valga 1. El elipsoide homogéneo
    equivalente tiene exactamente las mismas proporciones.

    Si I3 = I1 + I2 (p. ej. I = (1, 2, 3)) el cuerpo es una lámina de espesor
    nulo; solo para poder verlo se impone un espesor mínimo de `min_rel`."""
    I1, I2, I3 = (float(x) for x in I)
    s = np.sqrt(np.maximum([I2 + I3 - I1, I1 + I3 - I2, I1 + I2 - I3], 0.0))
    s = s / s.max()
    return np.maximum(s, min_rel)


# =============================================================================
# 2. INTEGRADORES NUMÉRICOS  (todos los esquemas vienen de numerical_engine.py)
# =============================================================================
# Estado u = (ω1, ω2, ω3, q0, q1, q2, q3). Los esquemas del motor son genéricos
# en la dimensión de u0, así que integran a la vez las ecuaciones de Euler y la
# cinemática del cuaternión. El cuaternión no se renormaliza: ‖q‖ - 1 es otra
# medida de la calidad del esquema (para dibujar se normaliza).

def rigid_body_rhs(u, I_mat):
    """Campo vectorial completo: ecuaciones de Euler (space_physics) + q' = ½ q ⊗ (0, ω)."""
    w1, w2, w3, q0, q1, q2, q3 = u
    dw = euler_equations(u[:3], I_mat)
    return np.array([dw[0], dw[1], dw[2],
                     0.5 * (-q1 * w1 - q2 * w2 - q3 * w3),
                     0.5 * (q0 * w1 + q2 * w3 - q3 * w2),
                     0.5 * (q0 * w2 + q3 * w1 - q1 * w3),
                     0.5 * (q0 * w3 + q1 * w2 - q2 * w1)])


def make_rhs(I):
    """f(u) con la firma que esperan los esquemas del motor."""
    return partial(rigid_body_rhs, I_mat=np.diag(np.asarray(I, dtype=float)))


M_EULER = "Euler explícito"
M_CN = "Crank-Nicolson (implícito)"
M_RK4 = "Runge-Kutta 4"

# Tolerancia del punto fijo de Crank-Nicolson: más estricta que la de 1e-6 por
# defecto, para que el error del solver no tape la deriva propia del esquema.
CN_TOL = 1e-12

SCHEMES = {
    M_EULER: ne.euler,
    M_CN: partial(ne.crank_nicolson, tol=CN_TOL),
    M_RK4: ne.runge_kutta_4,
}

# Orden nominal p de cada esquema (para el error de Richardson)
ORDERS = {M_EULER: 1, M_CN: 2, M_RK4: 4}

METHOD_SHORT = {M_EULER: "Euler", M_CN: "C-N", M_RK4: "RK4"}

DIVERGENCE_LIMIT = 1e6


def integrate(method, y0, dt, I, n):
    """Integra n pasos con el esquema del motor. Devuelve un array (n+1, 7); si
    el método diverge, las filas a partir de la divergencia se rellenan con NaN."""
    with np.errstate(all="ignore"):          # Euler puede desbordar a inf/NaN
        out = SCHEMES[method](make_rhs(I), np.asarray(y0, dtype=float), dt, n)
        bad = ~(np.abs(out[:, :3]).sum(axis=1) < DIVERGENCE_LIMIT)   # también atrapa NaN
    if bad.any():
        out[np.argmax(bad):] = np.nan
    return out


def richardson_analysis(cfg, T):
    """Orden estimado y error asintótico del estado final en t = T, usando las
    funciones de extrapolación de Richardson de numerical_engine."""
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


# =============================================================================
# 3. MODELO DE SIMULACIÓN
# =============================================================================

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
    """Una simulación en curso con historial creciente (t, y, ΔT, Δ‖L‖)."""

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
        self.Ls0 = self.I * w0           # q0 = identidad -> L inercial inicial
        self._Y = np.empty((self.CHUNK, 7))
        self._dT = np.empty(self.CHUNK)
        self._dL = np.empty(self.CHUNK)
        self._Y[0] = self.y
        self._dT[0] = 0.0
        self._dL[0] = 0.0

    # --- acceso al historial -------------------------------------------------
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

    # --- avance ----------------------------------------------------------------
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
        """Avanza el número de pasos necesario para alcanzar t_target. Así varias
        simulaciones con distinto Δt avanzan sincronizadas en tiempo físico."""
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
    """Previsualización rápida: trayectoria estática completa calculada de golpe."""
    sim = Simulation(cfg)
    n = min(int(math.ceil(T_total / cfg.dt)), MAX_PREVIEW_STEPS)
    sim.MAX_STEPS_PER_CALL = n
    sim.advance(n)
    return {"t": sim.times.copy(), "W": sim.Y[:, :3].copy(),
            "dT": sim.dT.copy(), "dL": sim.dL.copy(),
            "diverged": sim.diverged, "truncated": n == MAX_PREVIEW_STEPS}


# =============================================================================
# 4. WIDGETS DE INTERFAZ
# =============================================================================

SIM_COLORS = ["#4FC3F7", "#FF8A65", "#AED581"]
AXIS_COLORS = ["#EF5350", "#66BB6A", "#42A5F5"]   # e1, e2, e3
BG = "#0e131f"
PANEL = "#151c2c"
TEXT = "#d5dbe8"
MUTED = "#8a94ab"


def rgba(hex_color, alpha=1.0, factor=1.0):
    c = QtGui.QColor(hex_color)
    return (min(c.redF() * factor, 1.0), min(c.greenF() * factor, 1.0),
            min(c.blueF() * factor, 1.0), alpha)


# --- Geometría del satélite (en ejes cuerpo) ---------------------------------

_BOX_FACES = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],     # -x, +x
                       [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],     # -y, +y
                       [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])    # -z, +z
_BOX_EDGES = np.array([(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
                       (0, 4), (1, 5), (2, 6), (3, 7)]).ravel()
PANEL_COLOR = "#1f3f8f"
PANEL_LINES = (0.62, 0.72, 1.0, 0.75)
METAL_COLOR = "#cfd4de"


def box_items(center, half, color, shades=(1.0, 1.0, 1.0), edge_color=(1, 1, 1, 0.6)):
    """Caja sombreada (una tonalidad por par de caras ±x, ±y, ±z) y sus aristas."""
    V = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    V = center + V * half
    C = np.array([rgba(color, 1.0, shades[f // 4]) for f in range(12)])
    mesh = gl.GLMeshItem(meshdata=gl.MeshData(vertexes=V, faces=_BOX_FACES, faceColors=C),
                         smooth=False, shader="shaded", glOptions="opaque")
    edges = gl.GLLinePlotItem(pos=V[_BOX_EDGES], mode="lines", color=edge_color,
                              width=1.5, antialias=True)
    return mesh, edges


def satellite_items(half, color):
    """Satélite básico: cuerpo central + dos paneles solares + antena parabólica.

    La envolvente respeta las proporciones de inercia: los paneles se extienden
    a lo largo del eje más largo (el de menor momento de inercia), su plano es
    perpendicular al eje más delgado y la antena apunta por ese eje."""
    h = np.asarray(half, dtype=float)
    c, b, a = np.argsort(h)          # a: eje más largo, c: más delgado

    def vec(va, vb, vc):
        v = np.zeros(3)
        v[a], v[b], v[c] = va, vb, vc
        return v

    items = []
    # Cuerpo central (bus), del color de la simulación
    bus = 0.2 + 0.2 * h
    items += box_items(np.zeros(3), bus, color, shades=(0.7, 1.1, 0.9))

    # Paneles solares con su retícula de células y los brazos de unión
    gap, pt = 0.12, 0.015
    Lp = max(1.25 * h[a] - bus[a] - gap, 0.45)
    pw = max(0.75 * h[b], 0.28)
    grid, struts = [], []
    for s in (-1, 1):
        inner = bus[a] + gap
        items += box_items(vec(s * (inner + Lp / 2), 0, 0), vec(Lp / 2, pw, pt),
                           PANEL_COLOR, shades=(0.9, 0.9, 1.2), edge_color=PANEL_LINES)
        for z in (-pt - 0.003, pt + 0.003):          # retícula en ambas caras
            for i in range(1, 4):
                x = s * (inner + i * Lp / 4)
                grid += [vec(x, -pw, z), vec(x, pw, z)]
            grid += [vec(s * inner, 0, z), vec(s * (inner + Lp), 0, z)]
        struts += [vec(s * bus[a], 0, 0), vec(s * inner, 0, 0)]
    items.append(gl.GLLinePlotItem(pos=np.array(grid), mode="lines", color=PANEL_LINES,
                                   width=1, antialias=True))

    # Antena: mástil + plato (cono abierto) apuntando por +eje delgado
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
    """Slider de coma flotante con etiqueta y caja numérica sincronizadas."""

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
    """Controles de una simulación: integrador, inercias, Ω0 y Δt."""

    changed = QtCore.Signal()
    copyRequested = QtCore.Signal(int)     # índice de esta simulación

    def __init__(self, index, cfg: SimConfig, parent=None):
        super().__init__(parent)
        self.index = index
        self.name = cfg.name
        self.color = cfg.color

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 8, 6, 6)
        lay.setSpacing(8)

        # Cabecera: activa + método
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

        # Inercias con restricción triangular
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
        """Impone la desigualdad triangular: Ik ∈ [|Ij - Il|, Ij + Il].
        Al modificar un único momento basta con acotar ese mismo."""
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
    """Panel con título para envolver una vista."""

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


STYLESHEET = f"""
QWidget {{ background: {BG}; color: {TEXT}; font-size: 12px; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QGroupBox {{ background: {PANEL}; border: 1px solid #222b40; border-radius: 8px;
             margin-top: 14px; padding: 10px 8px 8px 8px; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
                    color: {MUTED}; font-weight: 600; }}
QGroupBox QWidget {{ background: transparent; }}
QFrame#card {{ background: {PANEL}; border: 1px solid #222b40; border-radius: 10px; }}
QFrame#card QLabel {{ background: transparent; }}
QLabel#cardTitle {{ font-size: 13px; font-weight: 700; color: #ffffff; }}
QLabel#muted {{ color: {MUTED}; }}
QLabel#title {{ font-size: 20px; font-weight: 800; color: #ffffff; }}
QLabel#warning {{ color: #FFB74D; }}
QPushButton {{ background: #222c44; border: 1px solid #2f3b59; border-radius: 6px;
               padding: 6px 10px; }}
QPushButton:hover {{ background: #2b3756; }}
QPushButton:pressed {{ background: #1b2336; }}
QPushButton#primary {{ background: #1f6feb; border-color: #3b82f6; font-weight: 700; color: white; }}
QPushButton#primary:hover {{ background: #2f7cf6; }}
QPushButton#accent {{ background: #6d28d9; border-color: #8b5cf6; font-weight: 700; color: white; }}
QPushButton#accent:hover {{ background: #7c3aed; }}
QComboBox, QDoubleSpinBox, QSpinBox {{ background: #0f1626; border: 1px solid #2f3b59;
               border-radius: 5px; padding: 3px 6px; }}
QComboBox QAbstractItemView {{ background: #0f1626; selection-background-color: #1f6feb; }}
QSlider::groove:horizontal {{ height: 4px; background: #2a3450; border-radius: 2px; }}
QSlider::sub-page:horizontal {{ background: #3b82f6; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: #e6ecff; width: 14px; margin: -6px 0; border-radius: 7px; }}
QTabWidget::pane {{ border: 1px solid #222b40; border-radius: 8px; background: {BG}; }}
QTabBar::tab {{ background: {PANEL}; padding: 6px 14px; border-top-left-radius: 6px;
                border-top-right-radius: 6px; margin-right: 2px; color: {MUTED}; }}
QTabBar::tab:selected {{ background: #222c44; color: white; }}
QSplitter::handle {{ background: {BG}; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; }}
"""


# =============================================================================
# 5. VENTANA PRINCIPAL
# =============================================================================

DEFAULT_CONFIGS = [
    SimConfig("A", SIM_COLORS[0], True, M_EULER),
    SimConfig("B", SIM_COLORS[1], True, M_CN),
    SimConfig("C", SIM_COLORS[2], True, M_RK4),
]

# Cada preset: lista de 3 diccionarios con los campos de SimConfig a modificar.
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
}
DEFAULT_PRESET = "Comparar integradores (mismo Δt)"


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

        self._build_ui()
        self._build_phase_items()
        self._build_plot_items()

        # Debounce: los sliders emiten muchas señales; se aplican tras 120 ms
        self._param_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._param_timer.timeout.connect(self.apply_params)

        self.apply_preset(DEFAULT_PRESET)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.FRAME_MS)

        for key, slot in (("Space", self.toggle_run), ("R", self.reset),
                          ("P", self.compute_preview)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)

    # ------------------------------------------------------------------ UI ---
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

        # Vistas 3D
        self.phase_view = gl.GLViewWidget()
        self.phase_view.setBackgroundColor(BG)
        self.sat_view = gl.GLViewWidget()
        self.sat_view.setBackgroundColor(BG)
        self.phase_card = Card("Espacio de fases (ω₁, ω₂, ω₃) — Poinsot", self.phase_view,
                               "elipsoide T · elipsoide ‖L‖ · separatriz")
        self.sat_card = Card("Orientación del satélite (marco inercial)", self.sat_view,
                             "ejes cuerpo e₁ e₂ e₃ · L (amarillo) · ω (blanco)")
        center = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        center.addWidget(self.phase_card)
        center.addWidget(self.sat_card)
        center.setSizes([560, 420])

        # Gráficas 2D
        self.plot_T = pg.PlotWidget()
        self.plot_L = pg.PlotWidget()
        self.plot_w = pg.PlotWidget()
        for p, ylab in ((self.plot_T, "ΔT"), (self.plot_L, "Δ‖L‖"), (self.plot_w, "ω")):
            p.showGrid(x=True, y=True, alpha=0.15)
            p.setLabel("bottom", "t")
            p.setLabel("left", ylab)
            p.getPlotItem().setDownsampling(auto=True, mode="peak")
            p.getPlotItem().setClipToView(True)
            p.addLegend(offset=(8, 8), labelTextSize="8pt")
            p.getAxis("left").enableAutoSIPrefix(False)
        self.card_T = Card("Deriva de energía  ΔT(t) = T(t) − T₀", self.plot_T)
        self.card_L = Card("Deriva del momento  Δ‖L‖(t) = ‖L(t)‖ − ‖L₀‖", self.plot_L)
        self.card_w = Card("Componentes ω(t) — simulación de referencia", self.plot_w)
        right = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        for c in (self.card_T, self.card_L, self.card_w):
            right.addWidget(c)

        main_split = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        main_split.addWidget(center)
        main_split.addWidget(right)
        main_split.setSizes([720, 560])
        root.addWidget(main_split, 1)

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

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

        # --- Transporte -------------------------------------------------------
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
        self.chk_reactive.setChecked(False)   # por defecto se previsualiza con el botón
        self.chk_show_prev = QtWidgets.QCheckBox("Mostrar trayectoria previsualizada")
        self.chk_show_prev.setChecked(True)
        self.chk_show_prev.toggled.connect(self._update_preview_items)
        g.addWidget(self.chk_reactive, 4, 0, 1, 2)
        g.addWidget(self.chk_show_prev, 5, 0, 1, 2)
        lay.addWidget(box)

        # --- Presets ------------------------------------------------------------
        box = QtWidgets.QGroupBox("Escenarios predefinidos")
        h = QtWidgets.QHBoxLayout(box)
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(PRESETS))
        btn_apply = QtWidgets.QPushButton("Aplicar")
        btn_apply.clicked.connect(lambda: self.apply_preset(self.preset_combo.currentText()))
        h.addWidget(self.preset_combo, 1)
        h.addWidget(btn_apply)
        lay.addWidget(box)

        # --- Simulaciones A/B/C --------------------------------------------------
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

        # --- Visualización -----------------------------------------------------
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

        # --- Richardson ---------------------------------------------------------
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

        # --- Estado --------------------------------------------------------------
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

    # ------------------------------------------------- elementos de escena ---
    def _build_phase_items(self):
        v = self.phase_view
        # Ejes ω1, ω2, ω3
        self.phase_axes = []
        self.phase_axis_labels = []
        for k in range(3):
            line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(AXIS_COLORS[k], 0.8),
                                     width=1.5, antialias=True, mode="lines")
            v.addItem(line)
            lbl = gl.GLTextItem(text=f"ω{'₁₂₃'[k]}", color=QtGui.QColor(AXIS_COLORS[k]))
            v.addItem(lbl)
            self.phase_axes.append(line)
            self.phase_axis_labels.append(lbl)

        # Superficies de Poinsot (se rellenan en _update_static_phase)
        sphere = gl.MeshData.sphere(rows=36, cols=72)
        self.mesh_E = gl.GLMeshItem(meshdata=sphere, smooth=True, color=(0.25, 0.85, 0.55, 0.10),
                                    shader="balloon", glOptions="additive")
        self.mesh_L = gl.GLMeshItem(meshdata=sphere, smooth=True, color=(0.95, 0.35, 0.75, 0.10),
                                    shader="balloon", glOptions="additive")
        v.addItem(self.mesh_E)
        v.addItem(self.mesh_L)

        # Trayectorias por simulación: previsualización, estela viva y cabeza
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

        # Vista del satélite: rejilla de suelo
        grid = gl.GLGridItem()
        grid.setSize(20, 10)
        grid.setSpacing(1, 1)
        grid.translate(0, 0, -2.0)
        grid.setColor((80, 95, 130, 90))
        self.sat_view.addItem(grid)
        self.sat_view.setCameraPosition(distance=9, elevation=18, azimuth=-60)

    def _build_plot_items(self):
        self.curves_T, self.curves_L = [], []
        self.prev_T, self.prev_L = [], []
        for cfg in DEFAULT_CONFIGS:
            for plot, live, prev in ((self.plot_T, self.curves_T, self.prev_T),
                                     (self.plot_L, self.curves_L, self.prev_L)):
                prev.append(plot.plot(pen=pg.mkPen(QtGui.QColor(cfg.color).darker(170), width=1,
                                                   style=QtCore.Qt.PenStyle.DashLine)))
                live.append(plot.plot(pen=pg.mkPen(cfg.color, width=2), name=cfg.name))
        self.curves_w, self.prev_w = [], []
        for k in range(3):
            self.prev_w.append(self.plot_w.plot(pen=pg.mkPen(QtGui.QColor(AXIS_COLORS[k]).darker(180),
                                                             width=1, style=QtCore.Qt.PenStyle.DashLine)))
            self.curves_w.append(self.plot_w.plot(pen=pg.mkPen(AXIS_COLORS[k], width=2),
                                                  name=f"ω{k + 1}"))

    # ------------------------------------------------------- parámetros ---
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
        """Reconstruye todas las simulaciones con los parámetros actuales."""
        self._param_timer.stop()
        cfgs = self.configs()
        # Si la simulación de referencia está inactiva, usar la primera activa
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
        self._rebuild_satellites()
        self.plot_w.setTitle(None)
        if self.chk_reactive.isChecked():
            self.compute_preview()
        else:
            # La previsualización anterior ya no corresponde a los parámetros
            self._update_preview_items()
            self.status.showMessage(
                "Parámetros aplicados · pulsa ⚡ Previsualización rápida [P] para ver la trayectoria completa",
                6000)
        self._refresh_views()

    def _on_horizon_changed(self):
        """Cambiar T solo afecta a la previsualización, no reinicia la animación."""
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

    # --------------------------------------------------- previsualización ---
    def run_richardson(self):
        """Orden y error asintótico de cada simulación activa (numerical_engine)."""
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
                    body = "<span style='color:#ef5350'>no estimable (divergencia)</span>"
                rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                            f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g}<br>"
                            f"<span style='font-family:Consolas,monospace'>{body}</span>")
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

    # ------------------------------------------------ geometría estática ---
    def _on_ref_changed(self):
        self._update_static_phase()
        self._update_preview_items()
        self._refresh_views()

    def _update_static_phase(self):
        """Superficies de Poinsot, separatriz y polodias de la simulación de referencia."""
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
            f"ref. {cfg.name}:  2T = {twoT:.4g}   ‖L‖ = {np.sqrt(angular_momentum_sq(w0, I)):.4g}   ·   "
            f"{regime(I, w0)}")
        self._update_static_visibility()

    def _update_static_visibility(self):
        self.mesh_E.setVisible(self.chk_E.isChecked())
        self.mesh_L.setVisible(self.chk_L.isChecked())
        for it in self.separatrix_items:
            it.setVisible(self.chk_sep.isChecked())
        for it in self.family_items:
            it.setVisible(self.chk_fam.isChecked())

    # ---------------------------------------------------------- satélites ---
    def _rebuild_satellites(self):
        """Crea un satélite por simulación activa, colocados lado a lado."""
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
                                  color=QtGui.QColor(cfg.color))
            items = parts + axes + [Lvec, wvec, label]
            for it in items:
                self.sat_view.addItem(it)
            self.sat_items.append(dict(k=k, offset=offset, extent=extent, parts=parts,
                                       axes=axes, L=Lvec, w=wvec, all=items))
        self.sat_view.setCameraPosition(distance=4 + 2.6 * len(cfgs))
        self._update_satellites()

    def _make_body(self, half, color):
        """Piezas del cuerpo en ejes cuerpo (todas comparten la misma transformación).
        Devuelve (lista de items, semiextensión del conjunto en cada eje)."""
        shape = self.shape_combo.currentText()
        if shape == "Elipsoide":
            md = gl.MeshData.sphere(rows=24, cols=48)
            md.setVertexes(md.vertexes() * half)
            body = gl.GLMeshItem(meshdata=md, smooth=True, color=rgba(color, 1.0),
                                 shader="shaded", glOptions="opaque")
            return [body], half
        if shape == "Paralelepípedo":
            shades = [0.55, 1.15, 0.85]                     # caras ±e2 más claras
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
            # Momento angular en el marco inercial: constante en la física exacta
            Ls = R @ (sim.I * y[:3])
            Lref = max(np.linalg.norm(sim.Ls0), 1e-12)
            g["L"].setData(pos=np.array([o, o + 1.6 * Ls / Lref]))
            ws = R @ y[:3]
            wref = max(np.linalg.norm(sim.cfg.w0), 1e-12)
            g["w"].setData(pos=np.array([o, o + 1.25 * ws / wref]))

    # ------------------------------------------------------ bucle principal ---
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
                       else ("ΔT" if p is self.plot_T else "Δ‖L‖"))
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
                continue
            state = ("<span style='color:#ef5350'>DIVERGIÓ</span>" if sim.diverged
                     else regime(c.I, c.w0))
            lbl.setText(
                f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g} · t={sim.t:.2f} s<br>"
                f"<span style='font-family:Consolas,monospace'>"
                f"ΔT={sim.dT[-1]:+.3e} &nbsp; Δ‖L‖={sim.dL[-1]:+.3e}<br>"
                f"‖q‖−1={sim.quat_norm_error():+.3e} &nbsp; pasos={sim.n_steps}</span><br>"
                f"<span style='color:{MUTED}'>{state}</span>")
        # No pisar los mensajes temporales
        if not self.status.currentMessage().startswith(("Previsualización", "Parámetros", "Richardson")):
            self.status.showMessage(
                f"{'▶ en ejecución' if self.running else '⏸ en pausa'}   ·   reloj = {self.clock:.2f} s"
                f"   ·   [Espacio] iniciar/pausar  [R] reiniciar  [P] previsualizar")


def main():
    pg.setConfigOptions(antialias=True, background=PANEL, foreground=TEXT)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()





