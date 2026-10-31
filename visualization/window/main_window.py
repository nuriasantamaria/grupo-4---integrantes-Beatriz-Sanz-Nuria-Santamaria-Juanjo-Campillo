"""Ventana principal: compone controles, gráficas, vista 3D y análisis"""

from __future__ import annotations

import html
import math
from dataclasses import replace
from functools import partial
from typing import Any, Callable

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg
import pyqtgraph.opengl as gl

import numerical_engine as ne
from simulation import (convergence_report, linear_errors, MAX_LIVE_STEPS, MAX_PREVIEW_STEPS,
                        richardson_report, SimConfig, Simulation)
from space_physics import (angular_momentum_sq, Array, ArrayLike, equilibrium_amplification,
                           equilibrium_analysis, EulerExact, kinetic_energy, LinearErrors, noise_floor,
                           predicted_errors, relevant_eigenvalues)

from visualization import theme
from visualization.cells import DASH, dot, num_or_dash
from visualization.controls import FloatSlider, group_box, PhysicsSliders, SimConfigPanel
from visualization.geometry import (attitude_vectors, body_half_sizes, chart_axes, chart_domain,
                                    chart_reference_orbit, default_chart, exact_orbit, exact_sample,
                                    phase_plane_orbits, poinsot_semiaxes, polhode_family, project_to_chart,
                                    reduced_flow_field, separatrix_curves)
from visualization.help_text import HELP_HTML
from visualization.initial_conditions import same_problem, surface_mismatch
from visualization.palette import AMBER, AXIS_COLORS, GRADE_COLORS, MUTED, STABILITY_COLORS, TEXT, VIEW_BG
from visualization.plot_tools import arrow_segments, log_safe, mask_below, stability_window
from visualization.scenarios import DEFAULT_CONFIGS, DEFAULT_SCENARIO, SCENARIOS
from visualization.schemes import METHOD_SHORT, SCHEME_COLORS, SCHEME_ORDER, stability_layer
from visualization.skins.cow import cow_items
from visualization.skins.fighter import fighter_items
from visualization.skins.meshes import tube
from visualization.skins.satellite import box_items, satellite_items
from visualization.skins.ufo import ufo_items
from visualization.theme import axis_label_style, badge_css, pill, rgba
from visualization.verdicts import (amplitude_trend, COARSE_BETA_DT, grade_pair, GRADE_WORDS,
                                    orbit_stability, scheme_verdict, step_resolution, TINY_PSI, TINY_R)
from visualization.widgets import (AdaptiveSplitter, Card, control_row, DataTable, FlowLayout, LinkedCursor,
                                   muted_label, noise_floor_line, num, plain_body, scroll_area)
from visualization.workers import AnalysisWorker, PreviewWorker


class MainWindow(QtWidgets.QMainWindow):
    """Ventana principal con los controles, las vistas y el bucle de animación"""
    FRAME_MS = 30

    def __init__(self) -> None:
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
        self._workers: list[QtCore.QThread] = []          # hilos vivos (previsualización y análisis)
        self._preview_token = 0
        self._analysis: dict[str, AnalysisWorker] = {}    # análisis en curso: rich (Richardson) y conv
        self._rich_T = 0.0
        self.view_t = None
        self._div_notified = set()
        self._sat_key = None
        self._last_cfgs = None
        self._lin_cache = {}
        self._w_range_key = None

        self._build_ui()
        self._build_phase_items()
        self._build_plot_items()

        self._param_timer = QtCore.QTimer(self, singleShot=True, interval=120)
        self._param_timer.timeout.connect(self.apply_params)

        self.apply_preset(DEFAULT_SCENARIO)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(self.FRAME_MS)

        for key, slot in (("Space", self.toggle_run), ("R", self.reset),
                          ("P", self.compute_preview)):
            QtGui.QShortcut(QtGui.QKeySequence(key), self, activated=slot)
        QtGui.QShortcut(QtGui.QKeySequence("F1"), self, activated=self.show_help)
        QtGui.QShortcut(QtGui.QKeySequence("."), self, activated=self.step_forward)
        QtGui.QShortcut(QtGui.QKeySequence(","), self, activated=self.step_back)
        for k in range(self.view_tabs.count()):
            QtGui.QShortcut(QtGui.QKeySequence(f"Ctrl+{k + 1}"), self,
                            activated=partial(self.view_tabs.setCurrentIndex, k))

    def _build_ui(self) -> None:
        """Construye la ventana: controles a la izquierda y pestañas de vistas a la derecha"""
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QHBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        root = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.setChildrenCollapsible(False)
        root.setHandleWidth(10)
        outer.addWidget(root)

        scroll = scroll_area(self._build_controls(), frame=True)
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

        self.dyn_tab = self._build_dynamics_tab()

        self.drift_tab = self._build_drift_tab()

        self.view_tabs = self._build_view_tabs()
        main.addWidget(self.view_tabs, 1)
        main.addLayout(self._build_timeline())
        root.addWidget(main_w)
        root.setStretchFactor(0, 0)
        root.setStretchFactor(1, 1)
        root.setSizes([380, 1300])

        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)

    def _build_dynamics_tab(self) -> QtWidgets.QWidget:
        """Pestaña Dinámica: órbita en el espacio de fases y orientación del cuerpo"""
        self.phase_view = gl.GLViewWidget()
        self.phase_view.setBackgroundColor(VIEW_BG)
        self.sat_view = gl.GLViewWidget()
        self.sat_view.setBackgroundColor(VIEW_BG)
        self.phase_view.setMinimumSize(200, 200)
        self.sat_view.setMinimumSize(200, 200)

        def surface_check(text: str, color: str, tip: str) -> QtWidgets.QCheckBox:
            c = QtWidgets.QCheckBox(text)
            c.setChecked(True)
            c.setToolTip(tip)
            c.setStyleSheet(f"QCheckBox {{ color: {color}; font-weight: 600; }}")
            c.toggled.connect(self._update_static_visibility)
            return c

        self.chk_E = surface_check("Energía", "#40D98C", "Elipsoide de energía (2T constante)")
        self.chk_L = surface_check("Momento", "#F259BF", "Elipsoide de momento angular (‖L‖ constante)")
        self.chk_sep = surface_check("Separatriz", "#FFFFFF", "Curvas que separan los dos tipos de órbita")
        self.chk_fam = surface_check("Polhodias", "#8C9EC7", "Órbitas posibles con esta energía y este momento")
        self.chk_exact = surface_check("Exacta", AMBER,
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
        return self.dyn_tab

    def _build_drift_tab(self) -> QtWidgets.QWidget:
        """Pestaña Derivas: deriva de energía, deriva del momento y velocidad angular de la referencia"""
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
        return self.drift_tab

    def _build_view_tabs(self) -> QtWidgets.QTabWidget:
        """Las cinco pestañas con el reloj y el selector de referencia en la esquina"""
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
        return self.view_tabs

    def _on_tab_changed(self, _: int | None = None) -> None:
        self._refresh_views()
        self._update_status()

    @staticmethod
    def _style_plot(p: pg.PlotItem, xlab: str, ylab: str) -> None:
        """Aplica a una gráfica la rejilla, las etiquetas y las fuentes comunes"""
        p.showGrid(x=True, y=True, alpha=0.18)
        p.setLabel("bottom", xlab, **axis_label_style())
        p.setLabel("left", ylab, **axis_label_style())
        tick_font = QtGui.QFont(theme.UI_FONT, 9)
        for ax in ("left", "bottom"):
            p.getAxis(ax).setTickFont(tick_font)
            p.getAxis(ax).setTextPen(TEXT)
        p.getAxis("left").enableAutoSIPrefix(False)

    def _build_banner(self) -> QtWidgets.QHBoxLayout:
        """Construye la fila de tarjetas de estado, una por simulación"""
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(8)
        title = QtWidgets.QLabel("ESTADO")
        title.setObjectName("bannerTitle")
        title.setToolTip("Una tarjeta por simulación, con tres cosas distintas:\n"
                         "· Órbita: estabilidad física del movimiento (solo depende de I y Ω₀). Da el color del borde.\n"
                         "· Esquema: qué prevé la teoría lineal del método y el Δt para el módulo y la fase.\n"
                         "· Observado: el error medido contra la solución exacta; en ámbar si es relevante.")
        row.addWidget(title)
        self.badges, self._badge_state = [], [None] * len(DEFAULT_CONFIGS)
        for _ in DEFAULT_CONFIGS:
            b = QtWidgets.QLabel()
            b.setTextFormat(QtCore.Qt.TextFormat.RichText)
            b.setWordWrap(True)
            b.setMinimumWidth(150)
            b.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Preferred)
            b.hide()
            row.addWidget(b, 1)
            self.badges.append(b)
        return row

    def _build_error_tab(self) -> QtWidgets.QWidget:
        """Pestaña Error vs exacta: comparación con la solución exacta y errores de módulo y fase"""
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
        self.err_floor = [noise_floor_line(p) for p in (self.plot_er, self.plot_ep)]   # bajo ella: ruido de redondeo

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
        info = Card("Lectura del error", scroll_area(self.err_info, transparent=True))
        info.setMinimumWidth(250)

        w = AdaptiveSplitter(640, sizes_h=(900, 330), sizes_v=(700, 280))
        w.addWidget(left)
        w.addWidget(info)
        return w

    HOVER_HINT = "Pasa el ratón por las gráficas para leer los valores en un instante."

    def _sims_at(self, t: float) -> list[Any]:
        """Pares simulación e índice de fila de cada simulación activa en el instante t"""
        out = []
        for sim in self.sims:
            if sim.cfg.enabled and sim.nv > 1:
                out.append((sim, min(sim.index_at(t), sim.nv - 1)))
        return out

    def _on_drift_cursor(self, t: float) -> None:
        """Muestra en la pestaña Derivas los valores bajo el cursor"""
        if t != t:
            self.drift_readout.setText(f"<span style='color:{MUTED}'>{self.HOVER_HINT}</span>")
            return
        parts = [f"<span style='color:{s.cfg.color}'>●</span> {s.cfg.name}: ΔT {s.vdT[i]:+.2e} · "
                 f"Δ‖L‖ {s.vdL[i]:+.2e}" for s, i in self._sims_at(t)]
        self.drift_readout.setText(f"<span style='font-family:{theme.MONO}'>t = {max(t, 0):.2f} s</span> &nbsp; "
                                   + " &nbsp; ".join(parts))

    def _on_err_cursor(self, t: float) -> None:
        """Muestra en la pestaña de error los valores bajo el cursor"""
        if t != t:
            self.err_readout.setText(f"<span style='color:{MUTED}'>{self.HOVER_HINT}</span>")
            return
        def mark(v: float, floor: float) -> str:
            return " ≈0" if abs(v) <= floor else ""
        parts = []
        for s, i in self._sims_at(t):
            fl = noise_floor(i)
            parts.append(f"<span style='color:{s.cfg.color}'>●</span> {s.cfg.name}: "
                         f"r {s.v_er[i]:+.2e}{mark(s.v_er[i], fl)} · "
                         f"ψ {s.v_ep[i]:+.3g}°{mark(s.v_ep[i], math.degrees(fl))}")
        self.err_readout.setText(f"<span style='font-family:{theme.MONO}'>t = {max(t, 0):.2f} s</span> &nbsp; "
                                 + " &nbsp; ".join(parts))

    def _on_error_options(self, *_) -> None:
        self._apply_error_axes()
        self._err_prev_stale = True
        self._ew_cache = None
        self._ew_fit = True
        self._refresh_views()

    def _apply_error_axes(self) -> None:
        """Ajusta ejes, etiquetas y escala log de las gráficas de error según las opciones"""
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
        self._update_error_floor()

    def _update_error_floor(self) -> None:
        """Coloca la línea del ruido de redondeo en las gráficas de error"""
        log = self.chk_log_err.isChecked()
        n = max([s.v_steps for s in self.sims if s.cfg.enabled] + [1])
        fl = noise_floor(n)
        seconds = self.err_unit.currentIndex() == 1
        for ln, v, shown in ((self.err_floor[0], fl, True), (self.err_floor[1], math.degrees(fl), not seconds)):
            ln.setVisible(log and shown)
            if log and shown:
                ln.setPos(math.log10(v))

    @staticmethod
    def _fmt_err(d: Array, log: bool) -> Array:
        """En escala log omite los puntos por debajo del ruido de redondeo"""
        return mask_below(d, 1e-16) if log else d

    @staticmethod
    def _same_exact(a: SimConfig, b: SimConfig) -> bool:
        return same_problem(a.I, a.w0, b.I, b.w0)

    def _update_error_notice(self) -> None:
        """Avisa si alguna simulación tiene otra física o un paso demasiado grande"""
        cfgs = self.configs()
        ref = cfgs[self.ref_index()]
        diff = [c.name for k, c in enumerate(cfgs)
                if c.enabled and k != self.ref_index() and not self._same_exact(ref, c)]
        coarse = []
        for c in cfgs:
            lin = self._lin(c) if c.enabled else None
            res = step_resolution(lin, c.dt) if lin is not None else None
            if res is not None and res["coarse"]:
                coarse.append(f"<b>{c.name}</b> ({res['steps_per_cycle']:.0f} pasos por ciclo"
                              + (", |R| > 1: explota" if res["explodes"] else "") + ")")
        if diff or coarse:
            msgs = []
            if diff:
                msgs.append(f"<b>{' y '.join(diff)}</b> {'usa' if len(diff) == 1 else 'usan'} otra I u otro Ω₀: "
                            f"en la gráfica de arriba su solución exacta propia va punteada con su color; "
                            f"los errores siempre se miden frente a la exacta de cada simulación.")
            if coarse:
                msgs.append("⚠ Δt demasiado grande para la dinámica (Δt·β > "
                            f"{COARSE_BETA_DT:g}): {', '.join(coarse)}. Los errores serán grandes; "
                            "reduce Δt o cambia de esquema.")
            self.err_notice.setText("<br>".join(msgs))
            self.err_notice.show()
        else:
            self.err_notice.hide()

    @staticmethod
    def _exact_curve(exact: EulerExact, t_end: float, comp: int, live: bool) -> tuple[Array, Array]:
        """Componente de la solución exacta en una malla propia, independiente del paso"""
        t, W = exact_sample(exact, t_end, 8000 if live else 40000)
        return t, W[:, comp]

    def _build_plane_tab(self) -> QtWidgets.QWidget:
        """Pestaña Plano de fases: carta de fases con su panel de estabilidad del equilibrio"""
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
        iv.addWidget(scroll_area(self.plane_info, transparent=True), 1)

        w = AdaptiveSplitter(900, sizes_h=(720, 330), sizes_v=(620, 300))
        w.addWidget(self.plane_card)
        w.addWidget(info)
        return w

    def _build_analysis_tab(self) -> QtWidgets.QWidget:
        """Pestaña Análisis: conservación, Richardson y convergencia a la izquierda; equilibrios y estabilidad a la derecha"""
        cons_card, rich_card, conv_card = self._conservation_card(), self._richardson_card(), self._convergence_card()
        eq_card, stab_card = self._equilibria_card(), self._stability_card()
        right_w = QtWidgets.QWidget()
        right = QtWidgets.QVBoxLayout(right_w)
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        right.addWidget(eq_card)
        right.addWidget(stab_card)
        right.addStretch(1)

        left_w = QtWidgets.QWidget()
        left = QtWidgets.QVBoxLayout(left_w)
        left.setContentsMargins(0, 0, 0, 0)
        left.setSpacing(10)
        left.addWidget(cons_card)
        left.addWidget(rich_card)
        left.addWidget(conv_card)
        left.addStretch(1)
        w = AdaptiveSplitter(900, sizes_h=(620, 480))
        w.addWidget(scroll_area(left_w))
        w.addWidget(scroll_area(right_w))
        return w

    def _conservation_card(self) -> Card:
        """Tabla de conservación de las simulaciones, en vivo"""
        self.cons_table = DataTable(["", "Esquema", "Δt", "ΔT", "Δ‖L‖", "pasos"])
        cons_card = Card("Conservación", plain_body(
            self.cons_table,
            muted_label("ΔT y Δ‖L‖: deriva de la energía cinética y del módulo del momento angular respecto a sus "
                        "valores iniciales (la solución exacta los conserva).")), "en vivo")
        return cons_card

    def _richardson_card(self) -> Card:
        """Orden y error de cada esquema por Richardson (se calcula en un hilo)"""
        self.rich_T = QtWidgets.QDoubleSpinBox()
        self.rich_T.setRange(0.1, 200)
        self.rich_T.setValue(10)
        self.rich_T.setSuffix(" s")
        self.btn_rich = QtWidgets.QPushButton("Calcular")
        self.btn_rich.setToolTip("Integra con n y 2n pasos hasta t_f (n = t_f/Δt) para cada simulación activa")
        self.btn_rich.clicked.connect(self.run_richardson)
        ctl = control_row(QtWidgets.QLabel("Instante final t_f"), self.rich_T, self.btn_rich)
        self.rich_table = DataTable(["", "Esquema", "Δt", "p", "error estimado", "error real"])
        self._rich_note_idle = ("Pulsa «Calcular». Compara ω(t_f) con n y 2n pasos: su diferencia, dividida por 2ᵖ − 1 "
                                "(p = orden teórico del esquema), estima el error de ω (no el de la energía) en la "
                                "solución con 2n pasos. Real: ‖ω − ω exacta‖ de esa solución en t_f (solución de Jacobi). "
                                "El orden observado se mide en «Convergencia frente a Δt».")
        self._rich_note_done = (self._rich_note_idle.replace("Pulsa «Calcular». Compara", "t_f = {T:g} s. Compara")
                                + " En gris: diferencia entre mallas bajo el ruido de redondeo (error estimado no fiable).")
        self.rich_note = muted_label(self._rich_note_idle)
        self.rich_table.hide()
        rich_card = Card("Orden y error del método", plain_body(ctl, self.rich_table, self.rich_note))
        return rich_card

    def _convergence_card(self) -> Card:
        """Tarjeta de convergencia: error final frente al paso, con la pendiente como orden observado"""
        self.conv_levels = QtWidgets.QSpinBox()
        self.conv_levels.setRange(3, 7)
        self.conv_levels.setValue(5)
        self.conv_levels.setToolTip("Δt, Δt/2, Δt/4… hasta t_f (el de arriba): más niveles, más pasos")
        self.btn_conv = QtWidgets.QPushButton("Calcular")
        self.btn_conv.setToolTip("Error en t_f con Δt, Δt/2, Δt/4… para cada simulación activa, y pendiente en "
                                 "log-log. Con esquemas implícitos puede tardar unos segundos.")
        self.btn_conv.clicked.connect(self.run_convergence)
        cctl = control_row(QtWidgets.QLabel("Niveles"), self.conv_levels, self.btn_conv)
        self.plot_conv = pg.PlotWidget()
        self._style_plot(self.plot_conv, "Δt", "error en t_f")
        self.plot_conv.getAxis("bottom").enableAutoSIPrefix(False)
        self.plot_conv.setLogMode(True, True)
        self.plot_conv.setMinimumHeight(190)
        self.plot_conv.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.conv_floor = noise_floor_line(self.plot_conv)
        self.conv_lines, self.conv_bad, self.conv_ref = [], [], []
        for k, cfg in enumerate(DEFAULT_CONFIGS):
            self.conv_lines.append(self.plot_conv.plot(
                pen=self._live_pen(k, cfg.color), symbol="o", symbolSize=7, symbolBrush=cfg.color,
                symbolPen=cfg.color, name=cfg.name, connect="finite"))
            self.conv_bad.append(self.plot_conv.plot(           # puntos no fiables: huecos
                pen=None, symbol="o", symbolSize=7, symbolBrush=None, symbolPen=pg.mkPen(cfg.color, width=1.2)))
            self.conv_ref.append(self.plot_conv.plot(
                pen=pg.mkPen(QtGui.QColor(cfg.color).lighter(130), width=1, style=QtCore.Qt.PenStyle.DotLine)))
        self.conv_table = DataTable(["", "Esquema", "p nominal", "p Richardson", "p observada", "pendientes locales"])
        self.conv_table.hide()
        self._conv_note_idle = ("Pulsa «Calcular». Se integra hasta t_f con Δt, Δt/2, Δt/4… y se compara con la "
                                "solución exacta: la pendiente en log-log es el orden p. Los puntos huecos no "
                                "se usan en el ajuste: por debajo del ruido de redondeo (línea discontinua) o "
                                "saturados (error comparable a ‖ω‖). La línea de puntos es la pendiente nominal.")
        self.conv_note = muted_label(self._conv_note_idle)
        conv_card = Card("Convergencia frente a Δt", plain_body(cctl, self.plot_conv, self.conv_table, self.conv_note),
                         "complementa a Richardson")
        return conv_card

    def _equilibria_card(self) -> Card:
        """Equilibrios de la simulación de referencia y efecto de cada esquema sobre ellos"""
        self.eq_table = DataTable(["Eje", "Ω", "Tipo (lineal)", "λ", "No lineal"])
        self.eff_table = DataTable(["", "Esquema", "Rodea", "|R| − 1", "Efecto"])
        eq_card = Card("Equilibrios", plain_body(
            muted_label("Rotaciones puras ω = ±Ω eₖ de la simulación de referencia (Iₖ Ω² = 2T)."),
            self.eq_table,
            muted_label("«Lineal»: lo que dice la Jacobiana (un centro es solo marginal). «No lineal»: con T y ‖L‖² "
                 "los ejes de mayor y menor inercia son estables de verdad y el intermedio es inestable."),
            muted_label("Efecto de cada esquema sobre el centro que rodea su órbita: |R(iβΔt)| − 1 "
                 "mide cuánto crece (+) o decae (−) la amplitud en cada paso."),
            self.eff_table), "ref.")
        self.eq_card = eq_card
        return eq_card

    def _stability_card(self) -> Card:
        """Tarjeta de regiones de estabilidad absoluta y posición de cada simulación en ellas"""
        self.stab_shade = QtWidgets.QComboBox()
        self.stab_shade.addItems([METHOD_SHORT[m] for m in ne.SCHEMES])
        self.stab_shade.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.stab_shade.setMinimumContentsLength(10)
        self.stab_zoom = QtWidgets.QCheckBox("Acercar a los puntos")
        self.stab_zoom.setToolTip("Encuadra los z = λΔt de las simulaciones (con Δt pequeño quedan junto al origen)")
        sctl = control_row(QtWidgets.QLabel("Sombrear"), self.stab_shade, self.stab_zoom)
        self.plot_stab = pg.PlotWidget()
        self._style_plot(self.plot_stab, "Re z", "Im z")
        self.plot_stab.setAspectLocked(True)
        self.plot_stab.setMinimumHeight(300)
        self.plot_stab.addLegend(offset=(8, 8), labelTextSize="9pt")
        self.stab_fill = pg.ImageItem()
        self.plot_stab.addItem(self.stab_fill)
        for ang in (0, 90):
            self.plot_stab.addItem(pg.InfiniteLine(angle=ang, pen=pg.mkPen((255, 255, 255, 70), width=1)),
                                   ignoreBounds=True)
        self.stab_curves = {m: self.plot_stab.plot(pen=pg.mkPen(SCHEME_COLORS[m], width=1.8),
                                                   name=METHOD_SHORT[m], connect="finite") for m in ne.SCHEMES}
        self.stab_points = pg.ScatterPlotItem(size=10)
        self.plot_stab.addItem(self.stab_points)
        self.stab_shade.currentIndexChanged.connect(lambda _: self._update_stability())
        self.stab_zoom.toggled.connect(lambda _: self._update_stability())
        stab_card = Card("Regiones de estabilidad absoluta", plain_body(
            sctl, self.plot_stab,
            muted_label("Contorno |R(z)| = 1 de cada esquema; la zona sombreada es su región |R| ≤ 1. Los puntos son "
                 "z = λΔt de cada simulación (λ: autovalores del equilibrio que rodea su órbita, o del eje "
                 "intermedio si está en la separatriz). Dentro de la región el método numérico no hace crecer la "
                 "amplitud por sí mismo: es una propiedad del esquema y de Δt, no de la estabilidad del problema "
                 "físico (el eje intermedio es inestable aunque z caiga dentro; un buen esquema debe reproducirlo).")),
            "|R(z)| ≤ 1")
        return stab_card

    def _build_header(self) -> QtWidgets.QWidget:
        """Construye la cabecera fija con título y controles de ejecución"""
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
        btn_help = QtWidgets.QPushButton("?  Ayuda y atajos")
        btn_help.setToolTip("Atajos de teclado y cómo leer los indicadores  [F1]")
        btn_help.clicked.connect(self.show_help)
        g.addWidget(btn_help, 3, 0, 1, 2)
        lay.addWidget(box)
        return w

    def _build_controls(self) -> QtWidgets.QWidget:
        """Construye el panel izquierdo: escenarios, física, esquemas y previsualización"""
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(4, 4, 8, 4)
        lay.setSpacing(10)

        # ---- escenarios
        box = QtWidgets.QGroupBox("Escenarios predefinidos")
        v = QtWidgets.QVBoxLayout(box)
        h = QtWidgets.QHBoxLayout()
        self.preset_combo = QtWidgets.QComboBox()
        self.preset_combo.addItems(list(SCENARIOS))
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
        self.preset_note.setStyleSheet("padding-bottom: 5px;")   # los subíndices bajan de la línea
        v.addWidget(self.preset_note)
        self.preset_combo.currentTextChanged.connect(
            lambda t: self.preset_note.setText(SCENARIOS[t].note if t in SCENARIOS else ""))
        self.preset_note.setText(SCENARIOS[self.preset_combo.currentText()].note)
        lay.addWidget(box)

        # ---- problema físico compartido y esquema numérico de cada simulación
        base = DEFAULT_CONFIGS[0]
        self.physics = PhysicsSliders(base.I, base.w0)
        self.physics.changed.connect(self._schedule_apply)
        lay.addWidget(group_box("Problema físico · común a las simulaciones", [self.physics]))

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

        # ---- previsualización
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

    def _build_phase_items(self) -> None:
        """Crea los elementos 3D de la vista de fases y la rejilla de la vista del cuerpo"""
        v = self.phase_view
        self.phase_axes = []
        self.phase_axis_labels = []
        for k in range(3):
            line = gl.GLLinePlotItem(pos=np.zeros((2, 3)), color=rgba(AXIS_COLORS[k], 0.8),
                                     width=1.5, antialias=True, mode="lines")
            v.addItem(line)
            lbl = gl.GLTextItem(text=f"ω{'₁₂₃'[k]}", color=QtGui.QColor(AXIS_COLORS[k]),
                                font=QtGui.QFont(theme.UI_FONT, 13, QtGui.QFont.Weight.Bold))
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

        # órbita exacta por simulación y marcador de dónde debería estar ahora
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

    def _build_plot_items(self) -> None:
        """Crea las curvas de todas las gráficas, los cursores enlazados y los elementos de la carta"""
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
        self.plane_ref_orbit = pp.plot(pen=pg.mkPen(AMBER, width=2,
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

    LINE_PATTERNS = (None, [5, 2.5], [1.2, 2.4])      # A continuo, B rayas, C puntos

    @classmethod
    def _live_pen(cls, k: int, color: str, width: float = 2.0) -> QtGui.QPen:
        """Trazo de la simulación k: A continuo, B rayado y C punteado, para distinguirlas sin el color"""
        pen = pg.mkPen(color, width=width)
        pat = cls.LINE_PATTERNS[k % len(cls.LINE_PATTERNS)]
        if pat is not None:
            pen.setStyle(QtCore.Qt.PenStyle.CustomDashLine)
            pen.setDashPattern(pat)
            pen.setCapStyle(QtCore.Qt.PenCapStyle.FlatCap)
        return pen

    @staticmethod
    def _dash_pen(color: str, alpha: float = 150) -> QtGui.QPen:
        """Trazo discontinuo y semitransparente para las previsualizaciones"""
        c = QtGui.QColor(color)
        c.setAlpha(alpha)
        return pg.mkPen(c, width=1, style=QtCore.Qt.PenStyle.DashLine)

    def _schedule_apply(self, *_) -> None:
        self._param_timer.start()

    def _notice_label(self) -> QtWidgets.QLabel:
        lbl = QtWidgets.QLabel()
        lbl.setObjectName("notice")
        lbl.setWordWrap(True)
        lbl.setTextFormat(QtCore.Qt.TextFormat.RichText)
        lbl.setOpenExternalLinks(False)
        lbl.linkActivated.connect(self._on_notice_link)
        lbl.hide()
        return lbl

    def _on_notice_link(self, href: str) -> None:
        if href.startswith("ref:"):
            self.ref_combo.setCurrentIndex(int(href[4:]))

    def _refresh_tab_titles(self) -> None:
        for k, p in enumerate(self.panels):
            own = not p.is_linked()
            self.tabs.setTabText(k, f"● {p.name}" + ("  ✎" if own else ""))
            self.tabs.setTabToolTip(
                k, f"{p.name}: física propia (desvinculada del problema compartido)" if own
                else f"{p.name}: usa el problema físico compartido")

    def _update_surface_notice(self) -> None:
        """Avisa si alguna simulación no yace sobre las superficies de la simulación de referencia"""
        cfgs = self.configs()
        ref_k = self.ref_index()
        ref = cfgs[ref_k]
        rows = []
        for k, c in enumerate(cfgs):
            if k == ref_k or not c.enabled:
                continue
            m = surface_mismatch(ref.I, ref.w0, c.I, c.w0)
            if m is None:
                continue
            why = ("otra I" if not m["same_I"] else
                   f"otra energía ({m['dT']:+.1%}) y otro momento ({m['dL']:+.1%})")
            rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> {why} "
                        f"(<a href='ref:{k}' style='color:{AMBER}'>ver sus superficies</a>)")
        for label, what in ((self.surface_notice, "Las superficies de esta vista"),
                            (self.plane_notice, "La carta y las órbitas exactas")):
            if rows:
                label.setText(f"⚠ {what} son las de la simulación <b>{ref.name}</b> y no "
                              f"contienen exactamente la trayectoria de: " + " · ".join(rows) + ".")
                label.show()
            else:
                label.hide()

    def apply_preset(self, name: str) -> None:
        """Carga un escenario en los paneles y aplica los parámetros"""
        scenario = SCENARIOS[name]
        upd = scenario.sims
        I0, w00 = self.physics.get()
        I0, w00 = tuple(upd[0].get("I", I0)), tuple(upd[0].get("w0", w00))
        self.physics.set(I0, w00)           # el problema físico del escenario es el de A
        for panel, u in zip(self.panels, upd):
            panel.set_config(replace(panel.get_config(), **u))
            own = (tuple(u.get("I", I0)), tuple(u.get("w0", w00))) != (I0, w00)
            panel.set_linked(not own)       # solo se desvincula quien trae otra física
        self.preset_combo.setCurrentText(name)
        self.apply_params()
        if scenario.log_errors:         # errores de órdenes de magnitud distintos: en log se ven todos
            self.chk_log.setChecked(True)

    def configs(self) -> list[SimConfig]:
        return [p.get_config() for p in self.panels]

    def ref_index(self) -> int:
        return self.ref_combo.currentIndex()

    def apply_params(self) -> None:
        """Aplica los parámetros de los paneles y reinicia las simulaciones"""
        self._param_timer.stop()
        cfgs = self.configs()
        changed = self._describe_changes(self._last_cfgs, cfgs)
        old_cfgs, self._last_cfgs = self._last_cfgs, cfgs
        if not cfgs[self.ref_index()].enabled:
            first = next((k for k, c in enumerate(cfgs) if c.enabled), None)
            if first is not None:
                self.ref_combo.blockSignals(True)
                self.ref_combo.setCurrentIndex(first)
                self.ref_combo.blockSignals(False)
        if self._only_scheme_changed(old_cfgs, cfgs):
            self._switch_schemes(cfgs, changed)
            return
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
        self._clear_analysis_results()
        self._badge_state = [None] * len(self.badges)
        for b in self.badges:               # si no, quedarían a la vista las fichas de simulaciones desactivadas
            b.hide()
        self.plot_w.setTitle(None)
        self._update_preview_items()
        if self.chk_auto.isChecked():
            self.request_preview()
        else:
            self._set_preview_status("desactualizada · pulsa ↻ Calcular [P]")
        self.status.showMessage(f"Parámetros aplicados · simulación reiniciada (t = 0){changed}", 4000)
        self._refresh_views()

    @staticmethod
    def _only_scheme_changed(old: list[SimConfig], new: list[SimConfig]) -> bool:
        """Indica si solo cambió el esquema numérico, caso en que no hace falta reiniciar"""
        if not old or len(old) != len(new):
            return False
        return (all(replace(a, method=b.method) == b for a, b in zip(old, new))
                and any(a.method != b.method for a, b in zip(old, new)))

    def _switch_schemes(self, cfgs: list[SimConfig], changed: str) -> None:
        """Cambia de integrador al vuelo, conservando el tiempo y el historial"""
        for k, (sim, c) in enumerate(zip(self.sims, cfgs)):
            if sim.cfg.method != c.method:
                sim.switch_method(c.method)
                self._div_notified.discard(k)
        self._update_equilibria()
        self._refresh_tab_titles()
        self._update_error_notice()
        self._clear_analysis_results()
        self._badge_state = [None] * len(self.badges)
        self._rebuild_satellites()
        self.previews = {}
        self._update_preview_items()
        if self.chk_auto.isChecked():
            self.request_preview()
        else:
            self._set_preview_status("desactualizada · pulsa ↻ Calcular [P]")
        self.status.showMessage(f"Esquema cambiado al vuelo (el tiempo continúa){changed}", 4000)
        self._refresh_views()

    @staticmethod
    def _describe_changes(old: list[SimConfig], new: list[SimConfig]) -> str:
        """Describe qué cambió desde la última aplicación, para explicar el reinicio"""
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

    def _lin(self, cfg: SimConfig) -> LinearErrors | None:
        """Errores lineales del esquema, con caché porque se consultan en cada refresco"""
        key = (cfg.method, cfg.dt, tuple(cfg.I), tuple(cfg.w0))
        if key not in self._lin_cache:
            if len(self._lin_cache) > 64:
                self._lin_cache.clear()
            self._lin_cache[key] = linear_errors(cfg)
        return self._lin_cache[key]

    def _horizon_steps(self, sim: Simulation) -> int:
        """Pasos hasta el mayor entre el tiempo actual y el horizonte previsto"""
        return ne.n_steps(max(sim.vt, self.preview_T.value()), sim.cfg.dt)

    def _on_horizon_changed(self) -> None:
        if self.chk_auto.isChecked() or self.previews:
            self.request_preview()

    def reset(self) -> None:
        for s in self.sims:
            s.reset()
        self.clock = 0.0
        self.view_t = None
        self._div_notified = set()
        self._ew_cache = None
        self._sync_timeline()
        self._refresh_views()

    def _build_timeline(self) -> QtWidgets.QHBoxLayout:
        """Construye la fila de rebobinado con botones de paso, deslizador y etiqueta de tiempo"""
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
    def show_help(self) -> None:
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Ayuda · Janibekov Lab")
        dlg.resize(640, 640)
        lay = QtWidgets.QVBoxLayout(dlg)
        view = QtWidgets.QTextBrowser()
        view.setOpenExternalLinks(False)
        view.setHtml(HELP_HTML.format(muted=MUTED, ok=STABILITY_COLORS["estable"], mar=STABILITY_COLORS["marginal"],
                                      bad=STABILITY_COLORS["inestable"], g1=GRADE_COLORS[1], g2=GRADE_COLORS[2],
                                      g0=GRADE_COLORS[0]))
        lay.addWidget(view)
        close = QtWidgets.QPushButton("Cerrar")
        close.clicked.connect(dlg.accept)
        lay.addWidget(close, alignment=QtCore.Qt.AlignmentFlag.AlignRight)
        dlg.exec()

    def _pause(self, message: str | None = None) -> None:
        self.running = False
        self.btn_run.setText("▶  Iniciar")
        if message:
            self.status.showMessage(message, 8000)

    def toggle_run(self) -> None:
        self.running = not self.running
        self.btn_run.setText("⏸  Pausa" if self.running else "▶  Iniciar")
        if self.running and self.view_t is None:
            stop = self.stop_T.value()
            if stop > 0 and self.clock >= stop - 1e-12:
                self._pause(f"Pausa automática: ya se alcanzó t = {stop:g} s (cambia «Parar en» para seguir)")

    def _set_view_time(self, t: float) -> None:
        """Fija el instante mostrado del historial; None vuelve al presente"""
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

    def _sync_timeline(self) -> None:
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

    def _on_timeline(self, v: int) -> None:
        if self._syncing or self.clock <= 0:
            return
        self._pause()
        self._set_view_time(None if v >= 10000 else v / 10000.0 * self.clock)

    def _step_size(self) -> float:
        dts = [s.cfg.dt for s in self.sims if s.cfg.enabled]
        return min(dts) if dts else 0.0

    def step_forward(self) -> None:
        self._pause()
        h = self._step_size()
        if h <= 0:
            return
        if self.view_t is not None:
            self._set_view_time(self.view_t + h)
            return
        self._advance_live(h)
        self._refresh_views()

    def step_back(self) -> None:
        self._pause()
        h = self._step_size()
        if h <= 0 or self.clock <= 0:
            return
        base = self.clock if self.view_t is None else self.view_t
        self._set_view_time(max(0.0, base - h))

    def _advance_live(self, dt_clock: float) -> bool:
        """Avanza el reloj y las simulaciones aplicando la parada, la divergencia y el límite de pasos"""
        stop = self.stop_T.value()
        target = self.clock + dt_clock
        if stop > 0:
            if self.clock >= stop - 1e-12:
                self._pause(f"Pausa automática: ya se alcanzó t = {stop:g} s")
                return False
            target = min(target, stop)
        self.clock = target
        for sim in self.sims:
            if sim.cfg.enabled:
                sim.advance_to(self.clock)
        self._sync_timeline()
        active = [(k, s) for k, s in enumerate(self.sims) if s.cfg.enabled]
        for k, sim in active:
            if sim.diverged and k not in self._div_notified:
                self._div_notified.add(k)       # se congela donde diverge; las demás siguen
                what = (f"falló en t = {sim.t:.2f} s ({sim.error})" if sim.error
                        else f"divergió en t = {sim.t:.2f} s")
                self.status.showMessage(f"Divergencia: {sim.cfg.name} · {METHOD_SHORT[sim.cfg.method]} {what} "
                                        f"(queda congelada; las demás continúan)", 12000)
            elif sim.n_steps >= MAX_LIVE_STEPS:
                self._pause(f"Pausa automática: {sim.cfg.name} alcanzó {MAX_LIVE_STEPS:,} pasos")
        if active and all(s.diverged for _, s in active):
            self._pause("Pausa automática: todas las simulaciones activas divergieron")
        if stop > 0 and self.clock >= stop - 1e-12:
            self._pause(f"Pausa automática: se alcanzó t = {stop:g} s")
        return True


    # ---- análisis pesados: se calculan en un hilo y se pintan al terminar
    def _start_analysis(self, kind: str, job: Callable[[SimConfig], Any], button: QtWidgets.QPushButton) -> None:
        """Lanza un análisis en un hilo, cancelando el anterior del mismo tipo"""
        old = self._analysis.pop(kind, None)
        if old is not None:
            old.cancel()                        # el resultado anterior ya no interesa
        worker = AnalysisWorker(kind, self.configs(), job)
        worker.done.connect(self._on_analysis_done)
        worker.finished.connect(partial(self._worker_finished, worker))
        self._analysis[kind] = worker
        self._workers.append(worker)
        button.setEnabled(False)
        button.setText("⏳ …")
        self.status.showMessage("Calculando…")
        worker.start()

    def _on_analysis_done(self, kind: str, results: dict[int, Any], elapsed: float) -> None:
        """Muestra el resultado de un análisis si sigue vigente"""
        worker = self._analysis.get(kind)
        if worker is None or self.sender() is not worker:       # parámetros que ya no son los actuales
            return
        del self._analysis[kind]
        button = self.btn_rich if kind == "rich" else self.btn_conv
        button.setEnabled(True)
        button.setText("Calcular")
        if kind == "rich":
            self._show_richardson(worker.cfgs, results, elapsed)
        else:
            self._show_convergence(worker.cfgs, results, elapsed)

    def run_richardson(self) -> None:
        T = self.rich_T.value()
        self._rich_T = T
        self._start_analysis("rich", lambda c: richardson_report(c, T), self.btn_rich)

    def run_convergence(self) -> None:
        T, levels = self.rich_T.value(), self.conv_levels.value()
        self._rich_T = T
        self._start_analysis("conv", lambda c: convergence_report(c, T, levels), self.btn_conv)

    def _show_richardson(self, cfgs: list[SimConfig], results: dict[int, Any], elapsed: float) -> None:
        """Rellena la tabla de Richardson con el error estimado de cada simulación"""
        T, rows = self._rich_T, []
        for k, c in enumerate(cfgs):
            r = results.get(k)
            if r is None:
                continue
            head = [dot(c), (METHOD_SHORT[c.method], {}), (f"{c.dt:g}", {"mono": True}),
                    (f"{ne.SCHEMES[c.method].order}", {"mono": True, "right": True})]
            if isinstance(r, Exception):
                head[1] = (METHOD_SHORT[c.method], {"tip": f"{type(r).__name__}: {r}"})
                rows.append(head + [("falló", {"right": True, "color": DataTable.ALERT}), DASH])
                self.status.showMessage(f"Richardson: {c.name} falló · {type(r).__name__}: {r}", 12000)
                continue
            finite = np.isfinite(r["err"])
            w = r["u_extr"][:3]
            tip = (f"n = {r['n']} pasos · ω extrapolada = ({w[0]:+.5f}, {w[1]:+.5f}, {w[2]:+.5f})" if finite
                   else "no estimable (divergencia)")
            if r["noise"]:
                tip = "diferencia entre mallas al nivel del ruido de redondeo: el error estimado no es fiable"
            head[1] = (METHOD_SHORT[c.method], {"tip": tip})
            if r["noise"]:
                err = (f"{r['err']:.2e}", {"mono": True, "right": True, "color": MUTED, "tip": "ruido de redondeo"})
            elif finite:
                err = (f"{r['err']:.2e}", {"mono": True, "right": True})
            else:
                err = ("divergió", {"right": True, "color": DataTable.ALERT})
            rows.append(head + [err, num_or_dash(r["err_real"], "{:.2e}")])
        self.rich_table.set_rows(rows)
        self.rich_table.setVisible(bool(rows))
        self.rich_note.setText(self._rich_note_done.format(T=T) if rows else self._rich_note_idle)
        self.status.showMessage(f"Richardson hasta t_f = {T:g} s calculado en {1e3 * elapsed:.0f} ms", 6000)

    def _show_convergence(self, cfgs: list[SimConfig], results: dict[int, Any], elapsed: float) -> None:
        """Dibuja el estudio de convergencia y rellena su tabla"""
        T, rows, floors = self._rich_T, [], []
        for k, c in enumerate(cfgs):
            lines = (self.conv_lines[k], self.conv_bad[k], self.conv_ref[k])
            r = results.get(k)
            if r is None or isinstance(r, Exception):
                for it in lines:
                    it.setData([], [])
                if r is not None:
                    rows.append([dot(c), METHOD_SHORT[c.method], (f"{ne.SCHEMES[c.method].order}", {"mono": True, "right": True}),
                                 DASH, DASH, (f"falló · {type(r).__name__}: {r}", {"color": DataTable.ALERT})])
                    self.status.showMessage(f"Convergencia: {c.name} falló · {type(r).__name__}: {r}", 12000)
                continue
            res, p_rich = r["study"], r["p_richardson"]
            dts, errs, ok = res["dt"], res["err"], res["ok"]
            good = np.isfinite(errs) & (errs > 0)
            self.conv_lines[k].setData(dts[good & ok], errs[good & ok])
            self.conv_bad[k].setData(dts[good & ~ok], errs[good & ~ok])
            if ok.any():
                i0 = int(np.argmax(ok))
                self.conv_ref[k].setData(dts, errs[i0] * (dts / dts[i0]) ** ne.SCHEMES[c.method].order)
            else:
                self.conv_ref[k].setData([], [])
            floors.append(res["floor"])
            loc = " · ".join("—" if not np.isfinite(v) else f"{v:.2f}" for v in res["local"])
            rows.append([dot(c), METHOD_SHORT[c.method], (f"{ne.SCHEMES[c.method].order}", {"mono": True, "right": True}),
                         num_or_dash(p_rich), num_or_dash(res["p"]), (loc, {"mono": True})])
        for k in range(len(cfgs), len(self.conv_lines)):
            for it in (self.conv_lines[k], self.conv_bad[k], self.conv_ref[k]):
                it.setData([], [])
        if floors:
            self.conv_floor.setPos(math.log10(max(floors)))
        self.conv_floor.setVisible(bool(floors))
        self.conv_table.set_rows(rows)
        self.conv_table.setVisible(bool(rows))
        self.plot_conv.autoRange()
        self.conv_note.setText(self._conv_note_idle if rows else "No hay simulaciones activas.")
        self.status.showMessage(f"Convergencia hasta t_f = {T:g} s calculada en {1e3 * elapsed:.0f} ms", 6000)

    def _clear_analysis_results(self) -> None:
        """Borra los resultados de Richardson y convergencia, que dependen de los parámetros"""
        for kind, button in (("rich", self.btn_rich), ("conv", self.btn_conv)):
            worker = self._analysis.pop(kind, None)
            if worker is not None:                          # cálculo en curso con parámetros que ya no valen
                worker.cancel()
                button.setEnabled(True)
                button.setText("Calcular")
        self.rich_note.setText(self._rich_note_idle)
        self.conv_note.setText(self._conv_note_idle)
        self.rich_table.hide()
        self.conv_table.hide()
        for k in range(len(self.conv_lines)):
            for it in (self.conv_lines[k], self.conv_bad[k], self.conv_ref[k]):
                it.setData([], [])
        self.conv_floor.hide()

    def compute_preview(self) -> None:
        self.request_preview()

    def request_preview(self) -> None:
        """Calcula la trayectoria prevista en un hilo, cancelando el cálculo anterior"""
        for w in self._workers:
            if isinstance(w, PreviewWorker):
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

    def _worker_finished(self, worker: PreviewWorker) -> None:
        if worker in self._workers:
            self._workers.remove(worker)
        worker.deleteLater()

    def _on_preview_done(self, token: int, previews: dict[int, dict[str, Any]], elapsed: float) -> None:
        """Guarda la previsualización si no está obsoleta y avisa de errores del esquema"""
        if token != self._preview_token:        # resultado de una versión anterior de los parámetros
            return
        self.previews = previews
        note = f"✔ {self.preview_T.value():g} s calculados en {1e3 * elapsed:.0f} ms"
        if any(p["truncated"] for p in previews.values()):
            note += f" · limitada a {MAX_PREVIEW_STEPS:,} pasos"
        failed = [self.sims[k].cfg.name for k, p in previews.items() if p.get("error") and k < len(self.sims)]
        if failed:
            note += f" · ⚠ error del esquema en {', '.join(failed)}"
            self.status.showMessage("Previsualización: " + "; ".join(
                f"{self.sims[k].cfg.name}: {p['error']}" for k, p in previews.items()
                if p.get("error") and k < len(self.sims)), 12000)
        self._set_preview_status(note)
        self._update_preview_items()

    def _set_preview_status(self, text: str) -> None:
        self.preview_status.setText(text)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        for w in list(self._workers):
            w.cancel()
        for w in list(self._workers):
            w.wait(3000)
        super().closeEvent(event)

    def _update_preview_items(self) -> None:
        """Redibuja las curvas de previsualización en todas las vistas"""
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

    def _update_plane_preview(self) -> None:
        show = self.chk_show_prev.isChecked()
        for k in range(3):
            p = self.previews.get(k)
            if show and p is not None and len(p["t"]) > 1:
                P = self._project_plane(p["W"])
                self.plane_prev[k].setData(P[:, 0], P[:, 1], connect="finite")
            else:
                self.plane_prev[k].setData([], [])

    def _plane_equilibrium(self) -> tuple[int, int]:
        """Devuelve el eje y el signo del equilibrio de la carta, automáticos o elegidos"""
        cfg = self.configs()[self.ref_index()]
        idx = self.plane_axis.currentIndex()
        self.plane_sign.setEnabled(idx != 0)
        self.plane_sign_lbl.setText("Signo" if idx != 0 else "Signo (auto)")
        if idx == 0:
            return default_chart(cfg.I, cfg.w0)
        return idx - 1, (1 if self.plane_sign.currentIndex() == 0 else -1)

    def _project_plane(self, W: Array) -> Array:
        return project_to_chart(W, self._plane_k, self._plane_sign_val)

    def _update_phase_plane(self) -> None:
        """Redibuja la carta del plano de fases para la simulación de referencia"""
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

        D = chart_domain(I, twoT, k)
        xm, ym = D[:, 0].max(), D[:, 1].max()
        self.plane_domain.setData(D[:, 0], D[:, 1])

        if self.chk_plane_field.isChecked():
            X, Y, U, V, spacing = reduced_flow_field(I, twoT, k, sign)
            seg = arrow_segments(X, Y, U, V, 0.8 * spacing)
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

        P = chart_reference_orbit(I, w0, k, sign)
        if P is None:
            self.plane_ref_orbit.setData([], [])
        else:
            self.plane_ref_orbit.setData(P[:, 0], P[:, 1], connect="finite")

        e = equilibrium_analysis(I, twoT)[k]
        col = STABILITY_COLORS[e["stability"]]
        self.plane_eq.setData([0.0], [0.0], symbolBrush=col)
        pp.setRange(xRange=(-1.08 * xm, 1.08 * xm), yRange=(-1.08 * ym, 1.08 * ym), padding=0)

        self.plane_badge.setText(f"{'+' if sign > 0 else '−'}Ω e{sub[k]}  ·  "
                                 f"{e['type'].upper()}  ·  {e['stability']}")
        self.plane_badge.setStyleSheet(badge_css(col, 15))
        self._update_plane_info(cfg, I, twoT, k, sign, e)
        self._update_plane_preview()

    def _update_plane_info(self, cfg: SimConfig, I: Array, twoT: float, k: int, sign: int, e: Any) -> None:
        """Escribe el panel de información de la carta y el efecto de cada esquema"""
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
            shape = "dos inercias iguales (sólido simétrico): caso degenerado, sin separatriz"
        rows = [
            f"<b>Ω</b> = {Om:.4f} &nbsp;(I{sub[k]} Ω² = 2T)",
            f"<b>λ</b> = <span style='font-family:{theme.MONO}'>{lam}</span>",
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
            amp = equilibrium_amplification(c.I, c.w0, k, ne.SCHEMES[c.method].amplification, c.dt)
            if amp is None:
                rows.append(head + "—")
                continue
            R = amp["R"]
            if amp["type"] == "centro":
                g = R - 1
                what = amplitude_trend(g)
                rows.append(head + f"<span style='font-family:{theme.MONO}'>|R|−1 = {g:+.2e}</span> · amplitud {what}")
            else:
                rows.append(head + f"<span style='font-family:{theme.MONO}'>|R(λΔt)| = {R:.4f}</span>")
        self.plane_info.setText("<br><br>".join(rows))
    def _refresh_phase_plane(self) -> None:
        """Actualiza la estela y la cabeza de cada simulación en la carta"""
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

    def _on_ref_changed(self) -> None:
        self._update_static_phase()
        self._update_exact_orbits()
        self._update_equilibria()
        self._update_phase_plane()
        self._update_surface_notice()
        self._update_error_notice()
        self._update_preview_items()
        self._refresh_views()

    def _update_static_phase(self) -> None:
        """Reconstruye elipsoides, separatrices y polhodias de la vista 3D de fases"""
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

    def _update_equilibria(self) -> None:
        """Rellena las tablas de equilibrios y de efecto numérico por esquema"""
        cfgs = self.configs()
        ref_cfg = cfgs[self.ref_index()]
        pos, cols, rows = [], [], []
        for k, e in enumerate(equilibrium_analysis(ref_cfg.I, 2 * float(kinetic_energy(ref_cfg.w0, ref_cfg.I)))):
            col = STABILITY_COLORS[e["stability"]]
            Om = e["omega"][k]
            lam = e["eigvals"][np.argmax(np.abs(e["eigvals"]))]
            lam_txt = "0" if abs(lam) < 1e-12 else (f"±{abs(lam.imag):.4g} i" if abs(lam.imag) > abs(lam.real)
                                                    else f"±{abs(lam.real):.4g}")
            bad = e["stability"] == "inestable"
            rows.append([(f"ω{'₁₂₃'[k]}", {}),
                         (f"±{Om:.4g}", {"mono": True, "right": True}),
                         (f"{e['type']} · {e['stability']}", {"color": DataTable.ALERT if bad else TEXT}),
                         (lam_txt, {"mono": True, "right": True}),
                         (e["nonlinear"], {"color": STABILITY_COLORS[e["nonlinear"]]})])
            pos += [e["omega"], -e["omega"]]
            cols += [rgba(col, 1.0)] * 2
        self.eq_table.set_rows(rows)
        self.eq_card.subtitle.setText(f"ref. {ref_cfg.name}")
        self.eq_points.setData(pos=np.array(pos), color=np.array(cols))

        eff = []
        for c in cfgs:
            if not c.enabled:
                continue
            lin = self._lin(c)
            if lin is None:
                eff.append([dot(c), METHOD_SHORT[c.method], ("—", {"color": MUTED}), ("", {}),
                            ("separatriz o simétrico", {"color": MUTED})])
                continue
            g = lin["dmod"]
            (_, gr), _ = scheme_verdict(lin, ne.n_steps(self.preview_T.value(), c.dt))
            tail = " · ínfimo" if gr == 1 else ""
            if gr == 0:
                txt, alert = "conserva la amplitud (≈ 0)", False
            elif g > 0:
                txt, alert = "la amplitud crece" + tail, gr == 2
            else:
                txt, alert = "la amplitud decae (disipa)" + tail, False
            eff.append([dot(c), METHOD_SHORT[c.method], (f"ω{'₁₂₃'[lin['axis']]}", {}),
                        num(g, "{:+.2e}", alert), (txt, {"color": DataTable.ALERT if alert else TEXT})])
        self.eff_table.set_rows(eff)
        self._update_stability()

    def _update_stability(self) -> None:
        """Dibuja las regiones de estabilidad de los esquemas y los autovalores por el paso"""
        z = [(lam * c.dt, c.color) for c in self.configs() if c.enabled
             for lam in relevant_eigenvalues(c.I, c.w0)]
        view, re, im = stability_window(np.array([v for v, _ in z]), self.stab_zoom.isChecked())
        # la malla es más grande que la vista: con el aspecto fijo se ve más de lo pedido
        for m, curve in self.stab_curves.items():
            curve.setData(*stability_layer(m, re, im)[:2])
        m = list(ne.SCHEMES)[self.stab_shade.currentIndex()]
        _, _, inside, rect = stability_layer(m, re, im)
        img = np.zeros(inside.shape + (4,), dtype=np.ubyte)
        img[inside] = (*QtGui.QColor(SCHEME_COLORS[m]).getRgb()[:3], 50)
        self.stab_fill.setImage(img, autoLevels=False)
        self.stab_fill.setRect(QtCore.QRectF(*rect))
        self.stab_points.setData([dict(pos=(v.real, v.imag), brush=col, pen=pg.mkPen("w", width=0.8))
                                  for v, col in z])
        self.plot_stab.setRange(xRange=view[0], yRange=view[1], padding=0)

    def _update_exact_orbits(self) -> None:
        """Órbita exacta de la referencia y de las simulaciones con otra física (otra I u otra velocidad inicial)"""
        if not self.sims:
            return
        ref_k = self.ref_index()
        ref = self.sims[ref_k]
        for k, sim in enumerate(self.sims):
            own = sim.cfg.enabled and (k == ref_k or not self._same_exact(ref.cfg, sim.cfg))
            P = exact_orbit(sim.exact, 720) if own else None
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

    def _update_exact_markers(self) -> None:
        """Coloca la cabeza de cada órbita exacta en el instante mostrado"""
        if not self.sims:
            return
        t_now = self.clock if self.view_t is None else self.view_t
        for k, sim in enumerate(self.sims):
            if self._exact_show[k] and self.chk_exact.isChecked():
                self.exact_heads[k].setData(pos=sim.exact.omega(np.array([t_now])))
            else:
                self.exact_heads[k].setData(pos=np.zeros((1, 3)))

    def _update_static_visibility(self) -> None:
        """Aplica las casillas de visibilidad a los elementos estáticos del plano de fases"""
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

    def _rebuild_satellites(self, force: bool = False) -> None:
        """Reconstruye los satélites 3D solo si cambió algo del cuerpo (o se fuerza)"""
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
                                  font=QtGui.QFont(theme.UI_FONT, 12, QtGui.QFont.Weight.Bold))
            items = parts + axes + [Lvec, wvec, label]
            for it in items:
                self.sat_view.addItem(it)
            self.sat_items.append(dict(k=k, offset=offset, extent=extent, parts=parts,
                                       axes=axes, L=Lvec, w=wvec, all=items))
        self.sat_view.setCameraPosition(distance=4 + 2.6 * len(cfgs))
        self._update_satellites()

    def _make_body(self, half: ArrayLike, color: str) -> tuple[list[Any], Array]:
        """Crea las mallas del cuerpo según la forma elegida y devuelve su extensión"""
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

    def _update_satellites(self) -> None:
        """Orienta cada satélite y sus ejes, momento angular y velocidad angular"""
        for g in self.sat_items:
            sim = self.sims[g["k"]] if self.sims else None
            if sim is None:
                continue
            y = np.array(sim.vy)
            R, L_dir, w_dir = attitude_vectors(y[3:], sim.I, y[:3], float(np.linalg.norm(sim.Ls0)),
                                               float(np.linalg.norm(sim.cfg.w0)))
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = g["offset"]
            tr = pg.Transform3D(M)
            for part in g["parts"]:
                part.setTransform(tr)
            o = g["offset"]
            for a in range(3):
                g["axes"][a].setData(pos=np.array([o, o + R[:, a] * (g["extent"][a] + 0.35)]))
            g["L"].setData(pos=np.array([o, o + 1.6 * L_dir]))
            g["w"].setData(pos=np.array([o, o + 1.25 * w_dir]))

    def _tick(self) -> None:
        """Avanza la simulación un fotograma"""
        if not self.running:
            return
        step = self.speed.value() * self.FRAME_MS / 1000.0
        if self.view_t is not None:                 # reproduciendo el historial hasta el presente
            self._set_view_time(self.view_t + step)
            return
        self._advance_live(step)
        self._refresh_views()

    def _trail_points(self, sim: Simulation) -> int:
        """Puntos de la estela: la misma duración en segundos para todas, aunque el paso dt difiera"""
        return max(2, int(round(self.trail_spin.value() / sim.cfg.dt)))

    @staticmethod
    def _fmt(d: Array, log: bool) -> Array:
        return log_safe(d) if log else d

    def _on_log_toggled(self, on: bool) -> None:
        for p in (self.plot_T, self.plot_L):
            p.setLogMode(False, on)
            p.setLabel("left", ("|ΔT|" if p is self.plot_T else "|Δ‖L‖|") if on
                       else ("ΔT" if p is self.plot_T else "Δ‖L‖"), **axis_label_style())
        self._update_preview_items()
        self._refresh_views()

    def _refresh_error_tab(self) -> None:
        """Actualiza las curvas en vivo de la pestaña de error"""
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
            self._update_error_floor()

    def _update_exact_curves(self, comp: int) -> None:
        """Dibuja la solución exacta de la referencia y de las simulaciones con otra física"""
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

    def _update_error_preview(self) -> None:
        """Actualiza las curvas de previsualización de la pestaña de error"""
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

    def _update_error_info(self) -> None:
        """Compone el texto explicativo con errores medidos y predicción lineal"""
        def tag(g: int) -> str:
            return pill(GRADE_WORDS[g], GRADE_COLORS[g])
        rows = [f"<span style='color:{MUTED}'><b>Módulo r</b>: separación radial respecto a la órbita "
                f"exacta (0 si el punto está sobre ella).<br><b>Fase ψ</b>: cuánto adelanta (+) o "
                f"retrasa (−) respecto a la exacta <i>en el mismo instante</i>.</span><br>"]
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
            elif ex.kind == "simetrico":
                what = f"precesión uniforme · periodo {ex.period:.4g} s"
            else:
                what = "equilibrio: ω constante"
            rows.append(f"<span style='color:{c.color}; font-weight:700'>● {c.name}</span> "
                        f"<b>{METHOD_SHORT[c.method]}</b> · Δt={c.dt:g}<br>"
                        f"<span style='color:{MUTED}'>exacta: {what}</span>")
            if sim.diverged:
                rows.append(pill("FALLÓ" if sim.error else "DIVERGIÓ", STABILITY_COLORS["inestable"]) +
                            f" <span style='color:{MUTED}'>en t = {sim.t:.2f} s; queda congelada"
                            f"{' · ' + html.escape(sim.error) if sim.error else ''}</span>")
            if sim.nv > 1:
                r, psi, lag = sim.v_er[-1], sim.v_ep[-1], sim.v_el[-1]
                gr, gp = grade_pair(r, psi, sim.v_steps)
                rows.append(f"<span style='font-family:{theme.MONO}'>medido (t={sim.vt:.2f} s)<br>"
                            f"&nbsp;módulo r = {r:+.3e} </span>{tag(gr)}<br>"
                            f"<span style='font-family:{theme.MONO}'>&nbsp;fase ψ = {psi:+.4g}° ({lag:+.3e} s) </span>{tag(gp)}")
            lin = self._lin(c)
            if lin is None:
                rows.append(f"<span style='color:{MUTED}'>sin predicción lineal: la órbita está en (o muy cerca de) la "
                            f"separatriz o el sólido es simétrico; no oscila alrededor de ningún eje estable</span>")
            else:
                n_h = self._horizon_steps(sim)
                (mod, gr), (pha, gp) = scheme_verdict(lin, n_h)
                pr, pp = predicted_errors(lin, n_h)
                o_mod, o_pha = SCHEME_ORDER[c.method]
                rows.append(f"<span style='font-family:{theme.MONO}'>teoría lineal (β = {lin['beta']:.4g}), por paso:<br>"
                            f"&nbsp;módulo |R|−1 = {lin['dmod']:+.2e} → {mod}</span> {tag(gr)}<br>"
                            f"<span style='font-family:{theme.MONO}'>&nbsp;fase arg R − βΔt = {lin['dphase']:+.2e} rad → {pha}</span> {tag(gp)}<br>"
                            f"<span style='font-family:{theme.MONO}'>previsto a t = {n_h * c.dt:.4g} s: r ≈ {pr:+.2e}, ψ ≈ {pp:+.3g}°<br>"
                            f"orden global: módulo {o_mod} · fase {o_pha}</span>")
            rows.append("")
        if not any_row:
            rows.append("Sin simulaciones activas.")
        else:
            rows.append(f"<span style='color:{MUTED}'><b>≈ 0</b>: por debajo del ruido de redondeo "
                        f"(línea discontinua de las gráficas). <b>ínfimo</b>: por encima del redondeo "
                        f"pero |r| &lt; {TINY_R:g} y |ψ| &lt; {TINY_PSI:g}°, invisible en cualquier gráfica. "
                        f"<b>relevante</b>: a tener en cuenta.</span>")
        self.err_info.setText("<br>".join(rows))

    def _refresh_views(self) -> None:
        """Actualiza solo la pestaña visible"""
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
            if ref is not None and self._w_range_key != id(ref):
                # la velocidad angular no supera el máximo de la exacta: si la numérica explota, se sale del recuadro
                lim = 1.15 * ref.w_bound
                if lim > 0:
                    self.plot_w.setYRange(-lim, lim, padding=0)
                self._w_range_key = id(ref)
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

    def _update_status(self) -> None:
        """Actualiza las insignias, la tabla de conservación y la barra de estado"""
        show_detail = self.view_tabs.currentWidget() is self.analysis_tab
        rows = []
        for k, sim in enumerate(self.sims):
            c = sim.cfg
            if not c.enabled:
                self._set_badge(k, None)
                rows.append([dot(c), (f"{c.name} — inactiva", {"color": MUTED}), "", "", "", ""])
                continue
            orbit, okind = orbit_stability(c.I, c.w0)
            lin = self._lin(c)
            if lin is None:
                scheme, s_alert = "sin predicción lineal (separatriz o simétrico)", False
            else:
                (mod, gr_t), (pha, gp_t) = scheme_verdict(lin, self._horizon_steps(sim))
                tg = lambda g: " (ínfimo)" if g == 1 else ""
                scheme, s_alert = f"módulo {mod}{tg(gr_t)} · fase {pha}{tg(gp_t)}", max(gr_t, gp_t) == 2
            if sim.diverged and sim.nv >= sim.n:
                obs, lvl = (f"falló en t = {sim.t:.2f} s · {sim.error}" if sim.error
                            else f"divergió en t = {sim.t:.2f} s"), 3
            elif sim.nv <= 1:
                obs, lvl = "aún sin simular", 0
            else:
                r_, p_ = sim.v_er[-1], sim.v_ep[-1]
                lvl = max(grade_pair(r_, p_, sim.v_steps))
                obs = f"r {r_:+.1e} · ψ {p_:+.1e}° · {GRADE_WORDS[lvl]}"
            self._set_badge(k, (c.name, c.color, METHOD_SHORT[c.method], orbit,
                                STABILITY_COLORS[okind], scheme, s_alert, obs, lvl))
            if show_detail:
                rows.append([dot(c), METHOD_SHORT[c.method], (f"{c.dt:g}", {"mono": True, "right": True}),
                             num(sim.vdT[-1]), num(sim.vdL[-1]),
                             (f"{sim.v_steps}", {"mono": True, "right": True})])
        for k in range(len(self.sims), len(self.badges)):
            self._set_badge(k, None)
        if show_detail:
            self.cons_table.set_rows(rows)
        mensajes_propios = ("Previsualización", "Parámetros", "Richardson", "Convergencia",
                            "Pausa automática", "Divergencia")
        if not self.status.currentMessage().startswith(mensajes_propios):
            self.status.showMessage(
                f"{'▶ en ejecución' if self.running else '⏸ en pausa'}   ·   t = {self.clock:.2f} s"
                f"   ·   [Espacio] iniciar/pausar  [R] reiniciar  [P] recalcular previsualización"
                f"  [,] [.] paso")

    def _set_badge(self, k: int, state: tuple[Any, ...] | None) -> None:
        """Redibuja la insignia solo si su estado cambió"""
        if state == self._badge_state[k]:
            return
        prev = self._badge_state[k]
        self._badge_state[k] = state
        b = self.badges[k]
        if state is None:
            b.hide()
            return
        name, color, method, orbit, col, scheme, s_alert, obs, lvl = state
        amber, neutral = AMBER, "#d3d6db"
        sch_html = f"<span style='color:{amber if s_alert else neutral}'>{scheme}</span>"
        obs_html = f"<span style='color:{amber if lvl >= 2 else neutral}'>{obs}</span>"
        lab = lambda t: f"<span style='color:{MUTED}'>{t}</span>"
        b.setText(f"<span style='color:{color}'>●</span>&nbsp;<b>{name} · {method}</b><br>"
                  f"<span style='font-weight:400; font-size:11px'>{lab('Órbita')} {orbit}<br>"
                  f"{lab('Esquema')} {sch_html}<br>"
                  f"{lab('Observado')} {obs_html}</span>")
        if prev is None or prev[4] != col:
            b.setStyleSheet(badge_css(col, 13))
        b.show()
