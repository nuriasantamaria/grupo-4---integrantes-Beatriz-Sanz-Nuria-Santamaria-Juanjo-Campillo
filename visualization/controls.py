"""Controles de entrada: sliders, parámetros físicos (I, ω₀) y panel de cada simulación"""

from __future__ import annotations

from typing import Any, Sequence

from PySide6 import QtCore, QtWidgets

from simulation import SimConfig
from space_physics import Array, ArrayLike

from visualization.geometry import inertia_ranks, triangle_violation
from visualization.initial_conditions import axis_decomposition, omega_from_axis, reassign_axes
from visualization.schemes import METHOD_LABEL


class FloatSlider(QtWidgets.QWidget):
    """Deslizador con caja numérica sincronizada"""

    valueChanged = QtCore.Signal(float)

    def __init__(self, label: str, vmin: float, vmax: float, step: float, value: float, decimals: int = 3,
                 parent: QtWidgets.QWidget | None = None,
                 spin_range: tuple[float, float] | None = None) -> None:
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

    def value(self) -> float:
        return self.spin.value()

    def setValue(self, v: float, silent: bool = False) -> None:
        """Fija el valor dentro del rango y emite la señal salvo que silent sea verdadero"""
        v = min(max(float(v), self._lo), self._hi)
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.spin.setValue(v)
        self.slider.setValue(self._pos(v))
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        if not silent:
            self.valueChanged.emit(v)

    def _from_slider(self, i: int) -> None:
        v = self.vmin + i * self.step
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.spin.value())

    def _pos(self, v: float) -> int:
        return min(max(int(round((v - self.vmin) / self.step)), 0), self._n)

    def _from_spin(self, v: float) -> None:
        self.slider.blockSignals(True)
        self.slider.setValue(self._pos(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(v)


def group_box(title: str, widgets: Sequence[QtWidgets.QWidget]) -> QtWidgets.QGroupBox:
    box = QtWidgets.QGroupBox(title)
    v = QtWidgets.QVBoxLayout(box)
    v.setSpacing(6)
    for w in widgets:
        v.addWidget(w)
    return box


class PhysicsSliders(QtWidgets.QWidget):
    """Momentos de inercia y velocidad angular inicial, que se puede dar por eje más perturbaciones o por componentes"""

    changed = QtCore.Signal()
    AXIS_NAMES = ("e₁", "e₂", "e₃")

    def __init__(self, I: ArrayLike = (1.0, 2.0, 3.0), w0: ArrayLike = (0.0, 0.0, 0.0),
                 parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)

        def heading(text: str) -> QtWidgets.QLabel:
            lbl = QtWidgets.QLabel(text)
            lbl.setObjectName("muted")
            return lbl

        inertia_head = heading("Momentos principales de inercia")
        inertia_head.setToolTip("En kg·m² (solo importan las razones entre ellos)")
        lay.addWidget(inertia_head)
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
        w_head = heading("Velocidad angular Ω₀")
        w_head.setToolTip("En rad/s")
        head.addWidget(w_head, 1)
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

    # ---- vector inicial y su descomposición en eje y perturbaciones
    def _vector_from_axis(self) -> Array:
        return omega_from_axis(self._axes, self.w_main.value(), self.p1.value(),
                               self.p2.value() if self.chk_p2.isChecked() else 0.0)

    def _fill_axis_widgets(self, w: ArrayLike) -> None:
        """Rellena el modo eje con el vector w sin emitir señales"""
        self._busy = True
        self._axes = axis_decomposition(w, self._axes)
        a, b, c = self._axes
        self.w_main.setValue(w[a], silent=True)
        self.p1.setValue(w[b], silent=True)
        self.p2.setValue(w[c], silent=True)
        self.chk_p2.setChecked(abs(w[c]) > 0)
        self._rebuild_axis_combos()
        self._busy = False
        self._update_readout()

    def _axis_text(self, k: int) -> str:
        return f"{self.AXIS_NAMES[k]} · {self._tags[k]}" if self._tags[k] else self.AXIS_NAMES[k]

    def _rebuild_axis_combos(self) -> None:
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

    def _update_readout(self) -> None:
        w = self._vector_from_axis()
        self.readout.setText(f"Ω₀ = ({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})")

    def _sync_components(self, w: ArrayLike) -> None:
        for s, v in zip(self.w_sliders, w):
            s.setValue(v, silent=True)

    # ---- reacciones a la interfaz
    def _on_inertia(self, k: int) -> None:
        self._update_tags()
        self.changed.emit()

    def _on_components(self, *_) -> None:
        if not self._busy:
            self.changed.emit()

    def _on_axis_values(self, *_) -> None:
        if self._busy:
            return
        self.p2.setEnabled(self.chk_p2.isChecked())
        w = self._vector_from_axis()
        self._sync_components(w)
        self._update_readout()
        self.changed.emit()

    def _on_main_axis(self, _: Any = None) -> None:
        if self._busy:
            return
        a_new = self.axis_combo.currentData()
        a, b, c = self._axes
        if a_new == a:
            return
        self._axes = reassign_axes(self._axes, new_main=a_new)
        self._rebuild_axis_combos()
        self._on_axis_values()

    def _on_p1_axis(self, _: Any = None) -> None:
        if self._busy:
            return
        b_new = self.p1_axis.currentData()
        if b_new is None or b_new == self._axes[1]:
            return
        self._axes = reassign_axes(self._axes, new_p1=b_new)
        self._rebuild_axis_combos()
        self._on_axis_values()

    def _on_mode(self, i: int) -> None:
        axis_mode = (i == 0)
        if axis_mode:
            self._fill_axis_widgets([s.value() for s in self.w_sliders])
        self.axis_page.setVisible(axis_mode)
        self.comp_page.setVisible(not axis_mode)

    def _update_tags(self) -> None:
        """Rotula cada I como mínima, media o máxima"""
        I = [s.value() for s in self.I_sliders]
        tags = inertia_ranks(I)                 # '=' si empata: sólido simétrico
        self._tags = tags
        for k, s in enumerate(self.I_sliders):
            s.label.setText(f"I{k + 1} · {tags[k]}")
        self._rebuild_axis_combos()
        # solo se avisa, no se impide: las ecuaciones de Euler valen aunque el sólido no pueda existir
        k = triangle_violation(I)
        if k is not None:
            self.tri_warning.setText(
                f"I{k + 1} > I{(k + 1) % 3 + 1} + I{(k + 2) % 3 + 1}: ningún sólido real tiene estos "
                f"momentos de inercia (desigualdad triangular). Las ecuaciones de Euler siguen valiendo.")
            self.tri_warning.show()
        else:
            self.tri_warning.hide()

    def get(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Devuelve los momentos de inercia y las componentes de la velocidad angular inicial"""
        w = self._vector_from_axis() if self.mode.currentIndex() == 0 else [s.value() for s in self.w_sliders]
        return (tuple(s.value() for s in self.I_sliders), tuple(float(x) for x in w))

    def set(self, I: ArrayLike, w0: ArrayLike) -> None:
        """Fija los momentos de inercia y la velocidad angular inicial sin emitir señales"""
        for s, v in zip(self.I_sliders, I):
            s.setValue(v, silent=True)
        self._sync_components(w0)
        self._update_tags()
        self._fill_axis_widgets([s.value() for s in self.w_sliders])
        axis_mode = self.mode.currentIndex() == 0
        self.axis_page.setVisible(axis_mode)
        self.comp_page.setVisible(not axis_mode)


class SimConfigPanel(QtWidgets.QWidget):
    """Panel de una simulación: integrador, paso de tiempo y física propia opcional"""

    changed = QtCore.Signal()

    def __init__(self, index: int, cfg: SimConfig, shared: PhysicsSliders,
                 parent: QtWidgets.QWidget | None = None) -> None:
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
        for key, label in METHOD_LABEL.items():
            self.method.addItem(label, key)
        self.method.setSizeAdjustPolicy(
            QtWidgets.QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.method.setMinimumContentsLength(16)
        self.dt = FloatSlider("Δt", 0.001, 0.25, 0.001, 0.02, 3)
        lay.addWidget(group_box("Esquema numérico", [self.method, self.dt]))

        self.linked = QtWidgets.QCheckBox("Usar el problema físico compartido")
        self.linked.setChecked(True)
        self.linked.setToolTip("Desmárcalo para dar a esta simulación sus propios I y Ω₀ "
                               "(p. ej. para comparar cuerpos distintos).")
        self.phys_summary = QtWidgets.QLabel()
        self.phys_summary.setObjectName("muted")
        self.phys_summary.setWordWrap(True)
        self.own = PhysicsSliders(cfg.I, cfg.w0)
        self.own_box = group_box(f"Física propia de {cfg.name}", [self.own])
        lay.addWidget(group_box("Problema físico", [self.linked, self.phys_summary, self.own_box]))
        lay.addStretch(1)

        self.set_config(cfg)
        self._sync_link_ui()

        self.enabled.toggled.connect(self.changed)
        self.method.currentIndexChanged.connect(self.changed)
        self.dt.valueChanged.connect(self.changed)
        self.own.changed.connect(self.changed)
        self.linked.toggled.connect(self._on_link_toggled)
        shared.changed.connect(self._update_summary)

    def is_linked(self) -> bool:
        return self.linked.isChecked()

    def set_linked(self, flag: bool) -> None:
        self.linked.blockSignals(True)
        self.linked.setChecked(bool(flag))
        self.linked.blockSignals(False)
        self._sync_link_ui()

    def _on_link_toggled(self, on: bool) -> None:
        if not on:      # parte de los valores compartidos: al desvincular nada salta
            self.own.set(*self.shared.get())
        self._sync_link_ui()
        self.changed.emit()

    def _sync_link_ui(self) -> None:
        linked = self.linked.isChecked()
        self.own_box.setVisible(not linked)
        self.phys_summary.setVisible(linked)
        self._update_summary()

    def _update_summary(self) -> None:
        if not self.linked.isChecked():
            return
        I, w = self.shared.get()
        self.phys_summary.setText(
            f"I = ({I[0]:.2f}, {I[1]:.2f}, {I[2]:.2f})   "
            f"Ω₀ = ({w[0]:+.3f}, {w[1]:+.3f}, {w[2]:+.3f})")

    def get_config(self) -> SimConfig:
        """Construye la configuración con la física compartida o la propia según el vínculo"""
        I, w0 = self.shared.get() if self.linked.isChecked() else self.own.get()
        return SimConfig(
            name=self.name, color=self.color,
            enabled=self.enabled.isChecked(),
            method=self.method.currentData(),
            I=I, w0=w0, dt=self.dt.value())

    def set_config(self, cfg: SimConfig) -> None:
        widgets = [self.enabled, self.method]
        for w in widgets:
            w.blockSignals(True)
        self.enabled.setChecked(cfg.enabled)
        self.method.setCurrentIndex(self.method.findData(cfg.method))
        for w in widgets:
            w.blockSignals(False)
        self.own.set(cfg.I, cfg.w0)
        self.dt.setValue(cfg.dt, silent=True)
