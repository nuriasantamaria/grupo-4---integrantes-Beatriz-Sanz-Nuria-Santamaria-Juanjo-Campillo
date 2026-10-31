"""Widgets genéricos: tablas, tarjetas, layouts, cursor enlazado y ayudas de maquetación"""

from __future__ import annotations

from functools import partial
from typing import Any, Sequence

from PySide6 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from visualization.palette import MUTED, TEXT
from visualization.theme import mono_font


class DataTable(QtWidgets.QTableWidget):
    """Tabla de solo lectura, sin rejilla ni selección, que se ajusta a su contenido"""

    ALERT = "#FF8A80"

    def __init__(self, headers: Sequence[str], parent: QtWidgets.QWidget | None = None) -> None:
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

    def set_rows(self, rows: Sequence[Sequence[Any]]) -> None:
        """Rellena la tabla; cada celda es un texto o un par (texto, opciones con color, mono, right, tip)"""
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
        self._fit_height()

    def _fit_height(self) -> None:
        """Alto justo para las filas, más la barra horizontal si las columnas no caben"""
        h = self.horizontalHeader().height() + sum(self.rowHeight(i) for i in range(self.rowCount())) + 6
        if sum(self.columnWidth(c) for c in range(self.columnCount())) > self.width():
            h += self.horizontalScrollBar().sizeHint().height()
        self.setFixedHeight(max(h, 40))

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self._fit_height()


def scroll_area(widget: QtWidgets.QWidget, transparent: bool = False, frame: bool = False) -> QtWidgets.QScrollArea:
    """Área con scroll vertical (nunca horizontal) que envuelve `widget`; `transparent` la funde con la tarjeta"""
    sa = QtWidgets.QScrollArea()
    sa.setWidgetResizable(True)
    sa.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    if not frame:
        sa.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
    if transparent:
        sa.setStyleSheet("QScrollArea { background: transparent; }")
        sa.viewport().setAutoFillBackground(False)
    sa.setWidget(widget)
    return sa


def noise_floor_line(plot: pg.PlotWidget) -> pg.InfiniteLine:
    """Línea horizontal discontinua de ruido de redondeo, oculta hasta que se le da una altura"""
    line = pg.InfiniteLine(angle=0, pen=pg.mkPen((255, 255, 255, 110), width=1, style=QtCore.Qt.PenStyle.DashLine),
                           label="ruido de redondeo", labelOpts=dict(position=0.9, color=MUTED))
    plot.addItem(line, ignoreBounds=True)
    line.hide()
    return line


def plain_body(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Contenedor transparente que apila `widgets` (cuerpo de una tarjeta)"""
    w = QtWidgets.QWidget()
    w.setObjectName("plain")
    v = QtWidgets.QVBoxLayout(w)
    v.setContentsMargins(0, 0, 0, 0)
    v.setSpacing(8)
    for x in widgets:
        v.addWidget(x)
    v.addStretch(1)
    return w


def muted_label(text: str) -> QtWidgets.QLabel:
    """Texto explicativo gris con ajuste de línea"""
    lbl = QtWidgets.QLabel(text)
    lbl.setObjectName("muted")
    lbl.setWordWrap(True)
    return lbl


def control_row(*widgets: QtWidgets.QWidget) -> QtWidgets.QWidget:
    """Fila de controles alineados a la izquierda, sin márgenes"""
    w = QtWidgets.QWidget()
    lay = QtWidgets.QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    for x in widgets:
        lay.addWidget(x)
    lay.addStretch(1)
    return w


def num(x: float, fmt: str = "{:+.3e}", alert: Any = False) -> tuple[str, dict[str, Any]]:
    """Celda numérica en tipografía fija, resaltada si alert"""
    return (fmt.format(x), {"mono": True, "right": True, "color": DataTable.ALERT if alert else TEXT})


class LinkedCursor(QtCore.QObject):
    """Línea vertical que acompaña al ratón en varias gráficas a la vez y avisa del instante"""

    moved = QtCore.Signal(float)               # t bajo el ratón; NaN cuando el ratón sale

    def __init__(self, plots: Sequence[pg.PlotItem], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.lines: list[Any] = []
        pen = pg.mkPen((255, 255, 255, 140), width=1, style=QtCore.Qt.PenStyle.DashLine)
        for p in plots:
            ln = pg.InfiniteLine(angle=90, movable=False, pen=pen)
            ln.setZValue(20)
            ln.hide()
            p.addItem(ln, ignoreBounds=True)
            self.lines.append(ln)
            p.scene().sigMouseMoved.connect(partial(self._on_move, p))
            p.viewport().installEventFilter(self)

    def _on_move(self, plot: pg.PlotItem, pos: Any) -> None:
        vb = plot.getPlotItem().vb
        if not vb.sceneBoundingRect().contains(pos):
            return
        x = float(vb.mapSceneToView(pos).x())
        for ln in self.lines:
            ln.setPos(x)
            ln.show()
        self.moved.emit(x)

    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        if ev.type() == QtCore.QEvent.Type.Leave:
            for ln in self.lines:
                ln.hide()
            self.moved.emit(float("nan"))
        return False


class FlowLayout(QtWidgets.QLayout):
    """Coloca los widgets en fila y baja a otra línea cuando no caben (nada queda cortado)"""

    def __init__(self, parent: QtWidgets.QWidget | None = None, hspacing: int = 10, vspacing: int = 6) -> None:
        super().__init__(parent)
        self._items, self._h, self._v = [], hspacing, vspacing

    def addItem(self, item: QtWidgets.QLayoutItem) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, i: int) -> QtWidgets.QLayoutItem | None:
        return self._items[i] if 0 <= i < len(self._items) else None

    def takeAt(self, i: int) -> QtWidgets.QLayoutItem | None:
        return self._items.pop(i) if 0 <= i < len(self._items) else None

    def expandingDirections(self) -> QtCore.Qt.Orientation:
        return QtCore.Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, w: int) -> int:
        return self._layout(QtCore.QRect(0, 0, w, 0), True)

    def setGeometry(self, rect: QtCore.QRect) -> None:
        super().setGeometry(rect)
        self._layout(rect, False)

    def sizeHint(self) -> QtCore.QSize:
        return self.minimumSize()

    def minimumSize(self) -> QtCore.QSize:
        s = QtCore.QSize()
        for it in self._items:
            if not it.isEmpty():
                s = s.expandedTo(it.minimumSize())
        m = self.contentsMargins()
        return s + QtCore.QSize(m.left() + m.right(), m.top() + m.bottom())

    def _layout(self, rect: QtCore.QRect, test_only: bool) -> int:
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
    """Reparte en horizontal si hay sitio y en vertical si la ventana es estrecha"""

    def __init__(self, threshold: int = 1000, sizes_h: Sequence[int] = (1000, 1000),
                 sizes_v: Sequence[int] = (1000, 1000), parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(QtCore.Qt.Orientation.Horizontal, parent)
        self.threshold, self.sizes_h, self.sizes_v = threshold, sizes_h, sizes_v
        self._sized = False                 # el reparto inicial se aplica una sola vez, al mostrarse
        self.setChildrenCollapsible(False)

    def showEvent(self, e: QtGui.QShowEvent) -> None:
        super().showEvent(e)
        if not self._sized:
            self._sized = True
            self.setSizes(list(self.sizes_h if self.orientation() == QtCore.Qt.Orientation.Horizontal
                               else self.sizes_v))

    def minimumSizeHint(self) -> QtCore.QSize:
        # el mínimo es el del modo vertical (el más estrecho); en horizontal ya no se activa
        w = max((self.widget(i).minimumSizeHint().width() for i in range(self.count())), default=0)
        return QtCore.QSize(w, 2 * 220)

    def resizeEvent(self, e: QtGui.QResizeEvent) -> None:
        want = (QtCore.Qt.Orientation.Horizontal if e.size().width() >= self.threshold
                else QtCore.Qt.Orientation.Vertical)
        if want != self.orientation():
            self.setOrientation(want)
            self.setSizes(list(self.sizes_h if want == QtCore.Qt.Orientation.Horizontal else self.sizes_v))
        super().resizeEvent(e)


class Card(QtWidgets.QFrame):
    """Marco con título, subtítulo opcional y un widget dentro"""

    def __init__(self, title: str, widget: QtWidgets.QWidget, subtitle: str = "",
                 parent: QtWidgets.QWidget | None = None) -> None:
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
