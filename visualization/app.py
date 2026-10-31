"""Arranque de la interfaz: crea la aplicación Qt, elige fuentes y estilo y abre la ventana principal"""

from __future__ import annotations

import sys

import pyqtgraph as pg
from PySide6 import QtGui, QtWidgets

from visualization import theme
from visualization.palette import TEXT, VIEW_BG
from visualization.window import MainWindow


def main() -> None:
    """Arranca la aplicación"""
    pg.setConfigOptions(antialias=True, background=VIEW_BG, foreground=TEXT)
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    theme.setup_fonts()
    app.setFont(QtGui.QFont(theme.UI_FONT, 10))
    app.setStyleSheet(theme.build_stylesheet(theme.UI_FONT))
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
