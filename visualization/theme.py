"""Tema de la interfaz: fuentes, hoja de estilos y estilos HTML/CSS pequeños"""

from __future__ import annotations

from typing import Sequence

from PySide6 import QtGui

from visualization.palette import ACCENT, AMBER, BG, BORDER, BUTTON, FIELD, MUTED, PANEL, TEXT


UI_FONT_CANDIDATES = ["Bahnschrift", "Segoe UI Variable Display", "Segoe UI", "Arial"]
MONO_FONT_CANDIDATES = ["Cascadia Mono", "Consolas", "Courier New"]
UI_FONT = "Segoe UI"             # se fijan en setup_fonts(); el resto de módulos los lee como theme.UI_FONT
MONO = "Consolas, monospace"


def pick_font(candidates: Sequence[str]) -> str:
    """Primera fuente de la lista que está instalada"""
    families = set(QtGui.QFontDatabase.families())
    return next((f for f in candidates if f in families), QtGui.QFont().defaultFamily())


def setup_fonts() -> None:
    """Elige UI_FONT y MONO entre las fuentes instaladas; necesita una QApplication ya creada"""
    global UI_FONT, MONO
    UI_FONT = pick_font(UI_FONT_CANDIDATES)
    MONO = f'"{pick_font(MONO_FONT_CANDIDATES)}", monospace'


def axis_label_style() -> dict[str, str]:
    return {"font-family": UI_FONT, "font-size": "11pt", "color": TEXT}


def pill(text: str, color: str) -> str:
    """Etiqueta HTML con fondo de color"""
    text = text.replace(" ", "&nbsp;")           # una etiqueta no se parte en dos líneas
    return (f"<span style='background-color:{color}; color:#04123a; font-weight:700'>"
            f"&nbsp;{text}&nbsp;</span>")


def badge_css(color: str, size: int = 13) -> str:
    """Estilo CSS de una insignia con el color dado"""
    c = QtGui.QColor(color)
    return (f"QLabel {{ background: rgba({c.red()}, {c.green()}, {c.blue()}, 55); color: #ffffff;"
            f" border: 2px solid {color}; border-radius: 12px; padding: 4px 12px;"
            f" font-weight: 700; font-size: {size}px; }}")


def rgba(hex_color: str, alpha: float = 1.0, factor: float = 1.0) -> tuple[float, float, float, float]:
    """Pasa un color hexadecimal a RGBA con un factor de brillo"""
    c = QtGui.QColor(hex_color)
    return (min(c.redF() * factor, 1.0), min(c.greenF() * factor, 1.0),
            min(c.blueF() * factor, 1.0), alpha)


def mono_font(size: int = 11) -> QtGui.QFont:
    """Fuente monoespaciada a partir de la familia MONO"""
    fam = MONO.split('"')[1] if '"' in MONO else MONO.split(",")[0]
    f = QtGui.QFont(fam, size)
    f.setStyleHint(QtGui.QFont.StyleHint.Monospace)
    return f


def build_stylesheet(font: str) -> str:
    """Hoja de estilo de la aplicación con la paleta y la fuente dadas"""
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
QLabel#warning {{ color: {AMBER}; }}
QLabel#notice {{ background: rgba(255, 215, 64, 30); color: #ffe9a0; border: 1px solid {AMBER}; border-radius: 6px; padding: 6px 10px; }}
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
                                          border-bottom: 3px solid {AMBER}; }}
QTabWidget#views QTabBar::tab:hover:!selected {{ background: #4a4e55; color: white; }}
QSplitter::handle {{ background: {BG}; }}
QTableWidget {{ background: transparent; color: {TEXT}; border: none; }}
QHeaderView {{ background: transparent; }}
QHeaderView::section {{ background: transparent; color: {MUTED}; border: none;
                        border-bottom: 1px solid {BORDER}; padding: 4px 8px; font-weight: 600; }}
QTableWidget::item {{ padding: 0 8px; }}
QStatusBar {{ background: {PANEL}; color: {MUTED}; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {MUTED};
                         border-radius: 3px; background: {FIELD}; }}
QCheckBox::indicator:checked {{ background: #00b37e; border-color: #3ddc97; }}
"""
