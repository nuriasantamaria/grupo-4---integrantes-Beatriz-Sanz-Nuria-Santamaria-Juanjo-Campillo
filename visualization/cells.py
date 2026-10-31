"""Celdas de tabla (texto, estilo) que no dependen de Qt"""

from __future__ import annotations

from typing import Any

import numpy as np

from simulation import SimConfig

from visualization.palette import MUTED


DASH = ("—", {"right": True, "color": MUTED})      # celda sin dato


def num_or_dash(x: float, fmt: str = "{:.2f}") -> tuple[str, dict[str, Any]]:
    """Celda numérica a la derecha en tipografía fija, o un guion si el valor no es finito"""
    return (fmt.format(x), {"mono": True, "right": True}) if np.isfinite(x) else DASH


def dot(cfg: SimConfig) -> tuple[str, dict[str, Any]]:
    """Celda con un punto del color de la simulación"""
    return ("●", {"color": cfg.color})
