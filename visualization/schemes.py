"""Presentación de los esquemas numéricos: claves, etiquetas, colores, órdenes y región de estabilidad en caché (sin Qt)"""

from __future__ import annotations

from functools import lru_cache

import numerical_engine as ne

from visualization.plot_tools import stability_contour


# esquemas: claves del registro de numerical_engine (etiqueta, orden y R(z) se leen de ahí)
M_EULER, M_BE, M_CN, M_RK4 = "euler", "be", "cn", "rk4"
METHOD_LABEL = {k: s.label for k, s in ne.SCHEMES.items()}
METHOD_SHORT = {k: s.short for k, s in ne.SCHEMES.items()}
SCHEME_COLORS = {M_EULER: "#ff8a80", M_BE: "#b39ddb", M_CN: "#ffd740", M_RK4: "#80cbc4"}


def _power(n: int | None) -> str:
    """Texto del orden en dt: dt, dt2, dt3... (None si el esquema conserva el módulo)"""
    return "nulo (|R| = 1)" if n is None else "Δt" + ("" if n == 1 else "⁰¹²³⁴⁵⁶⁷⁸⁹"[n])


# orden global en dt del error de módulo y de fase de cada esquema en el centro lineal (registro de ne.SCHEMES)
SCHEME_ORDER = {k: (_power(s.module_order), _power(s.phase_order)) for k, s in ne.SCHEMES.items()}


@lru_cache(maxsize=64)
def stability_layer(method: str, re: tuple[float, float], im: tuple[float, float]):
    """Contorno |R(z)| = 1, máscara |R| <= 1 y ventana de un esquema, en caché para no recalcular al mover los sliders"""
    return stability_contour(ne.SCHEMES[method].amplification, re, im)
