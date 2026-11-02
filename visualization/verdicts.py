"""Veredictos del análisis (sin Qt): gravedad de los errores, resolución del paso y estabilidad de la órbita"""

from __future__ import annotations

import math
from typing import TypedDict

from numpy.typing import ArrayLike

from numerical_engine import noise_floor
from simulation import LinearErrors, predicted_errors
from space_physics import rotation_regime, rotation_stability

from visualization.cells import dot


GRADE_WORDS = ("≈ 0", "ínfimo", "relevante")


# ---- veredictos del analisis (texto y gravedad)

# por debajo de esto no se ve en ninguna gráfica (módulo adimensional, fase en grados)
TINY_R, TINY_PSI = 1e-6, 1e-3


def grade(x: float, floor: float, tiny: float) -> int:
    """0: ruido de redondeo, 1: ínfimo, 2: a tener en cuenta"""
    size = abs(x)
    if size <= floor:
        return 0
    if size < tiny:
        return 1
    return 2


def grade_pair(r: float, psi: float, n: int) -> tuple[int, int]:
    """Gravedad del error de módulo r y de fase psi (en grados) tras n pasos"""
    fl = noise_floor(n)
    return grade(r, fl, TINY_R), grade(psi, math.degrees(fl), TINY_PSI)


# dt*beta por encima de esto (unos 25 pasos por ciclo) el paso es grueso
COARSE_BETA_DT = 0.25


# |R| por encima de esto el error de módulo explota en pocos pasos
EXPLOSION_R = 1.5


# |R| - 1 por debajo de esto se considera que se conserva
TREND_TOL = 1e-10


def amplitude_trend(dmod: float, tol: float = TREND_TOL) -> str:
    """'crece', 'decae' o 'se conserva' según el signo de |R| - 1 por paso"""
    if dmod > tol:
        return "crece"
    if dmod < -tol:
        return "decae"
    return "se conserva"


class StepResolution(TypedDict):
    """Pasos por ciclo del esquema y si la teoría lineal lo da por perdido"""
    steps_per_cycle: float   # periodo de oscilación / dt
    coarse: bool             # dt*beta > COARSE_BETA_DT
    explodes: bool           # |R| > EXPLOSION_R


def step_resolution(lin: LinearErrors, dt: float) -> StepResolution:
    """Resolución temporal de dt frente a la oscilación del centro"""
    return {"steps_per_cycle": lin["period"] / dt, "coarse": lin["beta"] * dt > COARSE_BETA_DT,
            "explodes": abs(lin["R"]) > EXPLOSION_R}


def scheme_verdict(lin: LinearErrors, n: int) -> tuple[tuple[str, int], tuple[str, int]]:
    """Qué prevé la teoría lineal para módulo y fase tras n pasos, como (palabra, gravedad) de cada uno"""
    r, psi = predicted_errors(lin, n)
    gr, gp = grade_pair(r, psi, n)
    if gr == 0:
        mod = "se conserva"
    else:
        mod = "crece" if lin["dmod"] > 0 else "decae"
    if gp == 0:
        pha = "sin error"
    else:
        pha = "se retrasa" if lin["dphase"] < 0 else "se adelanta"
    return (mod, gr), (pha, gp)


# ---- estabilidad de la órbita (etiquetas de la GUI; la clasificación es de space_physics)

REGIME_TEXT = {"reposo": "reposo (ω = 0)", "esfera": "esfera inercial", "simetrico": "sólido simétrico",
               "separatriz": "sobre la separatriz", "eje_mayor": "precesa en torno al eje mayor",
               "eje_menor": "precesa en torno al eje menor"}
MARGINAL_TEXT = {"reposo": "en reposo · sin rotación", "esfera": "esfera inercial · ω constante",
                 "simetrico": "sólido simétrico · precesión uniforme"}


def regime(I: ArrayLike, w0: ArrayLike) -> str:
    """Texto del régimen geométrico de la órbita; la estabilidad la da orbit_stability"""
    return REGIME_TEXT[rotation_regime(I, w0)]


def orbit_stability(I: ArrayLike, w0: ArrayLike) -> tuple[str, str]:
    """Estabilidad física de la rotación: (texto, 'estable' | 'marginal' | 'inestable')"""
    s = rotation_stability(I, w0)
    if s["stability"] == "marginal":
        return MARGINAL_TEXT[s["regime"]], "marginal"
    dev, ax = s["deviation"], s["axis"]
    if ax is None:
        donde = " · sobre la separatriz"
    else:
        lado = "mayor" if s["regime"] == "eje_mayor" else "menor"
        donde = f" · rodea el eje {lado} e{'₁₂₃'[ax]}"
    if s["stability"] == "inestable":
        return f"inestable · vuelca: ω llega a {dev:.0f}° de su dirección inicial{donde}", "inestable"
    return f"estable · ω se desvía como máximo {dev:.0f}° de su dirección inicial{donde}", "estable"
