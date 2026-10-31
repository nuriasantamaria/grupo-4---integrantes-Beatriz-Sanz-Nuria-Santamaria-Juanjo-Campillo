"""Veredictos del análisis (sin Qt): gravedad de los errores, resolución del paso y estabilidad de la órbita"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TypedDict

import numpy as np

from space_physics import (ArrayLike, EulerExact, is_degenerate, kinetic_energy, LinearErrors, noise_floor,
                           predicted_errors, separatrix_side, SEPARATRIX_TOL, surrounded_axis)

from visualization.cells import dot
from visualization.geometry import exact_orbit


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


# ---- estabilidad de la órbita (etiquetas de la GUI)

def regime(I: ArrayLike, w0: ArrayLike) -> str:
    """Régimen geométrico de la órbita (reposo, esfera, simétrico, separatriz o precesión); la estabilidad la da orbit_stability"""
    twoT = 2 * kinetic_energy(w0, I)
    if twoT <= 0:
        return "reposo (ω = 0)"
    if np.ptp(np.asarray(I, dtype=float)) < 1e-6:
        return "esfera inercial"
    if is_degenerate(I):
        return "sólido simétrico"
    side = separatrix_side(I, w0)
    if abs(side) < SEPARATRIX_TOL:
        return "sobre la separatriz"
    if side > 0:
        return "precesa en torno al eje mayor"
    return "precesa en torno al eje menor"


# desviación máxima del eje de giro (grados) a partir de la cual se considera que la rotación vuelca
FLIP_DEG = 90.0


@lru_cache(maxsize=64)
def _max_deviation(I: tuple[float, ...], w0: tuple[float, ...]) -> float:
    """Versión con caché de max_deviation, con argumentos hashables"""
    orbit = exact_orbit(EulerExact(I, w0), 720)
    if orbit is None:
        return 0.0
    dot = np.sum(orbit * np.asarray(w0), axis=1)                 # producto escalar con w0, sin BLAS
    cos = dot / (np.linalg.norm(orbit, axis=1) * np.linalg.norm(w0))
    return float(np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0))))


def max_deviation(I: ArrayLike, w0: ArrayLike) -> float:
    """Ángulo máximo, en grados, entre omega(t) y omega0 a lo largo de la órbita exacta"""
    # distingue el eje intermedio: allí llega a ~180 grados por pequeña que sea la perturbación
    return _max_deviation(tuple(float(x) for x in np.ravel(I)), tuple(float(x) for x in np.ravel(w0)))


def orbit_stability(I: ArrayLike, w0: ArrayLike) -> tuple[str, str]:
    """Estabilidad física de la rotación: (texto, 'estable' | 'marginal' | 'inestable')"""
    r = regime(I, w0)
    if "reposo" in r:
        return "en reposo · sin rotación", "marginal"
    if "esfera" in r:
        return "esfera inercial · ω constante", "marginal"
    if "simétrico" in r:
        return "sólido simétrico · precesión uniforme", "marginal"
    dev = max_deviation(I, w0)
    ax = surrounded_axis(I, w0)
    if ax is None:
        donde = " · sobre la separatriz"
    else:
        lado = "mayor" if "mayor" in r else "menor"
        donde = f" · rodea el eje {lado} e{'₁₂₃'[ax]}"
    if dev >= FLIP_DEG:
        return f"inestable · vuelca: ω llega a {dev:.0f}° de su dirección inicial{donde}", "inestable"
    return f"estable · ω se desvía como máximo {dev:.0f}° de su dirección inicial{donde}", "estable"
