"""Determinación de la estabilidad de los ejes. Equilibrios, autovalores, estabilidad lineal y no lineal, y lado 
de la separatriz, y régimen y estabilidad de la rotación"""

from __future__ import annotations

from functools import lru_cache
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from space_physics.exact import EulerExact
from space_physics.model import Array, angular_momentum_sq, is_degenerate, kinetic_energy, sorted_axes

ComplexArray = NDArray[np.complex128]

class Equilibrium(TypedDict):
    """Equilibrio w = Omega*e_k y su estabilidad, resultado de equilibrium_analysis"""
    omega: Array           # rotación pura, con Omega^2 = 2T/I_k
    eigvals: ComplexArray  # autovalores de la Jacobiana
    type: str              # clasificación lineal
    stability: str         # 'estable' | 'marginal' | 'inestable' (lineal)
    nonlinear: str         # 'estable' | 'inestable' | 'neutro' (con T y |L|^2)

# ================ [1] EQUILIBRIOS ================

def principal_equilibria(I: ArrayLike, twoT: float) -> Array:
    """Cada fila k es una rotación pura sobre el eje k con energía 2T: omega = sqrt(2T/I_k) * e_k"""
    I = np.asarray(I, dtype=float)
    return np.diag(np.sqrt(twoT / I))


def analytic_eigenvalues(I: ArrayLike, twoT: float) -> ComplexArray:
    """Cada fila k contiene los autovalores (0, mu, -mu) de la Jacobiana en el eje k."""
    I = np.asarray(I, dtype=float)
    lam = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3 # Índices de los otros dos ejes distintos de k. %3 hace roten cíclicamente.
        omega2 = twoT / I[k]
        mu2 = omega2 * (I[k] - I[i]) * (I[j] - I[k]) / (I[i] * I[j])
        mu = np.sqrt(complex(mu2))
        lam[k] = [0, mu, -mu] # El autovalor 0 tiene asociado el autovector a lo largo del eje k. Una perturbación
        # en la dirección del eje k no provoca una salida del equilibrio.
    return lam


def equilibrium_analysis(I: ArrayLike, twoT: float) -> list[Equilibrium]:
    """Describe los equilibrios para cada una de las rotaciones puras en los ejes principales."""
    I = np.asarray(I, dtype=float)
    omegas = principal_equilibria(I, twoT)
    lam = analytic_eigenvalues(I, twoT)
    result: list[Equilibrium] = []
    for k, (omega, eigvals) in enumerate(zip(omegas, lam)): # Para cada eje k, con su omega y autovalores:
        # Análisis lineal: (0, mu, -mu). Traza nula, así que no hay nodos ni focos (solo centros o puntos de silla).
        small = 1e-9 * max(1.0, np.max(np.abs(eigvals)))  
        if np.all(np.abs(eigvals) <= small):
            kind, stability = "degenerado (todos los autovalores nulos)", "marginal"
        elif np.any(np.abs(eigvals.real) > small): # Si existe algún autovalor con parte real significativa.
            kind, stability = "punto de silla", "inestable"
        else:                                      # Si todos los autovalores tienen parte real despreciable.
            kind, stability = "centro", "marginal"
        # Análisis no lineal: Lyapunov con T y L^2. Eje extremo estable, intermedio inestable, empate neutro.
        others = np.delete(I, k) # Se elimina el momento de este eje para comparar con los otros dos ejes.
        if twoT <= 0:                               
            nonlinear = "neutro"
        elif I[k] > others.max() or I[k] < others.min():
            nonlinear = "estable"
        elif others.min() < I[k] < others.max():
            nonlinear = "inestable"
        else:
            nonlinear = "neutro" # Sólido simétrico (dos momentos de inercia iguales).                   
        result.append({"omega": omega, "eigvals": eigvals, "type": kind,
                       "stability": stability, "nonlinear": nonlinear})
    return result

# ================ [2] SEPARATRIZ ================ 

SEPARATRIX_TOL = 1e-3 # Tolerancia para considerar que una órbita está sobre la separatriz.      


def separatrix_side(I: ArrayLike, w0: ArrayLike) -> float:
    """Esta función nos indica cómo de lejos estamos de la separatriz, y en qué lado de la misma. Dicha separatriz 
    corresponde a |L|^2 = 2T*I_med. El valor devuelto se normaliza para que la tolerancia no dependa de T ni de I."""
    I = np.asarray(I, dtype=float)
    i_mid = sorted_axes(I)[1]
    L2_sep = 2 * kinetic_energy(w0, I) * I[i_mid]
    return float((angular_momentum_sq(w0, I) - L2_sep) / L2_sep) 


def surrounded_axis(I: ArrayLike, w0: ArrayLike, tol: float = SEPARATRIX_TOL) -> int | None:
    """Devuelve el eje (I_min, I_max) que rodea la órbita. Si devuelve None: órbita en separatriz, simétrico o reposo."""
    twoT = 2 * kinetic_energy(w0, I)
    if is_degenerate(I) or twoT <= 0:
        return None
    side = separatrix_side(I, w0)
    if abs(side) < tol:
        return None
    i_min, _, i_max = sorted_axes(I)
    return i_max if side > 0 else i_min

# ================ [3] AUTOVALORES CARACTERÍSTICOS  ================

def relevant_eigenvalues(I: ArrayLike, w0: ArrayLike) -> ComplexArray:
    """Autovalores no nulos del equilibrio que rodea la órbita o, en separatriz o simétrico, del eje intermedio."""
    k = surrounded_axis(I, w0)
    if k is None:
        k = sorted_axes(I)[1]
    lam: ComplexArray = analytic_eigenvalues(I, 2 * float(kinetic_energy(w0, I)))[k]
    return lam[np.abs(lam) > 1e-12]


def dominant_eigenvalue(I: ArrayLike, twoT: float, k:int) -> complex:
    """Se toma el autovalor que dicta la estabilidad del equilibrio k con la energía 2T. +i*beta en un centro
    (ambos son complejos conjugados, describen oscilación), y +mu real en una silla (amplifica la perturbación)."""
    return complex(analytic_eigenvalues(I, twoT)[k][1]) # Columna 1 es la raiz dominante.

# ================ [4] RÉGIMEN Y ESTABILIDAD DE LA ROTACIÓN ================

FLIP_DEG = 90.0 # Desviación máxima de omega (grados) a partir de la cual se considera que la rotación vuelca.


class RotationStability(TypedDict):
    """Estabilidad física de la rotación desde w0, medida sobre la órbita exacta."""
    regime: str            # El de rotation_regime.
    stability: str         # 'estable' | 'marginal' | 'inestable'.
    deviation: float       # Ángulo máximo entre omega(t) y w0, en grados (0 si es marginal).
    axis: int | None       # Eje que rodea la órbita (de surrounded_axis).


def rotation_regime(I: ArrayLike, w0: ArrayLike) -> str:
    """Régimen geométrico de la órbita: 'reposo', 'esfera', 'simetrico', 'separatriz', 'eje_mayor' o 'eje_menor'."""
    if 2 * kinetic_energy(w0, I) <= 0:
        return "reposo"
    if np.ptp(np.asarray(I, dtype=float)) < 1e-6:
        return "esfera"
    if is_degenerate(I):
        return "simetrico"
    side = separatrix_side(I, w0)
    if abs(side) < SEPARATRIX_TOL:
        return "separatriz"
    return "eje_mayor" if side > 0 else "eje_menor"


@lru_cache(maxsize=64)
def _max_deviation(I: tuple[float, ...], w0: tuple[float, ...]) -> float:
    """Versión con caché de max_deviation, con argumentos hashables."""
    orbit = EulerExact(I, w0).orbit(720)
    if orbit is None:
        return 0.0
    dot = np.sum(orbit * np.asarray(w0), axis=1)                 # Producto escalar con w0, sin BLAS.
    cos = dot / (np.linalg.norm(orbit, axis=1) * np.linalg.norm(w0))
    return float(np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0))))


def max_deviation(I: ArrayLike, w0: ArrayLike) -> float:
    """Ángulo máximo, en grados, entre omega(t) y omega0 a lo largo de la órbita exacta."""
    # Distingue el eje intermedio: allí llega a ~180 grados por pequeña que sea la perturbación.
    return _max_deviation(tuple(float(x) for x in np.ravel(I)), tuple(float(x) for x in np.ravel(w0)))


def rotation_stability(I: ArrayLike, w0: ArrayLike) -> RotationStability:
    """Estabilidad física de la rotación (ver RotationStability). Reposo, esfera y sólido simétrico son marginales:
    no hay eje intermedio. En el resto, vuelca (inestable) si omega se aleja FLIP_DEG o más de su dirección inicial."""
    regime = rotation_regime(I, w0)
    if regime in ("reposo", "esfera", "simetrico"):
        return {"regime": regime, "stability": "marginal", "deviation": 0.0, "axis": None}
    deviation = max_deviation(I, w0)
    return {"regime": regime, "stability": "inestable" if deviation >= FLIP_DEG else "estable",
            "deviation": deviation, "axis": surrounded_axis(I, w0)}
