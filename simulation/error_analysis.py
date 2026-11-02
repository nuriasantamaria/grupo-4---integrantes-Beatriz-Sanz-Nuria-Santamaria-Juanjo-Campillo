"""Error de los esquemas sobre el sólido: previsto por la teoría lineal y medido en el estudio de convergencia.
Une los invariantes, los autovalores y la solución exacta (space_physics) con R(z) y el ruido de redondeo 
(numerical_engine)."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from numerical_engine import Amplification, noise_floor
from space_physics import (Array, EulerExact, dominant_eigenvalue, equilibrium_analysis, is_degenerate,
                           kinetic_energy, surrounded_axis, angular_momentum_sq)

class LinearErrors(TypedDict):
    """Teoría lineal de un esquema en el centro que rodea la órbita actual (solo ese equilibrio, solo centros).
    En un paso con z = i*beta*dt, el paso exacto sería e^z (módulo 1, ángulo beta*dt); dmod y dphase miden cuánto
    se separa el esquema. Los otros equilibrios, sillas incluidas, se tratan en EquilibriumAmplification."""
    axis: int              # Eje del centro que rodea la órbita (0, 1, 2).
    beta: float            # Frecuencia de las pequeñas oscilaciones (rad/s), de los autovalores +-i*beta.
    R: complex             # Raíz r = R(z) de pi(r,z) = 0.
    dmod: float            # Error de módulo por paso, |R| - 1 por paso. <0 amortigua, >0 amplifica, =0 conserva.
    dphase: float          # Error de fase por paso, arg R - beta*dt, en rad. Mismo criterio de signos.
    period: float          # Periodo de las pequeñas oscilaciones, 2*pi/beta, en s.


class EquilibriumAmplification(TypedDict):
    """Teoría lineal de un esquema en cualquier equilibrio k (centro o silla), rodee o no a la órbita actual.
    Solo el módulo: |r| = |R(z)| con z = lam*dt, frente al exacto |e^(lam*dt)|. En un centro el exacto es 1 (si el
    centro es el de la órbita, coincide con LinearErrors). En una silla es e^(mu*dt) > 1: la perturbación crece por
    la física (inicio del vuelco), y |R| dice con qué fidelidad lo reproduce el esquema, esto es, el tiempo que la
    perturbación tarda en sacar al sistema de la zona lineal del equilibrio (menor: subestima; mayor: exagera)."""
    type: str              # "Centro" o "punto de silla" (de equilibrium_analysis).
    lam: complex           # Autovalor que dicta la estabilidad: +i*beta en un centro, +mu en una silla (de dominant_eigenvalue).
    absR: float            # |R(lam*dt)|: > 1 la perturbación crece en cada paso, < 1 se amortigua


class Convergence(TypedDict):
    """Error medido (no previsto): omega numérica en t = tf frente a la exacta (EulerExact), con dt0, dt0/2, dt0/4...
    Si el esquema es de orden p, err cae como dt^p: la pendiente log-log es el orden observado. A diferencia de
    Richardson, usa la solución exacta como referencia."""
    dt: Array                   # dt0 = tf/round(tf/dt), dt0/2, dt0/4, ... (n entero de pasos hasta t = tf).
    err: Array                  # |omega_num(tf) - omega_exacta(tf)| para cada dt (NaN si divergió).
    floor: float                # Ruido de redondeo, mismas unidades que err.
    ok: NDArray[np.bool_]       # Puntos fiables: finitos, sobre el ruido y sin saturar (err < 0.1*|omega0|)
    p: float                    # Orden observado, pendiente log-log de los fiables (NaN si hay menos de 2).
    local: Array                # Pendiente entre niveles consecutivos.


# ================ [1] DERIVA DE INVARIANTES ================

def conservation_drift(W: ArrayLike, I: ArrayLike, T0: float, L0: float) -> tuple[Array | float, Array | float]:
    """Deriva de T y de |L| a lo largo de la trayectoria W (n, 3), solo omega, respecto a T0 y L0 de
    conserved_quantities. Absoluta y con signo: > 0 el invariante crece, < 0 decrece. La solución exacta da 0."""
    dT = kinetic_energy(W, I) - T0
    dL = np.sqrt(angular_momentum_sq(W, I)) - L0
    return dT, dL

# ================ [2] TEORÍA LINEAL ================

def linear_step_errors(I: ArrayLike, w0: ArrayLike, amplification: Amplification, dt: float) -> LinearErrors | None:
    """Errores de módulo y fase por paso del esquema en el centro que rodea la órbita (ver LinearErrors).
    None si no hay tal centro: órbita en la separatriz, sólido simétrico o reposo."""
    axis = surrounded_axis(I, w0)
    if axis is None:
        return None
    twoT = 2 * float(kinetic_energy(w0, I))
    beta = dominant_eigenvalue(I, twoT, axis).imag # Positivo. En un centro, el autovalor dominante es +i*beta.
    R = complex(amplification(1j * beta * dt))     # Raíz r = R(z), con z = i*beta*dt   
    return {"axis": axis, "beta": beta, "R": R, "dmod": abs(R) - 1.0,
            "dphase": float(np.angle(R) - beta * dt), "period": 2 * np.pi / beta}


def predicted_errors(lin: LinearErrors, n: int) -> tuple[float, float]:
    """Errores previstos por la teoría lineal tras n pasos, comparables con los medidos (r, psi) de error_parts.
    r = |R|^n - 1: error relativo de amplitud (> 0 crece, < 0 decae); psi = n * dphase: error de fase
    acumulado, en grados (> 0 se adelanta, < 0 se retrasa)."""
    absR = abs(lin["R"])
    r = math.expm1(min(n * math.log(absR), 700.0)) if absR > 0 else -1.0 # e^x - 1 sin perder cifras, 700 evita overflow.
    return r, math.degrees(lin["dphase"]) * n


def equilibrium_amplification(I: ArrayLike, w0: ArrayLike, k: int, amplification: Amplification, 
                              dt: float) -> EquilibriumAmplification | None:
    """|R(lam*dt)| del esquema en el equilibrio k (0, 1 o 2, lo rodee o no la órbita) con la energía de w0
    (ver EquilibriumAmplification). None si no hay energía (reposo) o el sólido es simétrico."""
    twoT = 2 * float(kinetic_energy(w0, I))
    if twoT <= 0 or is_degenerate(I):
        return None
    kind = equilibrium_analysis(I, twoT)[k]["type"]
    lam = dominant_eigenvalue(I, twoT, k)
    return {"type": kind, "lam": lam, "absR": float(abs(amplification(lam * dt)))}

# ================ [3] CONVERGENCIA ================

def convergence_study(I: ArrayLike, w0: ArrayLike, integrate: Callable[[float, int], ArrayLike], tf: float, 
                      dt: float, levels: int = 5) -> Convergence:
    """Estudio de convergencia (ver Convergence): error de omega en t = tf frente a la solución exacta, y orden 
    observado. integrate(h, n): función que pasa quien llama (en la GUI, el esquema elegido). Dicha función
    entrega la trayectoria (n+1, >=3) con omega en las tres primeras columnas"""
    w_exact = EulerExact(I, w0).omega(tf)          # Referencia: omega exacta en tf (3,)
    n0 = max(1, int(round(tf / dt)))
    scale = max(float(np.linalg.norm(w0)), 1e-300) # Tamaño típico de omega (se evita el cero).

    # 1) Error en tf para dt0, dt0/2, dt0/4, ... (todos los niveles acaban justo en tf).
    n = n0 * 2 ** np.arange(levels)                # Vector con el número de pasos de cada nivel.
    dts = tf / n
    errs = np.empty(levels)
    with np.errstate(all="ignore"):                # Si un esquema diverge, su error es NaN (no da warning).
        for i in range(levels):
            w_num = np.asarray(integrate(dts[i], int(n[i])), dtype=float)[-1, :3] # Última fila.
            errs[i] = np.linalg.norm(w_num - w_exact) if np.all(np.isfinite(w_num)) else np.nan

    # 2) Puntos fiables: finitos, por encima del ruido de redondeo y por debajo de la saturación.
    floor = noise_floor(n[-1]) * scale
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(errs) & (errs > floor) & (errs < 0.1 * scale)

    # 3) Orden: pendiente de log(err) frente a log(dt), global y entre niveles consecutivos.
    p = np.polyfit(np.log(dts[ok]), np.log(errs[ok]), 1)[0] if ok.sum() >= 2 else np.nan # ax + b, [0] toma a.
    with np.errstate(all="ignore"):
        local = np.diff(np.log(errs)) / np.diff(np.log(dts)) # diff compara cada elemento del vector con el anterior.
    local[~(ok[:-1] & ok[1:])] = np.nan # Solo se consideran los intervalos entre puntos fiables.
    return {"dt": dts, "err": errs, "floor": floor, "ok": ok, "p": float(p), "local": local}
