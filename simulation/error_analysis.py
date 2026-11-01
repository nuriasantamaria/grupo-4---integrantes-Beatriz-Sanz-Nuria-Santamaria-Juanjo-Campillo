"""Error de los esquemas sobre el sólido: previsto por la teoría lineal y medido en el estudio de convergencia.

Une los autovalores y la solución exacta (`space_physics`) con R(z) y el ruido de redondeo (`numerical_engine`)."""

from __future__ import annotations

import math
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from numerical_engine import noise_floor
from space_physics import (Array, EulerExact, dominant_eigenvalue, equilibrium_analysis, is_degenerate,
                           kinetic_energy, surrounded_axis)


class LinearErrors(TypedDict):
    """Teoría lineal de un esquema en el centro que rodea la órbita, con z = i*beta*dt"""
    axis: int
    beta: float            # frecuencia de las pequeñas oscilaciones
    R: complex             # R(i*beta*dt)
    dmod: float            # |R| - 1 por paso
    dphase: float          # arg R - beta*dt por paso (rad)
    period: float          # 2*pi/beta


class EquilibriumAmplification(TypedDict):
    """|R(lambda*dt)| de un esquema en un equilibrio, resultado de equilibrium_amplification"""
    type: str              # tipo lineal del equilibrio
    lam: complex           # autovalor de mayor |Im| en un centro, de mayor Re en otro caso
    R: float


class Convergence(TypedDict):
    """Resultado de convergence_study"""
    dt: Array                   # dt0, dt0/2, dt0/4, ...
    err: Array                  # error en T para cada dt (NaN si divergió)
    floor: float                # ruido de redondeo
    ok: npt.NDArray[np.bool_]   # puntos fiables: finitos, sobre el ruido y sin saturar
    p: float                    # orden observado, pendiente log-log de los fiables (NaN si hay menos de 2)
    local: Array                # pendiente entre niveles consecutivos


# ============================================================== LINEAR THEORY


def linear_step_errors(I, w0, amplification, dt):
    """Errores por paso de módulo y fase del esquema en el centro que rodea la órbita; None si no hay centro"""
    axis = surrounded_axis(I, w0)
    if axis is None:
        return None
    twoT = 2 * float(kinetic_energy(w0, I))
    beta = abs(dominant_eigenvalue(I, twoT, axis).imag)    # frecuencia de las pequeñas oscilaciones
    if beta <= 0:
        return None
    R = complex(amplification(1j * beta * dt))
    return {"axis": axis, "beta": beta, "R": R, "dmod": abs(R) - 1.0,
            "dphase": float(np.angle(R) - beta * dt), "period": 2 * np.pi / beta}


def equilibrium_amplification(I, w0, k, amplification, dt):
    """|R(lambda*dt)| en el equilibrio k con la energía de w0; None si no hay energía o es simétrico"""
    twoT = 2 * float(kinetic_energy(w0, I))
    if twoT <= 0 or is_degenerate(I):
        return None
    kind = equilibrium_analysis(I, twoT)[k]["type"]
    lam = dominant_eigenvalue(I, twoT, k)
    return {"type": kind, "lam": lam, "R": float(abs(amplification(lam * dt)))}


def predicted_errors(lin, n):
    """(r, psi en grados) previstos por la teoría lineal tras n pasos: r = |R|^n - 1, psi = n * dphase"""
    R = abs(lin["R"])
    r = math.expm1(min(n * math.log(R), 700.0)) if R > 0 else -1.0     # expm1(x) = e^x - 1 sin perder cifras
    return r, math.degrees(lin["dphase"]) * n


# ============================================================== CONVERGENCE

def convergence_study(I, w0, integrate, T, dt, levels=5):
    """Error en T con dt0 = T/round(T/dt), dt0/2, dt0/4, ... y orden observado

    integrate(h, n) devuelve la trayectoria (n+1, >=3) con omega en las tres primeras columnas"""
    w_exact = EulerExact(I, w0).omega(np.array([float(T)]))[0]       # referencia: omega exacta en T
    n0 = max(1, int(round(T / dt)))
    scale = max(float(np.linalg.norm(w0)), 1e-300)                     # tamaño típico de omega

    # 1) error en T para dt0, dt0/2, dt0/4, ...
    n = n0 * 2 ** np.arange(levels)
    dts = T / n
    errs = np.empty(levels)
    with np.errstate(all="ignore"):
        for k in range(levels):
            w_num = np.asarray(integrate(dts[k], int(n[k])), dtype=float)[-1, :3]
            errs[k] = np.linalg.norm(w_num - w_exact) if np.all(np.isfinite(w_num)) else np.nan

    # 2) puntos fiables: finitos, por encima del ruido de redondeo y sin saturar
    floor = noise_floor(n[-1]) * scale
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(errs) & (errs > floor) & (errs < 0.1 * scale)

    # 3) orden: pendiente de log(err) frente a log(dt), global y entre niveles consecutivos
    p = np.polyfit(np.log(dts[ok]), np.log(errs[ok]), 1)[0] if ok.sum() >= 2 else np.nan
    with np.errstate(all="ignore"):
        local = np.diff(np.log(errs)) / np.diff(np.log(dts))
    local[~(ok[:-1] & ok[1:])] = np.nan
    return {"dt": dts, "err": errs, "floor": floor, "ok": ok, "p": float(p), "local": local}
