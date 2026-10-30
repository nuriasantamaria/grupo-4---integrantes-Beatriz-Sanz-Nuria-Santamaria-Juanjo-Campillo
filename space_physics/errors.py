"""¿Cuánto se equivoca el esquema? Error previsto (teoría lineal), medido (deriva, convergencia) y ruido de redondeo"""

import math
from typing import TypedDict

import numpy as np
import numpy.typing as npt

from space_physics.equilibria import analytic_eigenvalues, equilibrium_analysis, surrounded_axis
from space_physics.exact import EulerExact
from space_physics.model import Array, angular_momentum_sq, is_degenerate, kinetic_energy, sorted_axes


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


# ============================================================== DRIFT (error medido en los invariantes)

def conservation_drift(W, I, T0, L0):
    """(error de T, error de |L|) de la trayectoria W (n, 3) respecto a T0 y L0 (de conserved_quantities)"""
    dT = kinetic_energy(W, I) - T0
    dL = np.sqrt(angular_momentum_sq(W, I)) - L0
    return dT, dL


# ============================================================== LINEAR THEORY

def _dominant_eigenvalue(I, w0, k):
    """(tipo, λ) del equilibrio k con la energía de w0: en un centro el de mayor |Im|, si no el de mayor Re"""
    twoT = 2 * float(kinetic_energy(w0, I))
    if twoT <= 0 or is_degenerate(I):
        return None
    e = equilibrium_analysis(I, twoT)[k]
    lam = e["eigvals"]
    if e["type"] == "centro":
        return e["type"], complex(lam[np.argmax(np.abs(lam.imag))])
    return e["type"], complex(lam[np.argmax(lam.real)])


def linear_step_errors(I, w0, amplification, dt):
    """Errores por paso de módulo y fase del esquema en el centro que rodea la órbita; None si no hay centro"""
    axis = surrounded_axis(I, w0)
    if axis is None:
        return None
    kind, lam = _dominant_eigenvalue(I, w0, axis)
    beta = abs(lam.imag)                              # frecuencia de las pequeñas oscilaciones
    if kind != "centro" or beta <= 0:
        return None
    R = complex(amplification(1j * beta * dt))
    return {"axis": axis, "beta": beta, "R": R, "dmod": abs(R) - 1.0,
            "dphase": float(np.angle(R) - beta * dt), "period": 2 * np.pi / beta}


def equilibrium_amplification(I, w0, k, amplification, dt):
    """|R(λ·dt)| en el equilibrio k con la energía de w0; None si no hay energía o es simétrico"""
    found = _dominant_eigenvalue(I, w0, k)
    if found is None:
        return None
    kind, lam = found
    return {"type": kind, "lam": lam, "R": float(abs(amplification(lam * dt)))}


def relevant_eigenvalues(I, w0):
    """Autovalores no nulos del equilibrio que rodea la órbita o, en separatriz o simétrico, del eje intermedio"""
    k = surrounded_axis(I, w0)
    if k is None:
        k = sorted_axes(I)[1]
    lam = analytic_eigenvalues(I, 2 * float(kinetic_energy(w0, I)))[k]
    return lam[np.abs(lam) > 1e-12]


def noise_floor(n_steps):
    """Error de módulo indistinguible del redondeo acumulado tras n_steps pasos (crece como √n)"""
    eps = np.finfo(float).eps                          # precisión relativa de un float (2.2e-16)
    return float(50.0 * eps * math.sqrt(max(n_steps, 1)))


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
