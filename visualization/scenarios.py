"""Escenarios predefinidos y configuración inicial de las simulaciones (sin Qt)"""

from __future__ import annotations

from typing import NamedTuple, TypedDict

from simulation import SimConfig

from visualization.palette import SIM_COLORS
from visualization.schemes import M_CN, M_EULER, M_RK4


# ---- escenarios predefinidos

class ScenarioSim(TypedDict, total=False):
    """Una simulación de un escenario; lo que falta se deja como está"""
    enabled: bool
    method: str
    dt: float
    I: tuple[float, float, float]
    w0: tuple[float, float, float]


class Scenario(NamedTuple):
    """Escenario didáctico: nota, simulaciones A, B y C, y si los errores van en escala logarítmica"""
    note: str
    sims: tuple[ScenarioSim, ...]
    log_errors: bool = False


_I123 = (1.0, 2.0, 3.0)


_JANI_W0 = (0.01, 2.0, 0.01)    # rotación casi pura sobre el eje intermedio


_MIN_W0 = (2.0, 0.3, 0.3)       # rotación sobre el eje de menor inercia


def _sim(method: str, dt: float, w0: tuple[float, float, float], I: tuple[float, float, float] = _I123,
         enabled: bool = True) -> ScenarioSim:
    return ScenarioSim(enabled=enabled, method=method, dt=dt, I=I, w0=w0)


SCENARIOS: dict[str, Scenario] = {
    "Comparar integradores (mismo Δt)": Scenario(
        "Mismo problema y mismo Δt con Euler, Crank-Nicolson y RK4: compara cómo derivan T y ‖L‖.",
        tuple(_sim(m, 0.02, _JANI_W0) for m in ("euler", "cn", "rk4")), log_errors=True),
    "Efecto Janibekov (eje intermedio)": Scenario(
        "Rotación casi pura sobre el eje intermedio con una pequeña perturbación: observa el vuelco.",
        (_sim("rk4", 0.01, _JANI_W0), _sim("cn", 0.01, _JANI_W0, enabled=False),
         _sim("euler", 0.01, _JANI_W0, enabled=False))),
    "Estable: eje de menor inercia": Scenario(
        "Rotación sobre el eje de menor inercia: las órbitas próximas lo rodean.",
        (_sim("rk4", 0.01, (2.0, 0.08, 0.08)), ScenarioSim(enabled=False), ScenarioSim(enabled=False))),
    "Estable: eje de mayor inercia": Scenario(
        "Rotación sobre el eje de mayor inercia: las órbitas próximas lo rodean.",
        (_sim("rk4", 0.01, (0.08, 0.08, 2.0)), ScenarioSim(enabled=False), ScenarioSim(enabled=False))),
    "Separatriz: un lado y el otro": Scenario(
        "Dos Ω₀ casi iguales a ambos lados de la separatriz: una órbita rodea el eje mayor y la otra el menor.",
        (_sim("rk4", 0.01, (0.85, 1.0, 0.5)), _sim("rk4", 0.01, (0.88, 1.0, 0.5)), ScenarioSim(enabled=False))),
    "Comparar inercias (mismo Ω₀)": Scenario(
        "Mismo Ω₀ con tres cuerpos distintos (B y C con física propia): cambia la órbita con la I₂.",
        (_sim("rk4", 0.01, _JANI_W0), _sim("rk4", 0.01, _JANI_W0, (1.0, 2.7, 3.0)),
         _sim("rk4", 0.01, _JANI_W0, (1.0, 1.3, 3.0)))),
    "Euler vs Crank-Nicolson vs RK4 (Δt grande)": Scenario(
        "Los tres esquemas con Δt = 0.08 sobre el mismo problema: observa cuál se desvía antes.",
        tuple(_sim(m, 0.08, _JANI_W0) for m in ("euler", "cn", "rk4")), log_errors=True),
    "Euler explícito vs implícito vs C-N (Δt grande)": Scenario(
        "Euler explícito, implícito y C-N con Δt = 0.08 girando sobre el eje de menor inercia: "
        "observa si la energía crece, decae o se conserva.",
        tuple(_sim(m, 0.08, _MIN_W0) for m in ("euler", "be", "cn")), log_errors=True),
}


DEFAULT_SCENARIO = "Comparar integradores (mismo Δt)"


DEFAULT_CONFIGS = [
    SimConfig("A", SIM_COLORS[0], True, M_EULER),
    SimConfig("B", SIM_COLORS[1], True, M_CN),
    SimConfig("C", SIM_COLORS[2], True, M_RK4),
]
