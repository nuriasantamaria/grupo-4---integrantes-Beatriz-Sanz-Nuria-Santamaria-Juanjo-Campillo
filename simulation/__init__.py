"""Capa de simulación: une la física (`space_physics`) y los esquemas (`numerical_engine`) sin depender de Qt

  simulation.py      sesión de simulación (Simulation), previsualización, Richardson y convergencia
  divergence.py      detección de divergencia numérica a partir del módulo de ω
  error_analysis.py  error previsto por la teoría lineal y estudio de convergencia
"""

from simulation import divergence
from simulation.error_analysis import *  # noqa: F401,F403
from simulation.simulation import *  # noqa: F401,F403
