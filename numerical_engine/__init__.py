"""Capa algorítmica, genérica (no sabe nada del sólido):

  numerical_engine.py  Newton, esquemas de un paso con su R(z), integración y ruido de redondeo
  richardson.py        orden observado, error estimado y extrapolación de Richardson
  stability.py         regiones de estabilidad absoluta
"""

from numerical_engine.numerical_engine import *  # noqa: F401,F403
from numerical_engine.richardson import *  # noqa: F401,F403
from numerical_engine.stability import *  # noqa: F401,F403
