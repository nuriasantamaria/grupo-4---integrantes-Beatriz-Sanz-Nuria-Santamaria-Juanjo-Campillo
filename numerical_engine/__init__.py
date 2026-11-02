"""Capa algorítmica, genérica (no sabe nada del sólido):

  numerical_engine.py  tipos, Newton, esquemas de un paso y ruido de redondeo
  stability.py         R(z) de cada esquema sacada de su paso
  schemes.py           registro de esquemas SCHEMES, con sus propiedades teóricas
  integration.py       problema de Cauchy: cauchy_problem, integrate, n_steps y final_state
  richardson.py        orden observado, error estimado y extrapolación de Richardson
"""

from numerical_engine.numerical_engine import *  # noqa: F401,F403
from numerical_engine.stability import *  # noqa: F401,F403
from numerical_engine.schemes import *  # noqa: F401,F403
from numerical_engine.integration import *  # noqa: F401,F403
from numerical_engine.richardson import *  # noqa: F401,F403
