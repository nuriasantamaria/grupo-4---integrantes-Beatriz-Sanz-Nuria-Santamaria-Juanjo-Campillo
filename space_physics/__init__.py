"""Capa física del simulador:

  model.py       ecuaciones: inercia, Euler, invariantes T y L², campo f(u, t)
  exact.py       solución exacta: funciones elípticas de Jacobi y EulerExact
  equilibria.py  equilibrios, autovalores, estabilidad y lado de la separatriz
  errors.py      error previsto y medido de cada esquema, ruido de redondeo y convergencia
"""

from space_physics.equilibria import (SEPARATRIX_TOL, Equilibrium, analytic_eigenvalues, equilibrium_analysis,
                                separatrix_side, surrounded_axis)
from space_physics.errors import (Convergence, EquilibriumAmplification, LinearErrors, conservation_drift,
                            convergence_study, equilibrium_amplification, linear_step_errors, noise_floor,
                            predicted_errors, relevant_eigenvalues)
from space_physics.exact import SEP_M, EulerExact, ellip_F, ellip_K, jacobi
from space_physics.model import (Amplification, Array, ArrayLike, ComplexArray, Vec, angular_momentum_sq,
                           conserved_quantities, euler_equations, inertia_matrix, initial_state, is_degenerate,
                           jacobian_matrix, kinetic_energy, make_jacobian, make_rhs, omega_bound, rigid_body_jacobian,
                           rigid_body_rhs, sorted_axes)
