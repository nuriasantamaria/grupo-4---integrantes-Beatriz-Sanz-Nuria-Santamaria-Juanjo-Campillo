"""Capa física del simulador:

  model.py       ecuaciones: inercia, Euler, invariantes T y L², campo f(u, t)
  exact.py       solución exacta: funciones elípticas de Jacobi y EulerExact
  equilibria.py  equilibrios, autovalores, estabilidad, lado de la separatriz y régimen de la rotación
"""

from space_physics.equilibria import (FLIP_DEG, SEPARATRIX_TOL, Equilibrium, RotationStability, analytic_eigenvalues,
                                equilibrium_analysis, max_deviation, principal_equilibria, rotation_regime,
                                rotation_stability, separatrix_side, surrounded_axis, relevant_eigenvalues,
                                dominant_eigenvalue)
from space_physics.exact import SEP_M, EulerExact, ellip_F, ellip_K, jacobi
from space_physics.model import (Array, angular_momentum_sq, conserved_quantities, euler_equations, inertia_matrix, 
                           initial_state, is_degenerate, jacobian_matrix, kinetic_energy, make_jacobian, make_rhs, 
                           omega_bound, rigid_body_jacobian, rigid_body_rhs, sorted_axes)
