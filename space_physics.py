"""Física del sólido rígido libre (ecuaciones de Euler) y herramientas para analizarla.

Contenido
  · Ecuaciones del problema, invariantes (2T, ‖L‖²), cinemática del cuaternión.
  · Equilibrios: Jacobiana, autovalores analíticos y clasificación (lineal y no lineal).
  · Geometría del espacio de fases: elipsoides de Poinsot, polhodias, separatriz, plano de fases.
  · Solución exacta (funciones elípticas de Jacobi) y descomposición del error en módulo y fase.
  · Escenarios predefinidos (`SCENARIOS`) y reglas de edición de Ω₀ por ejes.
  · Análisis lineal de un esquema numérico en el centro que rodea la órbita, relevancia del error
    frente al ruido de redondeo y estudio de convergencia en Δt.

Convenios
  · `I` es el vector (I₁, I₂, I₃) de momentos principales; solo `euler_equations`, `jacobian_matrix` y
    `rigid_body_rhs` reciben la matriz diagonal 3×3 (`I_mat`).
  · Los vectores ω van en ejes del cuerpo; una trayectoria es un array (n, 3).
  · No depende de numerical_engine: los esquemas se reciben como funciones R(z) o como `integrate`."""

from __future__ import annotations

import math
from functools import lru_cache, partial
from typing import Callable, NamedTuple, Sequence, TypedDict, Union

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
ComplexArray = npt.NDArray[np.complex128]
ArrayLike = npt.ArrayLike
Vec = Union[Array, Sequence[float]]                    # algo que se puede indexar: ω, I, Ω₀
Amplification = Callable[[complex], complex]          # R(z) de un esquema


# ============================================================== ecuaciones e invariantes
def euler_equations(omega: Vec, I: Array) -> Array:
    """ω̇ de I_i ω̇_i = (I_j − I_k) ω_j ω_k. `I` es la matriz diagonal 3×3 de inercia."""
    return np.array([
        (I[1, 1] - I[2, 2]) / I[0, 0] * omega[1] * omega[2],
        (I[2, 2] - I[0, 0]) / I[1, 1] * omega[2] * omega[0],
        (I[0, 0] - I[1, 1]) / I[2, 2] * omega[0] * omega[1],
    ])


def kinetic_energy(omega: ArrayLike, I: ArrayLike) -> Array | float:
    """T = ½ Σ I_i ω_i² (admite ω de forma (..., 3))."""
    return 0.5 * np.sum(np.array(I) * np.asarray(omega) ** 2, axis=-1)


def angular_momentum_sq(omega: ArrayLike, I: ArrayLike) -> Array | float:
    """‖L‖² = Σ (I_i ω_i)² (admite ω de forma (..., 3))."""
    return np.sum((np.array(I) * np.asarray(omega)) ** 2, axis=-1)


def conserved_quantities(w0: ArrayLike, I: ArrayLike) -> tuple[float, float]:
    """(T, ‖L‖) del estado inicial: lo que debe conservar la solución."""
    return float(kinetic_energy(w0, I)), math.sqrt(float(angular_momentum_sq(w0, I)))


def conservation_drift(W: Array, I: Array, T0: float, L0: float) -> tuple[Array, Array]:
    """(ΔT, Δ‖L‖) de una trayectoria W (n, 3) respecto a los valores iniciales T0 y ‖L‖₀."""
    return np.asarray(kinetic_energy(W, I)) - T0, np.sqrt(angular_momentum_sq(W, I)) - L0


def omega_bound(I: ArrayLike, w0: ArrayLike) -> float:
    """Cota de ‖ω‖ sobre la órbita exacta: ‖ω‖² ≤ 2T / I_mín. Una trayectoria que la supera con
    creces ha dejado de seguir la dinámica (divergencia numérica)."""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    return max(math.sqrt(2.0 * float(kinetic_energy(w0, I)) / float(I.min())), float(np.linalg.norm(w0)))


# ============================================================== cuaterniones (orientación)
def quat_to_matrix(q: ArrayLike) -> Array:
    """Matriz de rotación cuerpo → inercial del cuaternión q = (w, x, y, z) (se normaliza)."""
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def attitude_vectors(q: ArrayLike, I: ArrayLike, w: ArrayLike, L0_norm: float,
                     w0_norm: float) -> tuple[Array, Array, Array]:
    """(R, L̂, ω̂) para dibujar la actitud: matriz de rotación cuerpo → inercial, momento angular y velocidad
    angular en ejes inerciales, normalizados con ‖L₀‖ y ‖ω₀‖ (el momento sale constante)."""
    R = quat_to_matrix(q)
    w = np.asarray(w, dtype=float)
    return R, R @ (np.asarray(I, dtype=float) * w) / max(L0_norm, 1e-12), R @ w / max(w0_norm, 1e-12)


def body_half_sizes(I: Vec, min_rel: float = 0.15) -> Array:
    """Semiejes relativos (máx = 1) de una caja homogénea con esos momentos de inercia
    (I₁ = m(b²+c²)/12 … ⇒ a² ∝ I₂ + I₃ − I₁); los que salen imposibles se acotan en `min_rel`."""
    I1, I2, I3 = (float(x) for x in I)
    s = np.sqrt(np.maximum([I2 + I3 - I1, I1 + I3 - I2, I1 + I2 - I3], 0.0))
    return np.asarray(np.maximum(s / s.max(), min_rel), dtype=float)


# ============================================================== cinemática del cuerpo y de Ω₀
def rigid_body_rhs(u: Array, t: float, I_mat: Array) -> Array:
    """Campo del sistema completo u = (ω, q): Euler + cinemática del cuaternión q' = ½ q ⊗ (0, ω)."""
    w1, w2, w3, q0, q1, q2, q3 = u
    dw = euler_equations(u[:3], I_mat)
    return np.array([dw[0], dw[1], dw[2],
                     0.5 * (-q1 * w1 - q2 * w2 - q3 * w3),
                     0.5 * (q0 * w1 + q2 * w3 - q3 * w2),
                     0.5 * (q0 * w2 + q3 * w1 - q1 * w3),
                     0.5 * (q0 * w3 + q1 * w2 - q2 * w1)])


def rigid_body_jacobian(u: Array, t: float, I_mat: Array) -> Array:
    """Jacobiana exacta ∂f/∂u (7×7) del campo `rigid_body_rhs`: bloque ∂ω̇/∂ω de Euler (`jacobian_matrix`),
    ∂q̇/∂ω y ∂q̇/∂q de la cinemática (que es bilineal en q y ω). Es la que usa Newton en los esquemas implícitos."""
    w1, w2, w3, q0, q1, q2, q3 = u
    J = np.zeros((7, 7))
    J[:3, :3] = jacobian_matrix(u[:3], I_mat)
    J[3:, :3] = 0.5 * np.array([[-q1, -q2, -q3], [q0, -q3, q2], [q3, q0, -q1], [-q2, q1, q0]])
    J[3:, 3:] = 0.5 * np.array([[0, -w1, -w2, -w3], [w1, 0, w3, -w2], [w2, -w3, 0, w1], [w3, w2, -w1, 0]])
    return J


def _inertia_matrix(I: ArrayLike) -> Array:
    """diag(I) validando que los tres momentos principales sean finitos y positivos (ValueError si no)."""
    I = np.asarray(I, dtype=float)
    if I.shape != (3,) or not np.all(np.isfinite(I)) or np.any(I <= 0):
        raise ValueError(f"I debe tener tres momentos principales positivos (recibido {I.tolist()})")
    return np.diag(I)


def make_jacobian(I: ArrayLike) -> Callable[[Array, float], Array]:
    """J(u, t) = ∂f/∂u del sólido I, el par exacto de `make_rhs`."""
    return partial(rigid_body_jacobian, I_mat=_inertia_matrix(I))


def make_rhs(I: ArrayLike) -> Callable[[Array, float], Array]:
    """f(u, t) del sólido de momentos principales I, listo para un integrador de numerical_engine."""
    return partial(rigid_body_rhs, I_mat=_inertia_matrix(I))


def initial_state(w0: ArrayLike) -> Array:
    """Estado inicial (ω₀, q = 1, 0, 0, 0) del sistema completo."""
    return np.array([*np.asarray(w0, dtype=float), 1.0, 0.0, 0.0, 0.0])


def axis_decomposition(w: Sequence[float], keep: tuple[int, ...] | None = None) -> tuple[int, int, int]:
    """(principal, perturbación 1, perturbación 2): el eje de mayor |ω| y, de los otros dos, el mayor
    primero. Si ω = 0 se conserva el eje principal de `keep`."""
    a = keep[0] if (max(abs(x) for x in w) < 1e-12 and keep is not None) else int(np.argmax([abs(x) for x in w]))
    b, c = sorted((k for k in range(3) if k != a), key=lambda k: (-abs(w[k]), k))
    return a, b, c


def omega_from_axis(axes: tuple[int, int, int], main: float, p1: float, p2: float = 0.0) -> list[float]:
    """Ω₀ = Ω sobre el eje principal + perturbaciones ε₁, ε₂ sobre los otros dos."""
    a, b, c = axes
    w = [0.0] * 3
    w[a], w[b], w[c] = main, p1, p2
    return w


def reassign_axes(axes: tuple[int, int, int], new_main: int | None = None,
                  new_p1: int | None = None) -> tuple[int, int, int]:
    """Nueva terna (principal, perturbación 1, perturbación 2) al elegir otro eje principal o otro eje para la
    primera perturbación. Si el nuevo principal era una perturbación, el antiguo principal ocupa su sitio."""
    a, b, c = axes
    if new_main is not None and new_main != a:
        rest = [k for k in range(3) if k != new_main]
        b2 = b if b in rest else a
        return new_main, b2, next(k for k in rest if k != b2)
    if new_p1 is not None and new_p1 != b:
        return a, new_p1, next(k for k in range(3) if k not in (a, new_p1))
    return a, b, c


def inertia_ranks(I: ArrayLike, tol: float = 1e-9) -> list[str]:
    """Etiqueta de cada momento: 'mín' / 'medio' / 'máx', o '=' si empata con otro (sólido simétrico)."""
    I = np.asarray(I, dtype=float)
    tags = [""] * 3
    for rank, k in enumerate(np.argsort(I, kind="stable")):
        tags[k] = ("mín", "medio", "máx")[rank]
    for k in range(3):
        if any(j != k and abs(I[j] - I[k]) < tol for j in range(3)):
            tags[k] = "="
    return tags


def triangle_violation(I: ArrayLike, tol: float = 1e-9) -> int | None:
    """Índice k con I_k > I_i + I_j (ningún sólido real lo tiene: desigualdad triangular), o None."""
    I = np.asarray(I, dtype=float)
    return next((k for k in range(3) if I[k] > I[(k + 1) % 3] + I[(k + 2) % 3] + tol), None)


def same_problem(I_a: ArrayLike, w_a: ArrayLike, I_b: ArrayLike, w_b: ArrayLike) -> bool:
    """¿Misma I y mismo Ω₀ (es decir, misma solución exacta)?"""
    return bool(np.allclose(I_a, I_b, rtol=1e-12, atol=0) and np.allclose(w_a, w_b, rtol=1e-12, atol=1e-15))


# ============================================================== escenarios predefinidos
class ScenarioSim(TypedDict, total=False):
    """Una simulación de un escenario; lo que falte se deja como está. `method` es una clave de
    numerical_engine.SCHEMES ('euler', 'be', 'cn', 'rk4')."""
    enabled: bool
    method: str
    dt: float
    I: tuple[float, float, float]
    w0: tuple[float, float, float]


class Scenario(NamedTuple):
    """Escenario didáctico: nota explicativa, simulaciones A, B, C y si conviene mostrar los errores en escala
    logarítmica (esquemas de órdenes de magnitud muy distintos)."""
    note: str
    sims: tuple[ScenarioSim, ...]
    log_errors: bool = False


_I123 = (1.0, 2.0, 3.0)
_JANI_W0 = (0.01, 2.0, 0.01)              # rotación casi pura sobre el eje intermedio
_MIN_W0 = (2.0, 0.3, 0.3)                 # rotación sobre el eje de menor inercia


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


# ============================================================== equilibrios y su estabilidad
class Equilibrium(TypedDict):
    """Un equilibrio ω = Ω eₖ del sólido libre y su estabilidad (`equilibrium_analysis`)."""
    omega: Array           # rotación pura ω = Ω eₖ (Ω² = 2T/Iₖ)
    eigvals: ComplexArray  # autovalores de la Jacobiana en ese punto
    type: str              # clasificación lineal (centro, silla, degenerado…)
    stability: str         # 'estable' | 'marginal' | 'inestable' según la linealización
    nonlinear: str         # 'estable' | 'inestable' | 'neutro' con T y ‖L‖² (más fiable que la lineal)


def jacobian_matrix(omega: Vec, I: Array) -> Array:
    """Jacobiana ∂ω̇/∂ω de las ecuaciones de Euler (`I` matriz diagonal). Define los autovalores;
    `analytic_eigenvalues` da su valor exacto en las rotaciones puras."""
    return np.array([
        [0, ((I[1, 1] - I[2, 2]) / I[0, 0]) * omega[2], ((I[1, 1] - I[2, 2]) / I[0, 0]) * omega[1]],
        [((I[2, 2] - I[0, 0]) / I[1, 1]) * omega[2], 0, ((I[2, 2] - I[0, 0]) / I[1, 1]) * omega[0]],
        [((I[0, 0] - I[1, 1]) / I[2, 2]) * omega[1], ((I[0, 0] - I[1, 1]) / I[2, 2]) * omega[0], 0],
    ])


def principal_equilibria(I: ArrayLike, twoT: float) -> Array:
    """Fila k = rotación pura sobre el eje principal k con energía 2T: ω = √(2T/Iₖ) eₖ."""
    I = np.asarray(I, dtype=float)
    eq = np.zeros((3, 3))
    for k in range(3):
        eq[k, k] = np.sqrt(twoT / I[k])
    return eq


def analytic_eigenvalues(I: ArrayLike, twoT: float) -> ComplexArray:
    """Autovalores de la Jacobiana en cada rotación pura (fila k = equilibrio sobre el eje k).

    En ω = Ω_k e_k, con Ω_k² = 2T/I_k, son λ = 0 (a lo largo de e_k) y
    λ² = Ω_k² (I_k − I_i)(I_j − I_k) / (I_i I_j), con (i, j, k) cíclico: imaginarios puros
    (centro) en los ejes extremos y reales ±μ (silla) en el intermedio."""
    I = np.asarray(I, dtype=float)
    lam = np.zeros((3, 3), dtype=complex)
    for k in range(3):
        i, j = (k + 1) % 3, (k + 2) % 3
        mu = np.sqrt(complex(twoT / I[k] * (I[k] - I[i]) * (I[j] - I[k]) / (I[i] * I[j])))
        lam[k] = [0, mu, -mu]
    return lam


def classify_equilibrium(eigvals: ArrayLike, tol: float = 1e-9) -> tuple[str, str]:
    """(tipo, estabilidad lineal) a partir de los autovalores de la Jacobiana."""
    lam = np.asarray(eigvals, dtype=complex)
    scale = max(1.0, float(np.max(np.abs(lam))))
    active = lam[np.abs(lam) > tol * scale]
    if active.size == 0:
        return "degenerado (todos los autovalores nulos)", "marginal"
    re = active.real
    complex_pair = np.any(np.abs(active.imag) > tol * scale)
    pos = np.any(re > tol * scale)
    neg = np.any(re < -tol * scale)
    if pos and neg:
        return "punto de silla", "inestable"
    if pos:
        return ("foco inestable" if complex_pair else "nodo inestable"), "inestable"
    if neg:
        return ("foco estable" if complex_pair else "nodo estable"), "estable"
    return "centro", "marginal"


def _nonlinear_stability(I: Array, k: int) -> str:
    """Estabilidad (de Lyapunov) de la rotación sobre el eje k: T y ‖L‖² acotan la órbita si I_k es el
    máximo o el mínimo estricto (estable); el eje intermedio es inestable; con empates, neutro."""
    others = np.delete(I, k)
    if np.all(I[k] > others) or np.all(I[k] < others):
        return "estable"
    return "inestable" if np.min(others) < I[k] < np.max(others) else "neutro"


def equilibrium_analysis(I: ArrayLike, twoT: float) -> list[Equilibrium]:
    """Los tres equilibrios ω = Ω_k e_k (y sus opuestos) con autovalores, tipo y estabilidad.
    Con I casi iguales (`is_degenerate`) el análisis sigue siendo exacto, pero los autovalores son tan
    pequeños que la frontera entre tipos (centro / silla) es numéricamente delicada. Con 2T = 0 no hay
    rotación y los tres equilibrios coinciden con el reposo: la estabilidad no lineal es 'neutro'."""
    I = np.asarray(I, dtype=float)
    lam, result = analytic_eigenvalues(I, twoT), []
    for k, omega in enumerate(principal_equilibria(I, twoT)):
        kind, stability = classify_equilibrium(lam[k])
        result.append(Equilibrium(omega=omega, eigvals=lam[k], type=kind, stability=stability,
                                  nonlinear="neutro" if twoT <= 0 else _nonlinear_stability(I, k)))
    return result


# ============================================================== geometría del espacio de fases
def sorted_axes(I: ArrayLike) -> tuple[int, int, int]:
    """Índices de los ejes de menor, intermedio y mayor momento de inercia."""
    a, b, c = np.argsort(np.asarray(I, dtype=float))
    return int(a), int(b), int(c)


def is_degenerate(I: ArrayLike, tol: float = 1e-6) -> bool:
    """¿Hay dos momentos iguales (sólido simétrico) o los tres (esfera)?"""
    Is = np.sort(np.asarray(I, dtype=float))
    return bool((Is[1] - Is[0] < tol) or (Is[2] - Is[1] < tol))


def poinsot_semiaxes(I: ArrayLike, w0: ArrayLike) -> tuple[Array, Array]:
    """Semiejes de los elipsoides de energía (√(2T/Iᵢ)) y de momento (‖L‖/Iᵢ) en el espacio de ω."""
    I = np.asarray(I, dtype=float)
    twoT = 2 * kinetic_energy(w0, I)
    L = np.sqrt(angular_momentum_sq(w0, I))
    return np.sqrt(twoT / I), L / I


def regime(I: ArrayLike, w0: ArrayLike) -> str:
    """Régimen geométrico de la órbita: reposo (ω = 0), esfera inercial (ω constante), sólido simétrico,
    sobre la separatriz, o precesión en torno al eje mayor/menor. No dice si es estable: de eso se encarga
    `orbit_stability`, que mira cuánto se aleja ω de su posición inicial."""
    twoT = 2 * kinetic_energy(w0, I)
    if twoT <= 0:
        return "reposo (ω = 0)"
    if np.ptp(np.asarray(I, dtype=float)) < 1e-6:
        return "esfera inercial"
    if is_degenerate(I):
        return "sólido simétrico"
    I = np.asarray(I, dtype=float)
    _, b, _ = sorted_axes(I)
    D = angular_momentum_sq(w0, I) - twoT * I[b]
    if abs(D) < 1e-3 * twoT * I[b]:
        return "sobre la separatriz"
    return "precesa en torno al eje mayor" if D > 0 else "precesa en torno al eje menor"


def surrounded_axis(I: ArrayLike, w0: ArrayLike) -> int | None:
    """Eje que rodea la órbita de ω (el de mayor o menor inercia), o None en separatriz / sólido simétrico."""
    twoT = 2 * kinetic_energy(w0, I)
    if is_degenerate(I) or twoT <= 0:
        return None
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    D = angular_momentum_sq(w0, I) - twoT * I[b]
    if abs(D) < 1e-3 * twoT * I[b]:
        return None
    return c if D > 0 else a


FLIP_DEG = 90.0     # desviación máxima del eje de giro a partir de la cual la rotación «vuelca»


@lru_cache(maxsize=64)
def _max_deviation(I: tuple[float, ...], w0: tuple[float, ...]) -> float:
    orbit = EulerExact(I, w0).orbit(720)
    if orbit is None:
        return 0.0
    cos = np.sum(orbit * np.asarray(w0), axis=1) / (np.linalg.norm(orbit, axis=1) * np.linalg.norm(w0))   # sin BLAS
    return float(np.degrees(np.arccos(np.clip(cos.min(), -1.0, 1.0))))


def max_deviation(I: ArrayLike, w0: ArrayLike) -> float:
    """Ángulo máximo (°) que forma ω(t) con su dirección inicial ω₀ a lo largo de la órbita exacta.

    Es lo que distingue el teorema del eje intermedio: girando casi sobre un eje de mayor o menor inercia,
    la desviación es del orden de la perturbación inicial; casi sobre el intermedio llega a ~180°
    (ω pasa al otro lado del eje) por pequeña que sea la perturbación."""
    return _max_deviation(tuple(float(x) for x in np.ravel(I)), tuple(float(x) for x in np.ravel(w0)))


def orbit_stability(I: ArrayLike, w0: ArrayLike) -> tuple[str, str]:
    """Estabilidad FÍSICA de la rotación (solo depende de I y Ω₀): (texto, 'estable'|'marginal'|'inestable').

    · estable: el eje de giro se queda cerca de su posición inicial (desviación máxima < FLIP_DEG);
    · inestable: la rotación vuelca, ω llega a más de FLIP_DEG de donde empezó (giro casi sobre el eje
      intermedio, o sobre la separatriz);
    · marginal: reposo, esfera o sólido simétrico (precesión uniforme, no hay eje intermedio)."""
    r = regime(I, w0)
    if "reposo" in r:
        return "en reposo · sin rotación", "marginal"
    if "esfera" in r:
        return "esfera inercial · ω constante", "marginal"
    if "simétrico" in r:
        return "sólido simétrico · precesión uniforme", "marginal"
    dev = max_deviation(I, w0)
    ax = surrounded_axis(I, w0)
    donde = " · sobre la separatriz" if ax is None else f" · rodea el eje {'mayor' if 'mayor' in r else 'menor'} e{'₁₂₃'[ax]}"
    if dev >= FLIP_DEG:
        return f"inestable · vuelca: ω llega a {dev:.0f}° de su dirección inicial{donde}", "inestable"
    return f"estable · ω se desvía como máximo {dev:.0f}° de su dirección inicial{donde}", "estable"


def separatrix_curves(I: ArrayLike, twoT: float, n_pts: int = 300) -> list[Array]:
    """Las dos curvas separatriz (n, 3) sobre el elipsoide de energía; [] en sólido simétrico o sin energía."""
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    k = math.sqrt(Ia * (Ib - Ia) / (Ic * (Ic - Ib)))
    s = np.linspace(0, 2 * np.pi, 4 * max(n_pts // 4, 1) + 1)     # 4m+1 puntos: incluye s = π/2 y 3π/2 (los equilibrios)
    n_pts = len(s)
    A = math.sqrt(twoT / (Ia + Ic * k * k))
    B = math.sqrt(twoT / Ib)
    curves = []
    for sign in (+1, -1):
        P = np.zeros((n_pts, 3))
        P[:, a] = A * np.cos(s)
        P[:, b] = B * np.sin(s)
        P[:, c] = sign * k * A * np.cos(s)
        curves.append(P)
    return curves


def polhode_family(I: ArrayLike, twoT: float, n_levels: int = 5, n_pts: int = 200) -> list[Array]:
    """Familia de polhodias (órbitas de ω sobre el elipsoide de energía) alrededor de los ejes extremos."""
    if is_degenerate(I) or twoT <= 0:
        return []
    I = np.asarray(I, dtype=float)
    a, b, c = sorted_axes(I)
    Ia, Ib, Ic = I[a], I[b], I[c]
    s = np.linspace(0, 2 * np.pi, n_pts)
    curves = []
    levels = np.linspace(0.15, 0.9, n_levels)

    for f in levels:
        L2 = twoT * (Ia + f * (Ib - Ia))
        wc = math.sqrt((L2 - twoT * Ia) / (Ic * (Ic - Ia))) * np.cos(s)
        wb = math.sqrt((L2 - twoT * Ia) / (Ib * (Ib - Ia))) * np.sin(s)
        wa = np.sqrt(np.maximum(twoT - Ib * wb ** 2 - Ic * wc ** 2, 0) / Ia)
        for sign in (+1, -1):
            P = np.zeros((n_pts, 3))
            P[:, a], P[:, b], P[:, c] = sign * wa, wb, wc
            curves.append(P)

    for f in levels:
        L2 = twoT * (Ic - f * (Ic - Ib))
        wa = math.sqrt((twoT * Ic - L2) / (Ia * (Ic - Ia))) * np.cos(s)
        wb = math.sqrt((twoT * Ic - L2) / (Ib * (Ic - Ib))) * np.sin(s)
        wc = np.sqrt(np.maximum(twoT - Ia * wa ** 2 - Ib * wb ** 2, 0) / Ic)
        for sign in (+1, -1):
            P = np.zeros((n_pts, 3))
            P[:, a], P[:, b], P[:, c] = wa, wb, sign * wc
            curves.append(P)
    return curves


# ---------------------------------------------------------------- plano de fases (carta alrededor de un eje)
def chart_axes(k: int) -> tuple[int, int]:
    """Ejes (i, j) de la carta ω_i–ω_j sobre el hemisferio ω_k ≷ 0."""
    return (k + 1) % 3, (k + 2) % 3


def default_chart(I: ArrayLike, w0: Vec) -> tuple[int, int]:
    """(eje k, signo) de la carta natural: la del eje que rodea la órbita (o el intermedio si no hay)
    y el hemisferio en el que está ω₀."""
    k = surrounded_axis(I, w0)
    if k is None:
        k = sorted_axes(I)[1]
    return k, (1 if w0[k] >= 0 else -1)


def project_to_chart(W: Array, k: int, sign: int) -> Array:
    """Proyecta la trayectoria W (n, 3) sobre la carta (ω_i, ω_j); NaN donde ω_k tiene el signo contrario."""
    i, j = chart_axes(k)
    P = np.asarray(W)[:, [i, j]].astype(float)
    P[~(sign * np.asarray(W)[:, k] > 0)] = np.nan
    return P


def chart_domain(I: ArrayLike, twoT: float, k: int, n_pts: int = 300) -> Array:
    """Borde (n, 2) de la región admisible de la carta: la elipse I_i ω_i² + I_j ω_j² = 2T."""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    s = np.linspace(0, 2 * np.pi, n_pts)
    return np.column_stack([math.sqrt(twoT / I[i]) * np.cos(s), math.sqrt(twoT / I[j]) * np.sin(s)])


def reduced_flow(I: ArrayLike, twoT: float, k: int, sign: int, X: ArrayLike, Y: ArrayLike) -> tuple[Array, Array]:
    """(ω̇_i, ω̇_j) del flujo en la carta, con ω_k = sign·√((2T − I_iω_i² − I_jω_j²)/I_k)."""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    X, Y = np.asarray(X, dtype=float), np.asarray(Y, dtype=float)
    W = np.zeros((3,) + X.shape)
    W[i], W[j] = X, Y
    W[k] = sign * np.sqrt(np.maximum(twoT - I[i] * X ** 2 - I[j] * Y ** 2, 0) / I[k])
    dW = euler_equations(W, np.diag(I))
    return dW[i], dW[j]


def reduced_flow_field(I: ArrayLike, twoT: float, k: int, sign: int, n: int = 21,
                       margin: float = 0.96) -> tuple[Array, Array, Array, Array, float]:
    """Campo de direcciones del plano de fases: (X, Y, U, V, paso) en una malla n×n del dominio de la carta
    (solo los puntos con I_iω_i² + I_jω_j² < margen·2T). `paso` es la separación de la malla."""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    D = chart_domain(I, twoT, k)
    xm, ym = float(D[:, 0].max()), float(D[:, 1].max())
    X, Y = np.meshgrid(np.linspace(-xm, xm, n), np.linspace(-ym, ym, n))
    inside = I[i] * X ** 2 + I[j] * Y ** 2 < margin * twoT
    U, V = reduced_flow(I, twoT, k, sign, X[inside], Y[inside])
    return X[inside], Y[inside], U, V, min(2 * xm, 2 * ym) / (n - 1)


def phase_plane_orbits(I: ArrayLike, twoT: float, k: int, n_levels: int = 9,
                       n_pts: int = 400) -> tuple[list[Array], list[Array]]:
    """(órbitas, separatrices) de la carta: curvas de nivel de a·ω_i² + b·ω_j² = c."""
    if is_degenerate(I) or twoT <= 0:
        return [], []
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    fr = np.linspace(0.12, 1.0, n_levels) ** 2
    qa, qb = a * twoT / I[i], b * twoT / I[j]
    if a * b > 0:
        levels = fr * (qa if abs(qa) > abs(qb) else qb)
        seps: list[Array] = []
    else:
        levels = np.concatenate([fr * qa, fr * qb])
        seps = chart_level_curve(I, twoT, k, 0.0, n_pts)
    orbits = [P for c in levels for P in chart_level_curve(I, twoT, k, c, n_pts)]
    return orbits, seps


def chart_level_curve(I: ArrayLike, twoT: float, k: int, c: float, n_pts: int = 400) -> list[Array]:
    """Curva(s) (n, 2) del nivel a·ω_i² + b·ω_j² = c recortadas al dominio de la carta."""
    I = np.asarray(I, dtype=float)
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    xm, ym = math.sqrt(twoT / I[i]), math.sqrt(twoT / I[j])

    def clip(x: Array, y: Array) -> Array:
        P = np.column_stack([x, y]).astype(float)
        P[I[i] * P[:, 0] ** 2 + I[j] * P[:, 1] ** 2 > twoT * (1 + 1e-9)] = np.nan
        return P

    if a * b > 0:
        if c / a <= 0:
            return []
        s = np.linspace(0, 2 * np.pi, n_pts)
        return [clip(math.sqrt(c / a) * np.cos(s), math.sqrt(c / b) * np.sin(s))]

    swap = a < 0
    A, B, XM, YM = (b, a, ym, xm) if swap else (a, b, xm, ym)
    if c > 0:
        r, q = math.sqrt(c / A), math.sqrt(c / -B)
        s = np.linspace(-1, 1, n_pts) * math.asinh(YM / q)
        raw = [(sg * r * np.cosh(s), q * np.sinh(s)) for sg in (1, -1)]
    elif c < 0:
        r, q = math.sqrt(c / B), math.sqrt(-c / A)
        s = np.linspace(-1, 1, n_pts) * math.asinh(XM / q)
        raw = [(q * np.sinh(s), sg * r * np.cosh(s)) for sg in (1, -1)]
    else:
        m = math.sqrt(A / -B)
        u = np.linspace(-XM, XM, n_pts)
        raw = [(u, m * u), (u, -m * u)]
    return [clip(v, u) if swap else clip(u, v) for u, v in raw]


def chart_reference_orbit(I: ArrayLike, w0: ArrayLike, k: int, sign: int, n_pts: int = 600) -> Array | None:
    """Órbita exacta de ω₀ en la carta (k, sign) como polilínea con NaN entre ramas; None si ω₀ no está
    en ese hemisferio o el cuerpo es simétrico."""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    if not (sign * w0[k] > 0) or is_degenerate(I):
        return None
    i, j = chart_axes(k)
    a, b = I[i] * (I[i] - I[k]), I[j] * (I[j] - I[k])
    curves = chart_level_curve(I, 2 * float(kinetic_energy(w0, I)), k, a * w0[i] ** 2 + b * w0[j] ** 2, n_pts)
    return np.vstack([np.vstack([C, [[np.nan, np.nan]]]) for C in curves]) if curves else None


# ============================================================== funciones elípticas de Jacobi (sin scipy)
SEP_M = 1.0 - 1e-13          # a partir de aquí m = 1: límite hiperbólico (separatriz)


@lru_cache(maxsize=64)
def _agm(m: float) -> tuple[list[float], list[float], list[float]]:
    """Media aritmético-geométrica descendente: listas (a_n, b_n, c_n) hasta c_N ≈ 0.
    Con ella se obtienen K, F, sn, cn y dn sin necesidad de scipy."""
    a, b, c = 1.0, math.sqrt(1.0 - m), math.sqrt(m)
    A, B, C = [a], [b], [c]
    while abs(c) > 1e-16 and len(A) < 40:
        a, b, c = (a + b) / 2, math.sqrt(a * b), (a - b) / 2
        A.append(a)
        B.append(b)
        C.append(c)
    return A, B, C


def ellip_K(m: float) -> float:
    """Integral elíptica completa de primera especie K(m); ∞ en la separatriz."""
    return math.inf if m >= SEP_M else math.pi / (2 * _agm(m)[0][-1])


def ellip_F(phi: ArrayLike, m: float) -> Array:
    """F(φ | m) para φ real cualquiera (continua en φ): inversa de la amplitud de Jacobi."""
    phi = np.asarray(phi, dtype=float)
    if m >= SEP_M:
        return np.arctanh(np.clip(np.sin(phi), -1 + 1e-16, 1 - 1e-16))
    A, B, _ = _agm(m)
    for n in range(len(A) - 1):                 # φ_{n+1} = φ_n + arctan(b_n/a_n · tan φ_n) + kπ
        phi = phi + np.arctan(B[n] / A[n] * np.tan(phi)) + math.pi * np.round(phi / math.pi)
    return phi / (2.0 ** (len(A) - 1) * A[-1])


def jacobi(u: ArrayLike, m: float) -> tuple[Array, Array, Array]:
    """(sn, cn, dn)(u | m), 0 ≤ m ≤ 1."""
    u = np.asarray(u, dtype=float)
    if m >= SEP_M:
        sech = 1.0 / np.cosh(u)
        return np.tanh(u), sech, sech
    A, _, C = _agm(m)
    N = len(A) - 1
    phi = 2.0 ** N * A[N] * u
    for n in range(N, 0, -1):
        phi = (phi + np.arcsin(np.clip(C[n] / A[n] * np.sin(phi), -1, 1))) / 2
    sn, cn = np.sin(phi), np.cos(phi)
    return sn, cn, np.sqrt(cn * cn + (1.0 - m) * sn * sn)


# ============================================================== solución exacta de Euler
class EulerExact:
    """Solución analítica de  I_i ω̇_i = (I_j − I_k) ω_j ω_k  desde ω(0) = w0.

    Toda la solución es  ω[p] = A_p·dn(u),  ω[q] = A_q·cn(u),  ω[r] = A_r·sn(u),  u = u0 + σΩ_u·t,
    con p el eje (mín. o máx.) que nunca cambia de signo, q el otro extremo y r el intermedio.
    El sólido simétrico es el caso m = 0 (dn = 1, sn y cn = seno y coseno): misma fórmula.

    kind: 'equilibrio' (ω constante) | 'simetrico' (precesión uniforme) | 'jacobi' (caso general).
    El error de una trayectoria numérica se separa en
      · módulo  r = distancia radial a la órbita exacta (adimensional),
      · fase    ψ = adelanto (+) o retraso (−) a lo largo de la órbita, en grados de ciclo,
        y su equivalente temporal `lag` en segundos."""

    _theta_last: float | None
    kind: str
    m: float                 # parámetro elíptico (0 ≤ m ≤ 1); m ≥ SEP_M es la separatriz
    period: float            # ∞ en equilibrio y separatriz
    axes: list[int]          # [p, q, r]
    amp: Array               # amplitudes con signo (A_p, A_q, A_r)

    def __init__(self, I: ArrayLike, w0: ArrayLike) -> None:
        self.I, self.w0 = I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
        self.kind, self.m, self.period, self._theta_last = "equilibrio", 0.0, math.inf, None
        wdot = np.asarray(euler_equations(w0, np.diag(I)), dtype=float).reshape(3)
        if np.linalg.norm(wdot) <= 1e-9 * (w0 @ w0):        # reposo, esfera, eje principal, ω ⟂ eje de simetría
            return
        a, b, c = sorted_axes(I)
        tol = 1e-9 * I[c]
        if I[b] - I[a] < tol or I[c] - I[b] < tol:            # sólido simétrico: p = eje distinto
            self.kind = "simetrico"
            p = c if I[b] - I[a] < tol else a
            q, r = (p + 1) % 3, (p + 2) % 3
            rho = math.hypot(w0[q], w0[r])
            A = np.array([abs(w0[p]), rho, rho])
            self.omega_u = abs((I[p] - I[q]) / I[q] * w0[p])
        else:
            self.kind = "jacobi"
            p, q = (c, a) if (I * w0) @ (I * w0) >= (I @ (w0 * w0)) * I[b] else (a, c)   # ‖L‖² ≥ 2T·I_b
            r = b
            gp = I[r] * (I[q] - I[r]) / (I[p] * (I[q] - I[p]))   # ω_p² + gp·ω_r² = A_p²  (invariantes,
            gq = I[r] * (I[r] - I[p]) / (I[q] * (I[q] - I[p]))   # ω_q² + gq·ω_r² = A_q²   sin cancelaciones)
            Ap2, Aq2 = w0[p] ** 2 + gp * w0[r] ** 2, w0[q] ** 2 + gq * w0[r] ** 2
            A = np.sqrt([Ap2, Aq2, Aq2 / gq])
            self.m = min(gp * A[2] ** 2 / Ap2, 1.0)
            self.omega_u = abs(I[p] - I[q]) * A[0] * A[1] / (I[r] * A[2])
        m = self.m
        # signos: ω_p nunca cambia; en la separatriz ω_q tampoco (cn = sech > 0 solo cubre una rama)
        A = A * [1.0 if w0[p] >= 0 else -1.0, -1.0 if m >= SEP_M and w0[q] < 0 else 1.0, 1.0]
        self.axes, self.amp = [p, q, r], A
        self.theta0 = math.atan2(w0[r] / A[2], w0[q] / A[1])
        self.u0 = float(ellip_F(self.theta0, m))
        self.K = ellip_K(m)
        self.period = 4 * self.K / self.omega_u
        self._cycle = 90.0 / self.K if math.isfinite(self.K) else math.degrees(1.0)     # ° de ciclo por unidad de u
        # sentido en que avanza u: el que reproduce ω̇(0) (derivadas de dn, cn, sn respecto a u)
        sn, cn = math.sin(self.theta0), math.cos(self.theta0)
        dn = math.sqrt(1.0 - m * sn * sn)
        slope = A * self.omega_u * np.array([-m * sn * cn, -sn * dn, cn * dn])
        self.sigma = 1.0 if slope @ wdot[self.axes] >= 0 else -1.0

    def omega(self, t: ArrayLike) -> Array:
        """ω_exacta(t), forma t.shape + (3,)."""
        t = np.asarray(t, dtype=float)
        if self.kind == "equilibrio":
            return np.broadcast_to(self.w0, t.shape + (3,)).copy()
        sn, cn, dn = jacobi(self.u0 + self.sigma * self.omega_u * t, self.m)
        out = np.empty(t.shape + (3,))
        out[..., self.axes] = self.amp * np.stack([dn, cn, sn], axis=-1)
        return out

    def sample(self, t_end: float, max_points: int = 8000) -> tuple[Array, Array]:
        """(t, ω(t)) en [0, t_end] con puntos suficientes (≈ 100 por periodo, entre 600 y `max_points`).
        Es analítica: la malla no depende de ningún Δt."""
        if t_end <= 0:
            return np.empty(0), np.empty((0, 3))
        n = 1500 if not math.isfinite(self.period) else int(min(max_points, max(600, 100 * t_end / self.period)))
        t = np.linspace(0.0, t_end, n)
        return t, self.omega(t)

    def orbit(self, n: int = 720) -> Array | None:
        """Puntos (n, 3) de la órbita de ω en ejes del cuerpo (cerrada si es periódica), o None si es un
        equilibrio. Se parametriza por el ángulo de amplitud φ y no por el tiempo: queda bien resuelta
        aun cerca de la separatriz, donde solo se dibuja la rama que recorre ω."""
        if self.kind == "equilibrio":
            return None
        if self.m >= SEP_M:
            phi = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n + 1)[1:-1]
        else:
            phi = np.linspace(0.0, 2 * math.pi, n + 1)
        out = np.empty((len(phi), 3))
        s = np.sin(phi)
        out[:, self.axes] = self.amp * np.stack([np.sqrt(1.0 - self.m * s * s), np.cos(phi), s], axis=-1)
        return out

    def error_parts(self, t: ArrayLike, W: ArrayLike, reset: bool = False) -> tuple[Array, Array, Array]:
        """(r, psi_deg, lag_s) de la trayectoria numérica W(t) (n, 3). Se llama por tandas
        consecutivas (la fase se desenvuelve de una a otra); `reset=True` empieza de nuevo."""
        t, W = np.asarray(t, dtype=float), np.asarray(W, dtype=float)
        if self.kind == "equilibrio":
            return np.linalg.norm(W - self.w0, axis=1) / max(np.linalg.norm(self.w0), 1e-300), \
                np.zeros(len(t)), np.zeros(len(t))
        if reset:
            self._theta_last = None
        _, q, r = self.axes
        x, y = W[:, q] / self.amp[1], W[:, r] / self.amp[2]
        last = self.theta0 if self._theta_last is None else self._theta_last
        d = (np.diff(np.concatenate([[last], np.arctan2(y, x)])) + math.pi) % (2 * math.pi) - math.pi
        th = last + np.cumsum(d)
        self._theta_last = float(th[-1])
        du = (ellip_F(th, self.m) - self.u0 - self.sigma * self.omega_u * t) * self.sigma   # > 0: adelanto
        return np.hypot(x, y) - 1.0, du * self._cycle, du / self.omega_u


def euler_matches_exact(I: ArrayLike) -> bool:
    """¿Satisface `euler_equations` la ecuación que resuelve EulerExact? Si no (otra convención),
    la solución exacta no sería la de este problema."""
    I = np.asarray(I, dtype=float)

    def std(w: Array) -> Array:
        return np.asarray((np.roll(I, -1) - np.roll(I, -2)) / I * np.roll(w, -1) * np.roll(w, -2), dtype=float)

    ws = np.random.default_rng(1).normal(size=(3, 3))
    return all(np.allclose(np.reshape(euler_equations(w, np.diag(I)), 3), std(w), rtol=1e-8, atol=1e-10)
               for w in ws)


# ============================================================== análisis lineal de un esquema
class LinearErrors(TypedDict):
    """Teoría lineal de un esquema en el centro que rodea la órbita (z = iβΔt, ver `linear_step_errors`)."""
    axis: int
    beta: float            # frecuencia de las pequeñas oscilaciones
    R: complex             # R(iβΔt)
    dmod: float            # |R| − 1 por paso
    dphase: float          # arg R − βΔt por paso (rad)
    period: float          # 2π/β


def _center(I: Array, w0: Array) -> tuple[int, float] | None:
    """(eje, β) del centro lineal que rodea la órbita; None en separatriz, sólido simétrico o sin oscilación."""
    axis = surrounded_axis(I, w0)
    if axis is None:
        return None
    e = equilibrium_analysis(I, 2 * float(kinetic_energy(w0, I)))[axis]
    if e["type"] != "centro":
        return None
    beta = float(np.max(np.abs(e["eigvals"].imag)))
    return (axis, beta) if beta > 0 else None


def relevant_eigenvalues(I: ArrayLike, w0: ArrayLike) -> ComplexArray:
    """Autovalores no nulos del equilibrio relevante de la órbita: el que la rodea o, si está en la
    separatriz o es simétrico, el del eje intermedio."""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    k = surrounded_axis(I, w0)
    k = sorted_axes(I)[1] if k is None else k
    lam = analytic_eigenvalues(I, 2 * float(kinetic_energy(w0, I)))[k]
    return np.asarray(lam[np.abs(lam) > 1e-12], dtype=complex)


def linear_step_errors(I: ArrayLike, w0: ArrayLike, amplification: Amplification, dt: float) -> LinearErrors | None:
    """Error por paso del esquema frente a la solución exacta e^{z} en el centro que rodea la
    órbita (pequeñas oscilaciones): módulo |R(iβΔt)| − 1 y fase arg R − βΔt. None si no hay centro."""
    center = _center(np.asarray(I, dtype=float), np.asarray(w0, dtype=float))
    if center is None:
        return None
    axis, beta = center
    R = complex(amplification(1j * beta * dt))
    return {"axis": axis, "beta": beta, "R": R, "dmod": abs(R) - 1.0,
            "dphase": float(np.angle(R) - beta * dt), "period": 2 * math.pi / beta}


class EquilibriumAmplification(TypedDict):
    """|R(λΔt)| de un esquema en un equilibrio concreto (`equilibrium_amplification`)."""
    type: str              # tipo lineal del equilibrio
    lam: complex           # autovalor relevante (el de mayor |Im| en un centro; el de mayor Re en otro caso)
    R: float               # |R(λΔt)|


def equilibrium_amplification(I: ArrayLike, w0: ArrayLike, k: int, amplification: Amplification,
                              dt: float) -> EquilibriumAmplification | None:
    """|R(λΔt)| del esquema en el equilibrio k del cuerpo (I, w0) (misma energía que w0); None si no hay
    energía o es simétrico."""
    I = np.asarray(I, dtype=float)
    twoT = 2 * float(kinetic_energy(w0, I))
    if twoT <= 0 or is_degenerate(I):
        return None
    e = equilibrium_analysis(I, twoT)[k]
    ev = e["eigvals"]
    lam = ev[np.argmax(np.abs(ev.imag))] if e["type"] == "centro" else ev[np.argmax(ev.real)]
    return {"type": e["type"], "lam": complex(lam), "R": float(abs(amplification(lam * dt)))}


# ============================================================== ¿cuándo un error es relevante?
EPS = float(np.finfo(float).eps)
TINY_R, TINY_PSI = 1e-6, 1e-3    # por debajo: invisible en cualquier gráfica (módulo adimensional, fase en °)


def noise_floor(n_steps: int) -> float:
    """Error de módulo por debajo del cual no se distingue del redondeo acumulado tras n pasos."""
    return 50.0 * EPS * math.sqrt(max(n_steps, 1))


def grade(x: float, floor: float, tiny: float) -> int:
    """0: ≤ ruido de redondeo · 1: ínfimo (invisible) · 2: a tener en cuenta."""
    ax = abs(x)
    return 0 if ax <= floor else (1 if ax < tiny else 2)


def grade_pair(r: float, psi: float, n: int) -> tuple[int, int]:
    """Gravedad del error de módulo r y de fase psi (°) tras n pasos."""
    fl = noise_floor(n)
    return grade(r, fl, TINY_R), grade(psi, math.degrees(fl), TINY_PSI)


def predicted_errors(lin: LinearErrors, n: int) -> tuple[float, float]:
    """(r, psi en °) previstos por la teoría lineal tras n pasos."""
    R = abs(lin["R"])
    r = math.expm1(min(n * math.log(R), 700.0)) if R > 0 else -1.0
    return r, math.degrees(lin["dphase"]) * n


COARSE_BETA_DT = 0.25         # Δt·β por encima de esto (≲ 25 pasos por ciclo) el paso es grueso
EXPLOSION_R = 1.5             # |R| por encima de esto el error de módulo explota en pocos pasos
TREND_TOL = 1e-10             # |R| − 1 por debajo de esto se considera «se conserva»


def amplitude_trend(dmod: float, tol: float = TREND_TOL) -> str:
    """'crece' / 'decae' / 'se conserva' según el signo de |R| − 1 (por paso)."""
    return "crece" if dmod > tol else ("decae" if dmod < -tol else "se conserva")


class StepResolution(TypedDict):
    """Cuántos pasos por ciclo da el esquema y si la teoría lineal lo da por perdido."""
    steps_per_cycle: float   # periodo de oscilación / Δt
    coarse: bool             # Δt·β > COARSE_BETA_DT
    explodes: bool           # |R| > EXPLOSION_R


def step_resolution(lin: LinearErrors, dt: float) -> StepResolution:
    """Resolución temporal de Δt frente a la oscilación del centro (a partir de `linear_step_errors`)."""
    return {"steps_per_cycle": lin["period"] / dt, "coarse": lin["beta"] * dt > COARSE_BETA_DT,
            "explodes": abs(lin["R"]) > EXPLOSION_R}


def scheme_verdict(lin: LinearErrors, n: int) -> tuple[tuple[str, int], tuple[str, int]]:
    """Qué prevé la teoría lineal para el módulo y la fase tras n pasos: ((palabra, gravedad) del módulo,
    (palabra, gravedad) de la fase)."""
    r, psi = predicted_errors(lin, n)
    gr, gp = grade_pair(r, psi, n)
    mod = "se conserva" if gr == 0 else ("crece" if lin["dmod"] > 0 else "decae")
    pha = "sin error" if gp == 0 else ("se retrasa" if lin["dphase"] < 0 else "se adelanta")
    return (mod, gr), (pha, gp)


# ============================================================== comparación de problemas
SURFACE_TOL = 1e-3


class SurfaceMismatch(TypedDict):
    """Diferencia entre las superficies de dos problemas (`surface_mismatch`)."""
    same_I: bool
    dT: float              # diferencia relativa de energía
    dL: float              # diferencia relativa de ‖L‖²


def surface_mismatch(I_ref: ArrayLike, w_ref: ArrayLike, I: ArrayLike, w: ArrayLike,
                     tol: float = SURFACE_TOL) -> SurfaceMismatch | None:
    """None si la trayectoria de (I, w) yace (a tol) sobre las superficies de (I_ref, w_ref)
    (elipsoides de energía y momento); si no, dict(same_I, dT, dL) con las diferencias
    relativas de energía y de ‖L‖²."""
    Ir, Ic = np.asarray(I_ref, dtype=float), np.asarray(I, dtype=float)
    Tr, Tc = float(kinetic_energy(w_ref, Ir)), float(kinetic_energy(w, Ic))
    Lr, Lc = float(angular_momentum_sq(w_ref, Ir)), float(angular_momentum_sq(w, Ic))

    def rel(a: float, b: float) -> float:
        return (b / a - 1.0) if a > 0 else (0.0 if b == 0 else math.inf)

    same_I = bool(np.allclose(Ir, Ic, rtol=1e-9, atol=1e-12))
    dT, dL = rel(Tr, Tc), rel(Lr, Lc)
    if same_I and abs(dT) <= tol and abs(dL) <= tol:
        return None
    return {"same_I": same_I, "dT": dT, "dL": dL}


# ============================================================== convergencia frente a Δt
class Convergence(TypedDict):
    """Resultado de `convergence_study`: errores por Δt y orden observado."""
    dt: Array
    err: Array
    floor: float
    ok: npt.NDArray[np.bool_]
    p: float
    local: Array
    n0: int


def convergence_study(I: ArrayLike, w0: ArrayLike, integrate: Callable[[float, int], Array], T: float,
                      dt: float, levels: int = 5) -> Convergence:
    """Error en T frente a Δt: ‖ω_num(T) − ω_exacta(T)‖ con Δt₀, Δt₀/2, Δt₀/4, … (Δt₀ = T/round(T/dt)).

    integrate(h, n) debe devolver la trayectoria (n+1, ≥3) con ω en las tres primeras columnas, partiendo
    de ω(0) = w0 con paso h. Complementa a Richardson: éste da un orden local con tres pasos y no necesita
    la solución exacta; aquí se usa la exacta y varios Δt, y se ve dónde el error deja de ser fiable.

    Un punto es fiable si es finito, está por encima del ruido de redondeo y no está saturado (error
    comparable a ‖ω‖). El orden observado p es la pendiente del ajuste en log-log de los fiables."""
    I, w0 = np.asarray(I, dtype=float), np.asarray(w0, dtype=float)
    w_ex = EulerExact(I, w0).omega(np.array([float(T)]))[0]
    n0 = max(1, int(round(T / dt)))
    scale = max(float(np.linalg.norm(w0)), 1e-300)
    dts, errs = [], []
    for k in range(levels):
        n, h = n0 * 2 ** k, T / (n0 * 2 ** k)
        with np.errstate(all="ignore"):
            Y = np.asarray(integrate(h, n), dtype=float)
            e = float(np.linalg.norm(Y[-1, :3] - w_ex)) if np.isfinite(Y[-1, :3]).all() else math.nan
        dts.append(h)
        errs.append(e)
    dts_a, errs_a = np.array(dts), np.array(errs)
    floor = noise_floor(n0 * 2 ** (levels - 1)) * scale
    with np.errstate(invalid="ignore"):
        ok = np.isfinite(errs_a) & (errs_a > floor) & (errs_a < 0.1 * scale)
    p = float(np.polyfit(np.log(dts_a[ok]), np.log(errs_a[ok]), 1)[0]) if ok.sum() >= 2 else math.nan
    local = [math.log(errs_a[k + 1] / errs_a[k]) / math.log(dts_a[k + 1] / dts_a[k])
             if ok[k] and ok[k + 1] else math.nan for k in range(levels - 1)]
    return {"dt": dts_a, "err": errs_a, "floor": floor, "ok": ok, "p": p, "local": np.array(local), "n0": n0}
