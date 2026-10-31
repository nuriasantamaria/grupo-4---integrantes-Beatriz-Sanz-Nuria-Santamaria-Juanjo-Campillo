"""Sesión de simulación: une la física (`space_physics`) y los esquemas (`numerical_engine`) sin depender de Qt.

Es el modelo que consume la interfaz: guarda el historial de una simulación (estado, derivas de T y ‖L‖,
errores de módulo y fase frente a la solución exacta), avanza el tiempo por bloques, detecta la divergencia
y calcula los análisis pesados (previsualización, Richardson, convergencia). La interfaz solo lo dibuja.

Las dos capas de abajo no se conocen entre sí; este módulo es el único que las compone."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import numpy as np

import simulation.divergence as dv
import numerical_engine.numerical_engine as ne
from numerical_engine.numerical_engine import Array
from space_physics import (ArrayLike, Convergence, EulerExact, LinearErrors, conservation_drift, conserved_quantities,
                           convergence_study, initial_state, make_jacobian, make_rhs, omega_bound,
                           linear_step_errors)

MAX_PREVIEW_STEPS = 300_000
MAX_LIVE_STEPS = 1_000_000
PREVIEW_CHUNK = 2000

# paso de cada esquema por clave del registro de numerical_engine ('euler', 'be', 'cn', 'rk4')
STEPS: Dict[str, ne.Step] = {k: s.step for k, s in ne.SCHEMES.items()}


def step_for(method: str, I: ArrayLike) -> ne.Step:
    """Paso del esquema `method`; los implícitos reciben la Jacobiana exacta del sólido I para su Newton."""
    return partial(STEPS[method], jac=make_jacobian(I)) if ne.SCHEMES[method].implicit else STEPS[method]


@dataclass
class SimConfig:
    """Parámetros de una simulación. `method` es una clave de `numerical_engine.SCHEMES`."""
    name: str
    color: str
    enabled: bool = True
    method: str = "rk4"
    I: Tuple[float, ...] = (1.0, 2.0, 3.0)
    w0: Tuple[float, ...] = (0.01, 2.0, 0.01)
    dt: float = 0.02


def integrate(method: str, y0: ArrayLike, dt: float, I: ArrayLike, n: int, t0: float = 0.0) -> Array:
    """n pasos de `method` sobre el sólido I desde el estado completo y0 (ω, q)."""
    return ne.integrate(step_for(method, I), make_rhs(I), np.asarray(y0, dtype=float), dt, n, t0, stop=dv.stop_criterion())


def linear_errors(cfg: SimConfig) -> Optional[LinearErrors]:
    """Teoría lineal del esquema de `cfg` en el centro que rodea su órbita (None si no hay)."""
    return linear_step_errors(cfg.I, cfg.w0, ne.SCHEMES[cfg.method].amplification, cfg.dt)


class Simulation:
    """Historial de una simulación que avanza en el tiempo y se puede rebobinar (`view_n`)."""

    CHUNK = 8192
    MAX_STEPS_PER_CALL = 20000

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = replace(cfg)
        self.I = np.array(cfg.I, dtype=float)
        self.reset()

    def reset(self) -> None:
        """Vuelve a t = 0 con el estado inicial de `cfg`."""
        w0 = np.array(self.cfg.w0, dtype=float)
        self.y: Tuple[float, ...] = tuple(initial_state(w0))
        self.n = 1
        self.diverged = False
        self.error: Optional[str] = None         # texto si numerical_engine lanzó una excepción
        self.T0, self.L0 = conserved_quantities(w0, self.I)
        self.Ls0 = self.I * w0
        self.w_bound = omega_bound(self.I, w0)   # ‖ω‖ máximo posible en la órbita exacta
        self._blow = dv.omega_threshold(self.I, w0)
        self.exact = EulerExact(self.I, w0)
        self.view_n: Optional[int] = None        # None = en vivo; k = se muestra el historial hasta la fila k-1
        self._Y = np.empty((self.CHUNK, 7))
        self._dT = np.empty(self.CHUNK)
        self._dL = np.empty(self.CHUNK)
        self._er = np.zeros(self.CHUNK)          # error de módulo
        self._ep = np.zeros(self.CHUNK)          # error de fase (° de ciclo)
        self._el = np.zeros(self.CHUNK)          # error de fase (s)
        self._Y[0] = self.y
        self._dT[0] = 0.0
        self._dL[0] = 0.0

    # ---- historial completo
    @property
    def t(self) -> float:
        return self.n_steps * self.cfg.dt

    @property
    def n_steps(self) -> int:
        return self.n - 1

    @property
    def times(self) -> Array:
        return np.asarray(np.arange(self.n) * self.cfg.dt, dtype=float)

    @property
    def Y(self) -> Array:
        return self._Y[:self.n]

    @property
    def dT(self) -> Array:
        return self._dT[:self.n]

    @property
    def dL(self) -> Array:
        return self._dL[:self.n]

    @property
    def err_r(self) -> Array:
        return self._er[:self.n]

    @property
    def err_psi(self) -> Array:
        return self._ep[:self.n]

    @property
    def err_lag(self) -> Array:
        return self._el[:self.n]

    # ---- vista del historial (rebobinar): todo lo que se dibuja sale de aquí
    @property
    def nv(self) -> int:
        return self.n if self.view_n is None else max(1, min(self.n, self.view_n))

    @property
    def vY(self) -> Array:
        return self._Y[:self.nv]

    @property
    def vdT(self) -> Array:
        return self._dT[:self.nv]

    @property
    def vdL(self) -> Array:
        return self._dL[:self.nv]

    @property
    def v_er(self) -> Array:
        return self._er[:self.nv]

    @property
    def v_ep(self) -> Array:
        return self._ep[:self.nv]

    @property
    def v_el(self) -> Array:
        return self._el[:self.nv]

    @property
    def vtimes(self) -> Array:
        return np.asarray(np.arange(self.nv) * self.cfg.dt, dtype=float)

    @property
    def vt(self) -> float:
        return (self.nv - 1) * self.cfg.dt

    @property
    def v_steps(self) -> int:
        return self.nv - 1

    @property
    def vy(self) -> Tuple[float, ...]:
        return tuple(self._Y[self.nv - 1])

    def index_at(self, t: float) -> int:
        """Índice de la fila más cercana al instante t, acotado al historial."""
        return max(0, min(self.n - 1, int(round(t / self.cfg.dt))))

    def switch_method(self, method: str) -> None:
        """Cambia de esquema al vuelo: la simulación sigue desde su último estado válido (también si el
        esquema anterior había divergido) y el historial conserva lo ya calculado."""
        self.cfg = replace(self.cfg, method=method)
        self.diverged, self.error = False, None

    # ---- avance
    def _grow(self, extra: int) -> None:
        need = self.n + extra
        if need <= len(self._Y):
            return
        cap = max(need, 2 * len(self._Y))
        for name in ("_Y", "_dT", "_dL", "_er", "_ep", "_el"):
            old = getattr(self, name)
            new = np.empty((cap,) + old.shape[1:])
            new[:self.n] = old[:self.n]
            setattr(self, name, new)

    def advance_to(self, t_target: float) -> None:
        """Avanza hasta el instante t_target (como mucho MAX_STEPS_PER_CALL pasos por llamada)."""
        n = int(math.ceil(t_target / self.cfg.dt - 1e-9)) - self.n_steps
        if n > 0:
            self.advance(min(n, self.MAX_STEPS_PER_CALL))

    def advance(self, n: int) -> None:
        """Avanza n pasos. Si el esquema falla (p. ej. Newton sin converger) o la solución diverge, la
        simulación se congela: `diverged` pasa a True y `error` guarda el motivo si fue una excepción."""
        if self.diverged or n <= 0:
            return
        try:
            Y = integrate(self.cfg.method, self.y, self.cfg.dt, self.I, n, t0=self.t)[1:]
        except Exception as exc:                # p. ej. Newton sin converger en un esquema implícito
            self.diverged, self.error = True, f"{type(exc).__name__}: {exc}"
            return
        bad = dv.first_divergence(Y, self._blow)    # no finito o muy por encima de lo posible en la órbita exacta
        if bad is not None:
            Y = Y[:bad]
            self.diverged = True
        if len(Y) == 0:
            return
        self._grow(len(Y))
        sl = slice(self.n, self.n + len(Y))
        self._Y[sl] = Y
        self._dT[sl], self._dL[sl] = conservation_drift(Y[:, :3], self.I, self.T0, self.L0)
        tt = np.arange(self.n, self.n + len(Y)) * self.cfg.dt
        with np.errstate(all="ignore"):
            self._er[sl], self._ep[sl], self._el[sl] = self.exact.error_parts(tt, Y[:, :3])
        self.n += len(Y)
        self.y = tuple(Y[-1])


# ============================================================== análisis pesados (se calculan en un hilo)
class Preview(TypedDict):
    """Trayectoria completa de una simulación hasta el horizonte de previsualización."""
    t: Array
    W: Array
    dT: Array
    dL: Array
    r: Array
    psi: Array
    lag: Array
    diverged: bool
    error: Optional[str]
    truncated: bool


def run_preview(cfg: SimConfig, T_total: float, should_cancel: Optional[Callable[[], bool]] = None,
                yield_s: float = 0.001) -> Optional[Preview]:
    """Trayectoria completa hasta T_total. Se integra por bloques para poder cancelarla (devuelve None si
    should_cancel() pasa a True); entre bloques duerme `yield_s` para ceder el GIL a la interfaz."""
    sim = Simulation(cfg)
    n = min(int(math.ceil(T_total / cfg.dt)), MAX_PREVIEW_STEPS)
    left = n
    while left > 0 and not sim.diverged:
        if should_cancel is not None and should_cancel():
            return None
        m = min(PREVIEW_CHUNK, left)
        sim.advance(m)
        left -= m
        time.sleep(yield_s)
    return {"t": sim.times.copy(), "W": sim.Y[:, :3].copy(),
            "dT": sim.dT.copy(), "dL": sim.dL.copy(),
            "r": sim.err_r.copy(), "psi": sim.err_psi.copy(), "lag": sim.err_lag.copy(),
            "diverged": sim.diverged, "error": sim.error, "truncated": n == MAX_PREVIEW_STEPS}


def richardson_report(cfg: SimConfig, T: float) -> Dict[str, Any]:
    """Richardson de la simulación `cfg` hasta T (dict con n, p, err, u_extr, err_real, noise)."""
    exact = EulerExact(cfg.I, cfg.w0)
    return ne.richardson_analysis(
        step_for(cfg.method, cfg.I), make_rhs(cfg.I), initial_state(cfg.w0), T, ne.n_steps(T, cfg.dt),
        ne.SCHEMES[cfg.method].order, reference=lambda t: exact.omega(np.array([t]))[0])._asdict()


class ConvergenceReport(TypedDict):
    """Estudio de convergencia en Δt de una simulación y orden de Richardson para compararlo."""
    study: Convergence
    p_richardson: float


def convergence_report(cfg: SimConfig, T: float, levels: int) -> ConvergenceReport:
    """Error en T frente a Δt (`convergence_study`) más el orden local de Richardson (NaN si no se pudo)."""
    I = np.asarray(cfg.I, dtype=float)
    y0 = initial_state(cfg.w0)
    study = convergence_study(I, cfg.w0, lambda h, n: integrate(cfg.method, y0, h, I, n), T, cfg.dt, levels)
    try:
        p_rich = ne.scheme_order(step_for(cfg.method, cfg.I), make_rhs(cfg.I), y0, T, ne.n_steps(T, cfg.dt),
                                 slice(0, 3))
    except Exception:
        p_rich = math.nan
    return {"study": study, "p_richardson": p_rich}


def run_each(cfgs: List[SimConfig], job: Callable[[SimConfig], Any],
             should_cancel: Optional[Callable[[], bool]] = None) -> Dict[int, Any]:
    """Aplica `job` a cada configuración activa: {índice: resultado, o la excepción si falló}."""
    out: Dict[int, Any] = {}
    for k, c in enumerate(cfgs):
        if not c.enabled:
            continue
        if should_cancel is not None and should_cancel():
            break
        try:
            out[k] = job(c)
        except Exception as exc:                # un esquema que falla no impide analizar los demás
            out[k] = exc
    return out

