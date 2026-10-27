"""Comprobaciones de space_physics (solución exacta, análisis lineal, convergencia) y de numerical_engine.

Dos formas de usarlo, en la raíz del repositorio (junto a space_physics.py y numerical_engine.py):
    python test_fisica.py        → informe con ✔ / ✘ / ⚠ y código de salida ≠ 0 si algo falla
    pytest test_fisica.py        → igual, con el formato de pytest

· Las pruebas de space_physics no dependen de numerical_engine (llevan su propio RK4 de referencia).
· Las de numerical_engine usan la misma interfaz que la GUI; si no se puede importar, se omiten.
· ⚠ (aviso) es algo que no es un fallo pero merece una mirada (con `warnings.warn` en la prueba)."""
import math
import sys
import warnings
from functools import partial

import numpy as np

import space_physics as sp

try:
    import numerical_engine as ne
except Exception:               # pragma: no cover
    ne = None                   # type: ignore[assignment]

M = {"euler": "Euler explícito", "be": "Euler implícito", "cn": "Crank-Nicolson", "rk4": "Runge-Kutta 4"}
CASOS = {                                    # nombre: (I, w0)
    "Janibekov (casi eje intermedio)": ((1, 2, 3), (0.01, 2.0, 0.01)),
    "eje de menor inercia": ((1, 2, 3), (2.0, 0.08, 0.08)),
    "eje de mayor inercia": ((1, 2, 3), (0.08, 0.08, 2.0)),
    "órbita grande": ((1, 2, 3), (0.85, 1.0, 0.5)),
    "simétrico oblato": ((1, 1, 3), (0.3, 0.2, 1.0)),
    "simétrico alargado": ((3, 1, 1), (1.0, 0.3, 0.2)),
    "otros momentos": ((1.016, 1.091, 3.477), (-0.056, -0.960, -4.43)),
}


# ------------------------------------------------------------------ utilidades
def rhs3(I, w):
    return np.asarray(sp.euler_equations(w, np.diag(np.asarray(I, float))), dtype=float).reshape(3)


def rk4_ref(I, w0, h, n):
    """RK4 propio (sin numerical_engine) con paso muy pequeño: referencia independiente."""
    W = np.empty((n + 1, 3))
    w = W[0] = np.asarray(w0, float)
    for k in range(n):
        k1 = rhs3(I, w); k2 = rhs3(I, w + h / 2 * k1); k3 = rhs3(I, w + h / 2 * k2); k4 = rhs3(I, w + h * k3)
        w = w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        W[k + 1] = w
    return W


def separatriz(sgn):
    """ω₀ exactamente sobre la separatriz de I = (1, 2, 3) con ω₃ del signo pedido."""
    I = np.array([1, 2, 3.0]); w3 = 0.8 * sgn
    w1 = math.sqrt(I[2] * (I[2] - I[1]) * w3 ** 2 / (I[0] * (I[1] - I[0])))
    return I, np.array([w1, 1.0, w3])


# ------------------------------------------------------------------ space_physics: solución exacta
def test_euler_equations_coincide_con_la_ecuacion_resuelta():
    for I in ((1, 2, 3), (3, 1, 2), (1, 1, 3), (0.5, 4, 2.2)):
        assert sp.euler_matches_exact(I), f"euler_equations no es I_i ω̇_i = (I_j − I_k) ω_j ω_k para I={I}"


def test_exacta_coincide_con_integracion_fina():
    t = np.linspace(0, 12, 121)
    for nombre, (I, w0) in CASOS.items():
        I, w0 = np.array(I, float), np.array(w0, float)
        ex = sp.EulerExact(I, w0)
        ref = rk4_ref(I, w0, 0.0025, 4800)[::40]
        err = np.abs(ex.omega(t) - ref).max()
        assert err < 1e-6, f"{nombre}: |exacta − RK4 fino| = {err:.1e} (kind={ex.kind})"


def test_separatriz_exacta_con_omega_q_positivo_y_negativo():
    for sgn in (1, -1):
        I, w0 = separatriz(sgn)
        ex = sp.EulerExact(I, w0)
        assert ex.m >= sp.SEP_M and not math.isfinite(ex.period)
        tt = np.arange(0, 81) * 0.05
        err = np.abs(ex.omega(tt) - rk4_ref(I, w0, 0.0025, 1600)[::20]).max()
        assert err < 1e-6, f"signo {sgn:+d}: {err:.1e}"
        P = ex.orbit(200)
        assert (P[:, 2] * sgn >= -1e-12).all(), "la órbita debe ser solo la rama que recorre ω"


def test_exacta_conserva_energia_y_momento():
    t = np.linspace(0, 40, 4001)
    for nombre, (I, w0) in CASOS.items():
        I, w0 = np.array(I, float), np.array(w0, float)
        W = sp.EulerExact(I, w0).omega(t)
        T0, L0 = float(sp.kinetic_energy(w0, I)), float(sp.angular_momentum_sq(w0, I))
        assert np.abs(sp.kinetic_energy(W, I) - T0).max() < 1e-11 * max(T0, 1), nombre
        assert np.abs(sp.angular_momentum_sq(W, I) - L0).max() < 1e-11 * max(L0, 1), nombre


def test_equilibrios_son_constantes():
    for I, w0 in (((1, 2, 3), (0, 1.7, 0)), ((1, 2, 3), (0, 0, 0)), ((2, 2, 2), (1, 2, 3)), ((1, 1, 3), (1, 1, 0))):
        ex = sp.EulerExact(I, w0)
        assert ex.kind == "equilibrio" and ex.orbit() is None
        assert np.allclose(ex.omega(np.array([0.0, 5.0, 50.0])), np.asarray(w0, float))


def test_orbita_cumple_las_dos_conservaciones():
    for nombre, (I, w0) in CASOS.items():
        I, w0 = np.array(I, float), np.array(w0, float)
        P = sp.EulerExact(I, w0).orbit(300)
        assert np.abs(np.sum(I * P ** 2, axis=1) - np.sum(I * w0 ** 2)).max() < 1e-11, nombre
        assert np.abs(np.sum((I * P) ** 2, axis=1) - np.sum((I * w0) ** 2)).max() < 1e-11, nombre


def test_error_parts_es_cero_para_la_propia_exacta_y_no_depende_de_las_tandas():
    t = np.arange(0, 3000) * 0.01
    for nombre, (I, w0) in CASOS.items():
        ex = sp.EulerExact(np.array(I, float), np.array(w0, float))
        W = ex.omega(t)
        r, psi, lag = ex.error_parts(t, W, reset=True)
        assert np.abs(r).max() < 1e-9 and np.abs(lag).max() < 1e-8, f"{nombre}: {np.abs(r).max():.1e}"
        a = ex.error_parts(t[:1000], W[:1000], reset=True)
        b = ex.error_parts(t[1000:], W[1000:])
        assert np.allclose(np.r_[a[0], b[0]], r, atol=1e-9) and np.allclose(np.r_[a[2], b[2]], lag, atol=1e-8)


def test_error_parts_mide_un_retraso_conocido():
    t = np.arange(0, 2000) * 0.01
    for nombre, (I, w0) in CASOS.items():
        ex = sp.EulerExact(np.array(I, float), np.array(w0, float))
        if ex.kind == "equilibrio" or not math.isfinite(ex.period):
            continue
        delta = 0.01 * ex.period                          # la numérica va un 1 % de ciclo por detrás
        r, psi, lag = ex.error_parts(t, ex.omega(t - delta), reset=True)
        assert np.allclose(lag[500:], -delta, atol=1e-6), f"{nombre}: lag {lag[-1]:+.5f} ≠ {-delta:+.5f}"
        assert np.allclose(psi[500:], -3.6, atol=1e-4) and np.abs(r).max() < 1e-9, nombre


# ------------------------------------------------------------------ space_physics: análisis
def test_clasificacion_de_la_orbita():
    assert "menor" in sp.regime((1, 2, 3), (2.0, 0.08, 0.08)) and "mayor" in sp.regime((1, 2, 3), (0.08, 0.08, 2.0))
    assert sp.regime((1, 1, 3), (0.3, 0.2, 1.0)) == "sólido simétrico"
    I, w0 = separatriz(1)
    assert "separatriz" in sp.regime(I, w0)


def test_teoria_lineal_por_paso():
    R = {"euler": lambda z: 1 + z, "be": lambda z: 1 / (1 - z), "cn": lambda z: (1 + z / 2) / (1 - z / 2),
         "rk4": lambda z: 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24}
    I, w0, dt = (1, 2, 3), (2.0, 0.08, 0.08), 0.02
    lin = {k: sp.linear_step_errors(I, w0, f, dt) for k, f in R.items()}
    z = lin["euler"]["beta"] * dt
    assert abs(lin["euler"]["dmod"] - (math.sqrt(1 + z * z) - 1)) < 1e-14 and lin["euler"]["dmod"] > 0
    assert lin["be"]["dmod"] < 0 and abs(lin["cn"]["dmod"]) < 1e-14
    assert abs(lin["euler"]["dphase"] - (math.atan(z) - z)) < 1e-14
    assert abs(lin["cn"]["dphase"] - (2 * math.atan(z / 2) - z)) < 1e-14
    assert abs(lin["rk4"]["dphase"] / (-z ** 5 / 120) - 1) < 0.02 and abs(lin["rk4"]["dmod"] / (-z ** 6 / 144) - 1) < 0.02
    assert all(v["dphase"] < 0 for v in lin.values()), "todos los esquemas retrasan la fase"
    I_, w_ = separatriz(1)
    assert sp.linear_step_errors(I_, w_, R["rk4"], dt) is None, "sobre la separatriz no hay centro lineal"


def test_relevancia_del_error():
    assert sp.noise_floor(10_000) > sp.noise_floor(100) > 5e-15
    assert sp.grade(1e-17, 1e-13, 1e-6) == 0 and sp.grade(1e-9, 1e-13, 1e-6) == 1 and sp.grade(1e-3, 1e-13, 1e-6) == 2


def test_convergencia_con_integradores_propios():
    """convergence_study con Euler (p = 1) y RK4 (p = 4) escritos aquí: no depende de numerical_engine."""
    I, w0 = (1, 2, 3), (2.0, 0.3, 0.3)

    def integrar(paso, h, n):
        W = np.empty((n + 1, 3)); w = W[0] = np.asarray(w0, float)
        for k in range(n):
            w = paso(I, w, h); W[k + 1] = w
        return W
    euler = lambda I_, w, h: w + h * rhs3(I_, w)

    def rk4(I_, w, h):
        k1 = rhs3(I_, w); k2 = rhs3(I_, w + h / 2 * k1); k3 = rhs3(I_, w + h / 2 * k2); k4 = rhs3(I_, w + h * k3)
        return w + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    for paso, p, dt in ((euler, 1, 0.02), (rk4, 4, 0.08)):
        res = sp.convergence_study(I, w0, partial(integrar, paso), 10.0, dt, levels=5)
        assert res["ok"].sum() >= 3 and abs(res["p"] - p) < 0.2, f"p = {res['p']:.2f}, esperado {p}"


# ------------------------------------------------------------------ numerical_engine (misma interfaz que la GUI: scheme(f, u, delta, t), f(u, t))
def _motor():
    if ne is None:
        if "pytest" in sys.modules:
            sys.modules["pytest"].skip("numerical_engine no se pudo importar")
        raise ModuleNotFoundError("numerical_engine no se pudo importar")
    tol = 1e-12                                  # esquemas y R(z) del registro ne.SCHEMES (el mismo que usa la GUI)
    esquemas = {k: s.step for k, s in ne.SCHEMES.items()}
    esquemas["be"] = partial(esquemas["be"], tol=tol)
    esquemas["cn"] = partial(esquemas["cn"], tol=tol)
    ampl = {k: s.amplification for k, s in ne.SCHEMES.items()}
    return esquemas, ampl


def _rhs7(I):
    return sp.make_rhs(I)                        # (ω, q): Euler + cinemática del cuaternión, de space_physics


def _integrar(clave, I, w0, h, n):
    esquemas, _ = _motor()
    with np.errstate(all="ignore"):
        return np.asarray(ne.cauchy_problem(_rhs7(I), np.array([*w0, 1.0, 0, 0, 0]), h * np.arange(n + 1),
                                            esquemas[clave]))


def test_motor_funciones_de_amplificacion():
    _, ampl = _motor()
    z = 0.3j
    esperado = {"euler": 1 + z, "be": 1 / (1 - z), "cn": (1 + z / 2) / (1 - z / 2),
                "rk4": 1 + z + z ** 2 / 2 + z ** 3 / 6 + z ** 4 / 24}
    for k, f in ampl.items():
        assert abs(f(z) - esperado[k]) < 1e-14, k


def test_motor_ordenes_de_convergencia():
    I, w0 = (1, 2, 3), (2.0, 0.3, 0.3)
    nominal = {"euler": 1, "be": 1, "cn": 2, "rk4": 4}
    for clave, p in nominal.items():
        dt = 0.08 if p == 4 else 0.04
        res = sp.convergence_study(I, w0, lambda h, n, c=clave: _integrar(c, I, w0, h, n), 10.0, dt, levels=5)
        assert res["ok"].sum() >= 2 and abs(res["p"] - p) < 0.35, \
            f"{M[clave]}: orden observado {res['p']:.2f}, esperado {p} (errores {res['err']})"


def test_motor_errores_de_modulo_y_fase_siguen_la_teoria_lineal():
    """Con una perturbación pequeña la teoría lineal predice el signo y el tamaño del error."""
    _, ampl = _motor()
    I, w0, dt, n = (1, 2, 3), (2.0, 0.01, 0.01), 0.05, 400
    t = np.arange(n + 1) * dt
    for clave in ("euler", "be", "rk4"):
        Y = _integrar(clave, I, w0, dt, n)
        ex = sp.EulerExact(np.array(I, float), np.array(w0, float))
        r, psi, _ = ex.error_parts(t, Y[:, :3], reset=True)
        lin = sp.linear_step_errors(I, w0, ampl[clave], dt)
        r_th, psi_th = sp.predicted_errors(lin, n)
        if abs(r_th) > 1e-9:
            assert abs(r[-1] / r_th - 1) < 0.15, f"{M[clave]} módulo: medido {r[-1]:+.2e}, teoría {r_th:+.2e}"
        assert abs(psi[-1] / psi_th - 1) < 0.15, f"{M[clave]} fase: medida {psi[-1]:+.3e}°, teoría {psi_th:+.3e}°"


def test_motor_richardson_sobre_problema_con_solucion_conocida():
    """y' = -y: el error estimado por Richardson debe parecerse al real y la extrapolación mejorarlo."""
    esquemas, _ = _motor()
    f, exacta, T = (lambda u, t: -u), np.exp(-1.0), 1.0
    for clave, p, n in (("euler", 1, 200), ("cn", 2, 50), ("rk4", 4, 10)):
        sch = esquemas[clave]
        orden = ne.scheme_order(sch, f, [1.0], T, n)
        assert abs(orden - p) < 0.2, f"{M[clave]}: orden {orden:.2f}, esperado {p}"
        fino = ne.final_state(sch, f, [1.0], T, 2 * n)[0]
        est = float(np.linalg.norm(ne.richardson_error(sch, f, [1.0], T, n, p)[-1]))   # error en T
        real = abs(fino - exacta)
        assert 0.5 < est / real < 2.0, f"{M[clave]}: estimado {est:.2e}, real {real:.2e}"
        extr = ne.richardson_extrapolation(sch, f, [1.0], T, n, p)[0]
        assert abs(extr - exacta) < real, f"{M[clave]}: la extrapolación no mejora ({abs(extr - exacta):.2e} ≥ {real:.2e})"


def test_motor_implicitos_convergen_con_dt_moderado():
    """El Newton de BE y C-N debe converger (el residuo final bajo la tolerancia)."""
    esquemas, _ = _motor()
    I, w0 = (1, 2, 3), (2.0, 0.3, 0.3)
    f, u = _rhs7(I), np.array([*w0, 1.0, 0, 0, 0])
    for clave in ("be", "cn"):
        v = esquemas[clave](f, u, 0.05, 0.0)
        assert np.isfinite(v).all()
        if clave == "be":
            res = np.linalg.norm(v - u - 0.05 * f(v, 0.05))
            assert res < 1e-9, f"BE: residuo implícito {res:.1e}"


def test_motor_implicitos_avisan_si_no_convergen():
    """Si Newton no llega a la tolerancia debe lanzar un error, no devolver un valor que no es la solución."""
    _motor()
    f, u = _rhs7((1, 2, 3)), np.array([2.0, 0.3, 0.3, 1.0, 0, 0, 0])
    for nombre, sch in (("be", ne.implicit_euler), ("cn", ne.crank_nicolson)):
        try:
            sch(f, u, 0.05, 0.0, max_iter=1)
        except (RuntimeError, ArithmeticError):
            continue
        raise AssertionError(f"{M[nombre]}: con max_iter = 1 no avisó de que Newton no convergía")


def test_motor_registro_de_esquemas():
    """ne.SCHEMES empareja cada esquema con su R(z) y su orden teórico."""
    _motor()
    nominal = {"euler": 1, "be": 1, "cn": 2, "rk4": 4}
    assert {k: s.order for k, s in ne.SCHEMES.items()} == nominal
    b, A = 0.3, np.array([[0.0, -0.3], [0.3, 0.0]])    # u' = A u real, autovalores ±ib
    for k, s in ne.SCHEMES.items():                     # un paso multiplica: por R(−0.5) en u' = −0.5 u, y ‖u‖ por |R(ib)|
        decay = s.step(lambda u, t: -0.5 * u, np.array([1.0]), 1.0, 0.0)[0]
        giro = np.linalg.norm(s.step(lambda u, t: A @ u, np.array([1.0, 0.0]), 1.0, 0.0))
        assert abs(decay - s.amplification(-0.5)) < 1e-9, f"{M[k]}: R(−0.5) = {s.amplification(-0.5)}, paso = {decay}"
        assert abs(giro - abs(s.amplification(1j * b))) < 1e-9, f"{M[k]}: |R(ib)| no coincide con el paso"


def test_motor_regiones_de_estabilidad():
    """stability_grid: |R| ≤ 1 donde la teoría dice (Euler: círculo |1 + z| ≤ 1; BE: |1 − z| ≥ 1; C-N: Re z ≤ 0)."""
    _motor()
    dentro = {"euler": [-1.0, -1.9], "be": [3.0, -2.0, 2.0j], "cn": [-0.5, -3 + 3j], "rk4": [-2.7, 2.8j]}
    fuera = {"euler": [0.5j, 0.1], "be": [1.0 + 0.5j, 1.5], "cn": [0.1, 0.1 + 2j], "rk4": [-2.9, 3.0j]}
    for k, s in ne.SCHEMES.items():
        x, y, A = ne.stability_grid(s.amplification, (-4.0, 4.0), (-4.0, 4.0), n=401)
        en = lambda z: A[np.argmin(abs(x - z.real)), np.argmin(abs(y - z.imag))]
        for z in dentro[k]:
            assert en(complex(z)) <= 1.0, f"{M[k]}: z = {z} debería ser estable (|R| = {en(complex(z)):.3f})"
        for z in fuera[k]:
            assert en(complex(z)) > 1.0, f"{M[k]}: z = {z} debería ser inestable (|R| = {en(complex(z)):.3f})"



def test_motor_exponentes_de_error_del_registro_coinciden_con_R_de_z():
    """module_order y phase_order de ne.SCHEMES son los exponentes reales de |R(ix)| − 1 y arg R − x."""
    _motor()
    for k, s in ne.SCHEMES.items():
        a, b = ne.error_exponents(s.amplification)
        assert (a is None) == (s.module_order is None), M[k]
        if a is not None:
            assert abs(a - (s.module_order + 1)) < 0.1, f"{M[k]}: |R|−1 ∝ z^{a:.2f}"
        assert abs(b - (s.phase_order + 1)) < 0.1, f"{M[k]}: arg R − x ∝ z^{b:.2f}"


def test_motor_integrate_corta_en_la_divergencia_y_first_divergence_la_localiza():
    _motor()
    f = lambda u, t: u ** 2                                   # u' = u²: explota en t = 1/u0
    Y = ne.integrate(ne.euler, f, np.array([1.0, 0.0, 0.0]), 0.05, 400, limit=1e3)
    bad = int(np.argmax(np.isnan(Y[:, 0])))
    assert 0 < bad < 400 and np.isnan(Y[bad:]).all() and np.isfinite(Y[:bad]).all()
    assert abs(Y[bad - 1, 0]) < 1e3 and ne.first_divergence(Y, 1e6) == bad
    assert ne.first_divergence(np.ones((5, 3)), 10.0) is None
    assert ne.first_divergence(np.array([[0, 0, 0], [20, 0, 0.0]]), 10.0) == 1
    sin_limite = ne.integrate(ne.runge_kutta_4, lambda u, t: -u, np.array([1.0]), 0.1, 10, components=slice(0, 1))
    assert np.allclose(sin_limite[-1], np.exp(-1.0), atol=1e-5)


def test_motor_richardson_analysis_equivale_a_las_funciones_por_separado():
    esquemas, _ = _motor()
    I, w0, T, n = (1, 2, 3), (2.0, 0.3, 0.3), 5.0, 100
    f, u0 = sp.make_rhs(I), sp.initial_state(w0)
    ex = sp.EulerExact(np.array(I, float), np.array(w0, float))
    for clave in ("euler", "cn", "rk4"):
        sch, p = esquemas[clave], ne.SCHEMES[clave].order
        rep = ne.richardson_analysis(sch, f, u0, T, n, p, reference=lambda t: ex.omega(np.array([t]))[0])
        assert abs(rep.err - np.linalg.norm(ne.richardson_error(sch, f, u0, T, n, p)[-1][:3])) < 1e-12, clave
        assert np.allclose(rep.u_extr, ne.richardson_extrapolation(sch, f, u0, T, n, p), atol=1e-13), clave
        fino = ne.final_state(sch, f, u0, T, 2 * n)[:3]
        assert abs(rep.err_real - np.linalg.norm(fino - ex.omega(np.array([T]))[0])) < 1e-12, clave


def test_motor_crank_nicolson_conserva_modulo():
    I, w0, dt, n = (1, 2, 3), (2.0, 0.3, 0.3), 0.08, 400
    Y = _integrar("cn", I, w0, dt, n)
    T0, L0 = float(sp.kinetic_energy(w0, I)), float(sp.angular_momentum_sq(w0, I))
    dT = abs(sp.kinetic_energy(Y[-1, :3], I) / T0 - 1)
    dL = abs(sp.angular_momentum_sq(Y[-1, :3], I) / L0 - 1)
    # el C-N de numerical_engine es la regla del trapecio: conserva |R| = 1 en lo lineal,
    # pero no los invariantes cuadráticos exactos; basta con que la deriva sea pequeña
    assert max(dT, dL) < 1e-3, f"deriva excesiva: ΔT/T₀ = {dT:.1e}, Δ‖L‖²/‖L₀‖² = {dL:.1e}"



# ------------------------------------------------------------------ space_physics: equilibrios y autovalores
def test_autovalores_analiticos_coinciden_con_los_numericos():
    rng = np.random.default_rng(0)
    for _ in range(300):
        I, twoT = rng.uniform(0.3, 5, 3), rng.uniform(0.1, 20)
        lam = sp.analytic_eigenvalues(I, twoT)
        for k, om in enumerate(sp.principal_equilibria(I, twoT)):
            num = np.linalg.eigvals(sp.jacobian_matrix(om, np.diag(I)))
            assert np.allclose(np.sort(np.abs(num)), np.sort(np.abs(lam[k])), atol=1e-12), (I, k)


def test_clasificacion_de_equilibrios_y_estabilidad_no_lineal():
    eq = sp.equilibrium_analysis((1, 2, 3), 1.0)
    assert [e["type"] for e in eq] == ["centro", "punto de silla", "centro"]
    assert [e["nonlinear"] for e in eq] == ["estable", "inestable", "estable"]
    assert np.allclose(eq[1]["eigvals"].imag, 0) and np.abs(eq[1]["eigvals"].real).max() > 0
    top = sp.equilibrium_analysis((1, 1, 3), 1.0)             # sólido simétrico: el eje distinto es estable, el resto neutro
    assert [e["nonlinear"] for e in top] == ["neutro", "neutro", "estable"]
    assert all(e["nonlinear"] == "neutro" for e in sp.equilibrium_analysis((2, 2, 2), 1.0))


def test_descomposicion_eje_mas_perturbaciones():
    for w in ((0.01, 2.0, 0.01), (0.3, -0.1, 1.7), (0.0, 0.0, -2.0)):
        axes = sp.axis_decomposition(w)
        a, b, c = axes
        assert abs(w[a]) >= abs(w[b]) >= abs(w[c]) and sorted(axes) == [0, 1, 2]
        assert np.allclose(sp.omega_from_axis(axes, w[a], w[b], w[c]), w)
    assert sp.axis_decomposition((0, 0, 0), keep=(2, 0, 1))[0] == 2
    assert sp.inertia_ranks((3, 1, 2)) == ["máx", "mín", "medio"] and sp.inertia_ranks((1, 1, 3))[:2] == ["=", "="]
    assert sp.triangle_violation((1, 2, 3)) is None and sp.triangle_violation((1, 1, 3)) == 2


def test_cartas_del_plano_de_fases():
    I, w0 = (1, 2, 3), (2.0, 0.3, 0.3)
    k, sign = sp.default_chart(I, w0)
    assert (k, sign) == (0, 1)
    i, j = sp.chart_axes(k)
    twoT = 2 * float(sp.kinetic_energy(w0, I))
    P = sp.chart_reference_orbit(I, w0, k, sign)
    ok = np.isfinite(P).all(axis=1)
    wk = sign * np.sqrt((twoT - I[i] * P[ok, 0] ** 2 - I[j] * P[ok, 1] ** 2) / I[k])
    W = np.zeros((ok.sum(), 3)); W[:, i], W[:, j], W[:, k] = P[ok, 0], P[ok, 1], wk
    assert np.abs(sp.angular_momentum_sq(W, I) - sp.angular_momentum_sq(w0, I)).max() < 1e-8, "la órbita no conserva ‖L‖²"
    assert sp.chart_reference_orbit(I, w0, k, -sign) is None
    assert np.isnan(sp.project_to_chart(np.array([[1.0, 2.0, -3.0]]), 2, 1)).all()
    D = sp.chart_domain(I, twoT, k)
    assert np.allclose(I[i] * D[:, 0] ** 2 + I[j] * D[:, 1] ** 2, twoT)


def test_muestreo_de_la_solucion_exacta_y_comparacion_de_problemas():
    ex = sp.EulerExact((1, 2, 3), (2.0, 0.3, 0.3))
    t, W = ex.sample(5 * ex.period)
    assert 600 <= len(t) <= 8000 and np.allclose(W, ex.omega(t))
    assert sp.same_problem((1, 2, 3), (2, .3, .3), (1, 2, 3), (2, .3, .3)) and not sp.same_problem((1, 2, 3), (2, .3, .3), (1, 2, 4), (2, .3, .3))
    assert sp.surface_mismatch((1, 2, 3), (2, .3, .3), (1, 2, 3), (2, .3, .3)) is None
    m = sp.surface_mismatch((1, 2, 3), (2, .3, .3), (1, 2, 4), (2, .3, .3))
    assert m is not None and not m["same_I"]
    assert sp.omega_bound((1, 2, 3), (2.0, 0.3, 0.3)) >= np.abs(ex.omega(t)).max() * 0.999


def test_equilibrio_efecto_del_esquema_en_el_eje_del_plano():
    R = lambda z: 1 + z
    r = sp.equilibrium_amplification((1, 2, 3), (2.0, 0.3, 0.3), 0, R, 0.05)
    assert r["type"] == "centro" and r["R"] > 1 and abs(r["R"] - abs(1 + r["lam"] * 0.05)) < 1e-15
    assert sp.equilibrium_amplification((1, 1, 3), (1, 0.3, 0.2), 0, R, 0.05) is None
    assert len(sp.relevant_eigenvalues((1, 2, 3), (2.0, 0.3, 0.3))) == 2
    lin_sep = sp.relevant_eigenvalues(*separatriz(1))           # sobre la separatriz: eje intermedio (silla, ±μ reales)
    assert len(lin_sep) == 2 and np.allclose(lin_sep.imag, 0)

# ------------------------------------------------------------------ auditoría: casos límite y funciones nuevas
def test_separatriz_pasa_por_el_equilibrio_intermedio():
    I = (1, 2, 3)
    eq = sp.principal_equilibria(I, 8.0)[1]
    for P in sp.separatrix_curves(I, 8.0):
        assert min(np.linalg.norm(P - eq, axis=1).min(), np.linalg.norm(P + eq, axis=1).min()) < 1e-12
        assert np.allclose((P ** 2 * np.array(I, float)).sum(axis=1), 8.0, atol=1e-12)     # 2T constante


def test_reposo_omega_cero():
    assert sp.regime((1, 2, 3), (0, 0, 0)) == "reposo (ω = 0)"
    assert sp.surrounded_axis((1, 2, 3), (0, 0, 0)) is None
    assert sp.orbit_stability((1, 2, 3), (0, 0, 0))[1] == "marginal"
    assert sp.orbit_stability((1, 1, 3), (0.3, 0.2, 1.0))[1] == "marginal"
    assert all(e["nonlinear"] == "neutro" for e in sp.equilibrium_analysis((1, 2, 3), 0.0))


def test_reasignar_ejes():
    assert sp.reassign_axes((0, 1, 2), new_main=1) == (1, 0, 2)       # el principal viejo ocupa su sitio
    assert sp.reassign_axes((0, 1, 2), new_main=2) == (2, 1, 0)
    assert sp.reassign_axes((0, 1, 2), new_p1=2) == (0, 2, 1)
    assert sp.reassign_axes((0, 1, 2)) == (0, 1, 2)
    for a in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):                       # siempre una permutación
        for m in range(3):
            assert sorted(sp.reassign_axes(a, new_main=m)) == [0, 1, 2]


def test_escenarios_predefinidos():
    assert sp.DEFAULT_SCENARIO in sp.SCENARIOS and len(sp.SCENARIOS) == 8
    claves = set(ne.SCHEMES) if ne is not None else {"euler", "be", "cn", "rk4"}
    for nombre, sc in sp.SCENARIOS.items():
        assert len(sc.sims) == 3 and sc.note, nombre
        for sim in sc.sims:
            assert sim.get("method", "rk4") in claves, nombre
            if sim.get("enabled"):
                assert sim["dt"] > 0 and len(sim["I"]) == 3 and len(sim["w0"]) == 3, nombre


def test_campo_de_direcciones_y_actitud():
    X, Y, U, V, h = sp.reduced_flow_field((1, 2, 3), 8.0, 0, 1)
    assert X.shape == Y.shape == U.shape == V.shape and len(X) > 100 and h > 0
    assert np.all(2 * X ** 2 + 3 * Y ** 2 < 0.96 * 8.0 + 1e-12)       # solo puntos dentro del dominio de la carta
    I, w = np.array([1.0, 2.0, 3.0]), np.array([0.3, 1.2, 0.4])
    q = np.array([np.cos(0.4), 0.0, np.sin(0.4), 0.0])
    R, L_dir, w_dir = sp.attitude_vectors(q, I, w, float(np.linalg.norm(I * w)), float(np.linalg.norm(w)))
    assert np.allclose(R @ R.T, np.eye(3)) and abs(np.linalg.norm(L_dir) - 1) < 1e-12
    assert abs(np.linalg.norm(w_dir) - 1) < 1e-12


def test_tendencia_y_resolucion_del_paso():
    assert (sp.amplitude_trend(1e-3), sp.amplitude_trend(-1e-3), sp.amplitude_trend(0.0)) == ("crece", "decae", "se conserva")
    lin = sp.linear_step_errors((1, 2, 3), (2.0, 0.3, 0.3), lambda z: 1 + z, 0.05)
    res = sp.step_resolution(lin, 0.05)
    assert not res["coarse"] and res["steps_per_cycle"] > 50
    res = sp.step_resolution(sp.linear_step_errors((1, 2, 3), (2.0, 0.3, 0.3), lambda z: 1 + z, 0.3), 0.3)
    assert res["coarse"]


def test_motor_validacion_de_entradas():
    _motor()
    f = lambda u, t: -u
    for dt, n in ((0, 5), (-0.1, 5), (float("nan"), 5), (float("inf"), 5), (0.1, -1), (0.1, 2.5)):
        try:
            ne.integrate(ne.euler, f, np.ones(1), dt, n)
        except ValueError:
            continue
        raise AssertionError(f"integrate({dt}, {n}) debería lanzar ValueError")
    for args in ((0.0, 10, 1.0), (1.0, 0, 1.0), (1.0, 10, 0.0), (-1.0, 10, 1.0)):
        try:
            ne.richardson_analysis(ne.euler, f, np.ones(1), *args)
        except ValueError:
            continue
        raise AssertionError(f"richardson_analysis{args} debería lanzar ValueError")
    assert ne.n_steps(1.0, 0.3) == 3 and ne.n_steps(0.01, 1.0) == 1


def test_motor_richardson_detecta_ruido_de_redondeo():
    _motor()
    f = lambda u, t: -u
    ok = ne.richardson_analysis(ne.runge_kutta_4, f, np.ones(1), 1.0, 50, 4)
    assert not ok.noise and abs(ne.scheme_order(ne.runge_kutta_4, f, np.ones(1), 1.0, 50) - 4) < 0.2
    ruido = ne.richardson_analysis(ne.runge_kutta_4, f, np.ones(1), 1.0, 20000, 4)
    assert ruido.noise and np.isnan(ne.scheme_order(ne.runge_kutta_4, f, np.ones(1), 1.0, 20000))
    ex = ne.richardson_analysis(ne.euler, f, np.ones(1), 1.0, 100, 1, reference=lambda T: np.exp(-T) * np.ones(1))
    assert abs(ex.err - ex.err_real) < 0.1 * ex.err_real        # estimado y real miden lo mismo (misma norma)


def test_motor_contorno_de_estabilidad():
    _motor()
    for clave, s in ne.SCHEMES.items():
        X, Y, inside, rect = ne.stability_contour(s.amplification, (-7.0, 5.0), (-4.5, 4.5))
        z = (X + 1j * Y)[np.isfinite(X)]
        assert z.size > 100, clave
        assert np.max(np.abs(np.abs(s.amplification(z)) - 1)) < 5e-3, clave      # el contorno es |R| = 1
        assert inside.shape == (301, 301) and rect[2] == 12.0 and rect[3] == 9.0
    view, re, im = ne.stability_window(np.array([0.1 + 0.2j]), True)
    assert view[0][1] > 0 and re == (-3 * view[0][1], 3 * view[0][1])
    assert ne.stability_window(np.array([]), True)[0] == ((-4.0, 2.0), (-3.5, 3.5))


def test_motor_utilidades_numericas():
    _motor()
    d = np.array([1e-20, -1e-3, 0.5])
    assert np.isnan(ne.mask_below(d, 1e-16)[0]) and ne.mask_below(d, 1e-16)[1] == 1e-3
    assert np.all(ne.log_safe(d) > 0)
    seg = ne.arrow_segments(np.zeros(2), np.zeros(2), np.array([1.0, 0.0]), np.array([0.0, 0.0]), 1.0)
    assert seg.shape == (6, 2)                                  # el campo nulo no dibuja flecha
    assert np.allclose(seg[0], [-0.5, 0.0]) and np.allclose(seg[1], [0.5, 0.0])
    assert all(s.label and s.short for s in ne.SCHEMES.values())


# ------------------------------------------------------------------ ejecución directa
def test_estabilidad_del_eje_intermedio_frente_a_los_extremos():
    """Teorema del eje intermedio: la desviación del eje de giro no tiende a 0 con la perturbación en el intermedio
    (llega a ~180°) y sí en los extremos (∝ perturbación)."""
    I = (1.0, 2.0, 3.0)
    for eps in (0.3, 0.1, 0.01):
        assert sp.max_deviation(I, (eps, 2.0, eps)) > 160.0
        assert sp.orbit_stability(I, (eps, 2.0, eps))[1] == "inestable"
        assert sp.max_deviation(I, (2.0, eps, eps)) < 120 * eps
        assert sp.orbit_stability(I, (2.0, eps, eps))[1] == "estable"
        assert sp.orbit_stability(I, (eps, eps, 2.0))[1] == "estable"
    assert sp.max_deviation(I, (2.0, 0.01, 0.01)) < 1.0
    assert "inestable" in sp.orbit_stability(I, (0.01, 2.0, 0.01))[0]      # sobre la separatriz también


def main():
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, f in pruebas:
        with warnings.catch_warnings(record=True) as avisos:
            warnings.simplefilter("always")
            try:
                f()
                estado, detalle = "✔", ""
            except ModuleNotFoundError as e:
                estado, detalle = "·", f"omitida ({e})"
            except Exception as e:
                estado, detalle, fallos = "✘", f"{type(e).__name__}: {e}", fallos + 1
        print(f"  {estado} {nombre[5:].replace('_', ' ')}" + (f"\n      {detalle}" if detalle else ""))
        for a in avisos:
            print(f"  ⚠ {a.message}")
    print("RESULTADO:", "TODO OK" if not fallos else f"{fallos} FALLO(S)")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
