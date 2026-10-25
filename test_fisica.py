"""Comprobaciones de space_physics (solución exacta, análisis lineal, convergencia) y de numerical_engine.

Dos formas de usarlo, en la raíz del repositorio (junto a space_physics.py y numerical_engine.py):
    python test_fisica.py        → informe con ✔ / ✘ / ⚠ y código de salida ≠ 0 si algo falla
    pytest test_fisica.py        → igual, con el formato de pytest

· Las pruebas de space_physics no dependen de numerical_engine (llevan su propio RK4 de referencia).
· Las de numerical_engine usan la misma interfaz que la GUI; si no se puede importar, se omiten.
· ⚠ (aviso) es algo que no es un fallo pero merece una mirada: p. ej. que Crank-Nicolson no conserve
  2T y ‖L‖² (lo hace si es el punto medio implícito; la regla del trapecio solo lo hace aproximadamente)."""
import math
import sys
import warnings
from functools import partial

import numpy as np

import space_physics as sp

try:
    import numerical_engine as ne
except Exception:               # pragma: no cover
    ne = None

M = {"euler": "Euler explícito", "be": "Euler implícito", "cn": "Crank-Nicolson", "rk4": "Runge-Kutta 4"}
CASOS = {                                    # nombre: (I, w0)
    "Janibekov (casi eje intermedio)": ((1, 2, 3), (0.01, 2.0, 0.01)),
    "eje de menor inercia": ((1, 2, 3), (2.0, 0.08, 0.08)),
    "eje de mayor inercia": ((1, 2, 3), (0.08, 0.08, 2.0)),
    "órbita grande": ((1, 2, 3), (0.85, 1.0, 0.5)),
    "trompo oblato": ((1, 1, 3), (0.3, 0.2, 1.0)),
    "trompo alargado": ((3, 1, 1), (1.0, 0.3, 0.2)),
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
    assert sp.regime((1, 1, 3), (0.3, 0.2, 1.0)) == "trompo simétrico"
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


def test_motor_crank_nicolson_conserva_modulo():
    I, w0, dt, n = (1, 2, 3), (2.0, 0.3, 0.3), 0.08, 400
    Y = _integrar("cn", I, w0, dt, n)
    T0, L0 = float(sp.kinetic_energy(w0, I)), float(sp.angular_momentum_sq(w0, I))
    dT = abs(sp.kinetic_energy(Y[-1, :3], I) / T0 - 1)
    dL = abs(sp.angular_momentum_sq(Y[-1, :3], I) / L0 - 1)
    # el C-N de numerical_engine es la regla del trapecio: conserva |R| = 1 en lo lineal,
    # pero no los invariantes cuadráticos exactos; basta con que la deriva sea pequeña
    assert max(dT, dL) < 1e-3, f"deriva excesiva: ΔT/T₀ = {dT:.1e}, Δ‖L‖²/‖L₀‖² = {dL:.1e}"


# ------------------------------------------------------------------ ejecución directa
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