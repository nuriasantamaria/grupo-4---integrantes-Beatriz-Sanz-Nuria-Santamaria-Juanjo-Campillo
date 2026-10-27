"""Pruebas de simulation.py (la sesión de simulación, sin Qt).

    python test_simulation.py     → informe con ✔ / ✘
    pytest test_simulation.py"""
import sys

import numpy as np

import simulation as sm


def cfg(method="rk4", dt=0.02, I=(1.0, 2.0, 3.0), w0=(0.01, 2.0, 0.01)):
    return sm.SimConfig("A", "#fff", True, method, I, w0, dt)


def test_la_simulacion_coincide_con_la_exacta_y_conserva():
    s = sm.Simulation(cfg("rk4", 0.01))
    s.advance(1000)
    assert s.n == 1001 and abs(s.t - 10.0) < 1e-12 and not s.diverged
    w_ex = s.exact.omega(np.array([s.t]))[0]
    assert np.linalg.norm(s.Y[-1, :3] - w_ex) < 1e-8
    assert abs(s.dT[-1]) < 1e-9 and abs(s.dL[-1]) < 1e-9


def test_avance_por_bloques_equivale_a_un_solo_avance():
    a, b = sm.Simulation(cfg("rk4")), sm.Simulation(cfg("rk4"))
    a.advance(500)
    for _ in range(5):
        b.advance(100)
    assert np.allclose(a.Y, b.Y, atol=1e-9) and np.allclose(a.err_r, b.err_r, atol=1e-9)


def test_historial_vista_y_reinicio():
    s = sm.Simulation(cfg("rk4"))
    s.advance(100)
    s.view_n = 41
    assert s.nv == 41 and abs(s.vt - 0.8) < 1e-12 and s.vY.shape == (41, 7) and s.v_steps == 40
    s.view_n = 10_000
    assert s.nv == s.n                                    # acotada al historial
    assert s.index_at(0.5) == 25 and s.index_at(-1) == 0 and s.index_at(1e9) == s.n - 1
    s.reset()
    assert s.n == 1 and s.view_n is None and not s.diverged


def test_divergencia_congela_y_conserva_lo_anterior():
    s = sm.Simulation(cfg("euler", 0.25))
    for _ in range(100):
        s.advance(200)
        if s.diverged:
            break
    assert s.diverged and s.error is None
    assert np.all(np.isfinite(s.Y)) and np.max(np.linalg.norm(s.Y[:, :3], axis=1)) < s._blow
    n = s.n
    s.advance(50)
    assert s.n == n                                      # congelada: no avanza


def test_excepcion_del_esquema_se_registra(monkeypatch=None):
    original = sm.STEPS["be"]

    def malo(f, u, dt, t=0.0, **opciones):
        raise RuntimeError("Newton sin converger")

    sm.STEPS["be"] = malo
    try:
        s = sm.Simulation(cfg("be"))
        s.advance(10)
        assert s.diverged and "Newton sin converger" in s.error and s.n == 1
    finally:
        sm.STEPS["be"] = original


def test_cambio_de_esquema_al_vuelo():
    s = sm.Simulation(cfg("euler", 0.08, w0=(2.0, 0.3, 0.3)))
    s.advance(50)
    n, y = s.n, s.y
    s.switch_method("rk4")
    assert s.cfg.method == "rk4" and s.n == n and s.y == y and not s.diverged
    s.advance(50)
    assert s.n == n + 50 and np.all(np.isfinite(s.Y))
    assert abs(s.err_r[-1]) < abs(s.err_r[n - 1])        # RK4 ya no inyecta energía: el error de módulo deja de crecer
    d = sm.Simulation(cfg("euler", 0.25))                # una simulación divergida se reanuda desde su último estado válido
    for _ in range(100):
        d.advance(200)
        if d.diverged:
            break
    assert d.diverged
    d.switch_method("rk4")
    assert not d.diverged and d.error is None


def test_previsualizacion_completa_y_cancelable():
    p = sm.run_preview(cfg("rk4", 0.02), 10.0)
    assert p is not None and len(p["t"]) == 501 and p["W"].shape == (501, 3) and not p["diverged"]
    ref = sm.Simulation(cfg("rk4", 0.02))
    ref.advance(500)
    assert np.allclose(p["W"], ref.Y[:, :3]) and np.allclose(p["r"], ref.err_r)
    assert sm.run_preview(cfg(), 100, should_cancel=lambda: True) is None
    assert sm.run_preview(cfg("rk4", 0.001), 1e9)["truncated"] is True


def test_richardson_y_convergencia_de_la_sesion():
    r = sm.richardson_report(cfg("rk4", 0.05, w0=(2.0, 0.3, 0.3)), 5.0)
    assert r["n"] == 100 and r["err_real"] < 1e-6 and not r["noise"]
    c = sm.convergence_report(cfg("cn", 0.05, w0=(2.0, 0.3, 0.3)), 5.0, 4)
    assert abs(c["study"]["p"] - 2) < 0.15 and abs(c["p_richardson"] - 2) < 0.3


def test_run_each_aisla_los_fallos_y_respeta_los_desactivados():
    a, b, c = cfg("rk4"), cfg("rk4"), cfg("rk4")
    b.enabled = False

    def job(x):
        if x is c:
            raise ValueError("a propósito")
        return 7

    out = sm.run_each([a, b, c], job)
    assert out[0] == 7 and 1 not in out and isinstance(out[2], ValueError)
    assert sm.run_each([a, c], job, should_cancel=lambda: True) == {}


def test_jacobiana_exacta_del_campo_y_newton_con_ella():
    import numerical_engine as ne
    import space_physics as sp
    I = np.array([1.0, 2.0, 3.0])
    f, J = sp.make_rhs(I), sp.make_jacobian(I)
    u = np.array([0.3, 1.7, -0.4, 0.8, 0.1, -0.5, 0.3])
    assert np.max(np.abs(J(u, 0.0) - ne.my_jacobian(lambda y: f(y, 0.0), u, eps=1e-6))) < 1e-6
    y0 = sp.initial_state([0.3, 1.7, -0.4])
    for m in ("be", "cn"):                      # con Jacobiana exacta o por diferencias finitas: misma solución
        exacta = sm.integrate(m, y0, 0.05, I, 200)
        fd = ne.integrate(ne.SCHEMES[m].step, f, y0, 0.05, 200)
        assert np.max(np.abs(exacta - fd)) < 1e-9


def test_inercias_no_positivas_se_rechazan():
    import space_physics as sp
    for malo in ([1, -2, 3], [0, 1, 1], [1, 1], [1, np.nan, 2]):
        try:
            sp.make_rhs(malo)
        except ValueError:
            continue
        raise AssertionError(f"make_rhs acepto {malo}")


def test_errores_lineales_de_la_sesion():
    lin = sm.linear_errors(cfg("euler", 0.05, w0=(2.0, 0.3, 0.3)))
    assert lin is not None and abs(lin["R"]) > 1                # Euler inyecta energía en un centro
    assert sm.linear_errors(cfg("euler", 0.05, I=(1.0, 1.0, 3.0))) is None      # sólido simétrico: sin teoría lineal


def main():
    pruebas = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    fallos = 0
    for nombre, f in pruebas:
        try:
            f()
            estado, detalle = "✔", ""
        except Exception as e:
            estado, detalle, fallos = "✘", f"{type(e).__name__}: {e}", fallos + 1
        print(f"  {estado} {nombre[5:].replace('_', ' ')}" + (f"\n      {detalle}" if detalle else ""))
    print("RESULTADO:", "TODO OK" if not fallos else f"{fallos} FALLO(S)")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
