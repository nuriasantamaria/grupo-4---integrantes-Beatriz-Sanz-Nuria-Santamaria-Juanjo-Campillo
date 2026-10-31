"""Pruebas de Richardson sobre y' = -y (solución exacta e^-t).
Se puede ejecutar con `python test_richardson.py` o con pytest."""
import numpy as np

from numerical_engine.numerical_engine import (runge_kutta_4, final_state, richardson_error,
                              richardson_extrapolation, scheme_order)

f_test = lambda u, t: -u
EXACTA = np.exp(-1.0)


def test_orden_de_rk4():
    assert abs(scheme_order(runge_kutta_4, f_test, [1.0], 1.0, 20) - 4) < 0.2


def test_richardson_estima_el_error_de_la_solucion_fina():
    n = 10
    real = abs(final_state(runge_kutta_4, f_test, [1.0], 1.0, 2 * n)[0] - EXACTA)
    est = float(np.linalg.norm(richardson_error(runge_kutta_4, f_test, [1.0], 1.0, n, p=4)[-1]))   # error en T
    assert 0.5 < est / real < 2.0, f"estimado {est:.2e}, real {real:.2e}"


def test_la_extrapolacion_mejora_y_converge_con_mayor_orden():
    previo = None
    for n in (5, 10, 20, 40):
        fino = abs(final_state(runge_kutta_4, f_test, [1.0], 1.0, 2 * n)[0] - EXACTA)
        extr = abs(richardson_extrapolation(runge_kutta_4, f_test, [1.0], 1.0, n, p=4)[0] - EXACTA)
        assert extr < fino, f"n = {n}: la extrapolación no mejora ({extr:.2e} ≥ {fino:.2e})"
        if previo is not None:
            assert extr < previo / 16, f"n = {n}: el error extrapolado no baja al menos como Δt⁴"
        previo = extr


if __name__ == "__main__":
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_") and callable(fn):
            fn()
            print("✔", nombre[5:].replace("_", " "))