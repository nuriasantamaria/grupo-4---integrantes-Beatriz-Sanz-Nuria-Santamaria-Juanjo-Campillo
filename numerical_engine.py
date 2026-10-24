"""Integración temporal con esquemas de paso fijo (Euler, Euler implícito, Crank-Nicolson, RK4),
factores de amplificación R(z) de cada esquema (para estudiar su estabilidad y su error en
el centro lineal) y estimación del error asintótico por extrapolación de Richardson.

Convenio: f(u, t) es el campo y scheme(f, u, delta, t) avanza un paso de tamaño delta."""

import numpy as np
from numpy.linalg import norm


def euler(f, u, delta, t):
    return u + delta * f(u, t)

def implicit_euler(f, u, delta, t, tol=1e-12, max_iter=100):
    Y = u + delta * f(u, t)
    for _ in range(max_iter):
        Y_new = u + delta * f(Y, t + delta)
        R = Y_new - Y
        Y = Y_new
        if not norm(R) > tol:
            break
    return Y

def crank_nicolson(f, u, delta, t, tol=1e-12, max_iter=100):
    fn = f(u, t)
    Y = u + delta * fn
    for _ in range(max_iter):
        Y_new = u + delta * (fn + f(Y, t + delta)) / 2
        R = Y_new - Y
        Y = Y_new
        if not norm(R) > tol:
            break
    return Y

def runge_kutta_4(f, u, delta, t):
    k1=f(u, t)
    k2=f(u+delta*k1/2, t+delta/2)
    k3=f(u+delta*k2/2, t+delta/2)
    k4=f(u+delta*k3, t+delta)
    return u+delta*(k1+2*k2+2*k3+k4)/6

def amplification_euler(z):
    return 1 + z

def amplification_implicit_euler(z):
    return 1 / (1 - z)

def amplification_crank_nicolson(z):
    return (1 + z/2) / (1 - z/2)

def amplification_runge_kutta_4(z):
    return 1 + z + z**2/2 + z**3/6 + z**4/24

def jacobian(f, u, t=0, h=1e-6):
    # diferencias centradas: columna j = (f(u + h e_j) − f(u − h e_j)) / 2h, error O(h²)
    u = np.asarray(u, dtype=float)
    J = np.zeros((u.size, u.size))
    for j in range(u.size):
        e = np.zeros(u.size)
        e[j] = h
        J[:, j] = (f(u + e, t) - f(u - e, t)) / (2*h)
    return J

def stability_region(amplification, re=(-4, 2), im=(-3, 3), n=400):
    # malla del plano z = λΔt; la región de estabilidad absoluta es donde |R(z)| ≤ 1
    x, y = np.meshgrid(np.linspace(*re, n), np.linspace(*im, n))
    with np.errstate(divide='ignore', invalid='ignore'):    # polos de R (z=1, z=2 en implícitos)
        R = np.abs(amplification(x + 1j*y))
    return x, y, R <= 1

def cauchy_problem(f, u0, t, scheme):
    u0 = np.asarray(u0, dtype=float)
    t = np.asarray(t, dtype=float)
    N = len(t) - 1
    u = np.zeros((N + 1, u0.size))
    u[0, :] = u0

    for n in range(N):
        u[n + 1, :] = scheme(f, u[n, :], t[n + 1] - t[n], t[n])
    return u

def final_state(scheme, f, u0, T, N, t0=0):
    return cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)[-1, :]

def scheme_order(scheme, f, u0, T, N):
    u1 = final_state(scheme, f, u0, T, N)
    u2 = final_state(scheme, f, u0, T, 2*N)
    u3 = final_state(scheme, f, u0, T, 4*N)
    return np.log2(norm(u1 - u2) / norm(u2 - u3))


def richardson_error(scheme, f, u0, T, N, p, t0=0):
    # E(t_n) en los N+1 instantes de la malla gruesa; u_fine[0::2] son esos mismos instantes
    u_coarse = cauchy_problem(f, u0, np.linspace(t0, t0 + T, N + 1), scheme)
    u_fine = cauchy_problem(f, u0, np.linspace(t0, t0 + T, 2*N + 1), scheme)
    return (u_fine[0::2, :] - u_coarse) / (2**p - 1)

def convergence_rate(scheme, f, u0, T, N, m, components=slice(None)):
    # ‖u_2N − u_N‖ en T con N, 2N, 4N, ...: en log-log es una recta de pendiente −p
    log_N, log_E = np.zeros(m), np.zeros(m)
    u1 = final_state(scheme, f, u0, T, N)
    for i in range(m):
        N = 2*N
        u2 = final_state(scheme, f, u0, T, N)
        log_N[i] = np.log10(N)
        log_E[i] = np.log10(norm((u2 - u1)[components]))
        u1 = u2
    ok = np.isfinite(log_E)    # fuera los NaN (divergencia) y los −inf (diferencia nula)
    p = -np.polyfit(log_N[ok], log_E[ok], 1)[0] if ok.sum() >= 2 else np.nan
    return log_N, log_E, p

# no lo pide el enunciado PEI1. Obtiene una estimación mejor a partir de la estimación del error
def richardson_extrapolation(scheme, f, u0, T, N, p):
    w_coarse = final_state(scheme, f, u0, T, N)
    w_fine = final_state(scheme, f, u0, T, 2*N)
    return w_fine + (w_fine - w_coarse) / (2**p -1)

