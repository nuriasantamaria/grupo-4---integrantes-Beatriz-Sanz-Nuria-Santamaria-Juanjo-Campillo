"""Integra esquemas temporales de paso simple y adaptativo (Euler, Crank-Nicolson, RK4),
algoritmos de estimación de error asintótico por
extrapolación de Richardson y cálculo de regiones de estabilidad numérica |R(z)| ≤ 1
para evaluar la conservación física de las integrales primeras."""

import numpy as np
from numpy.linalg import norm

def euler(f, u0, delta, N):
    u0 = np.asarray(u0, dtype=float)
    u = np.zeros((N+1, u0.size))
    u[0, :] = u0
    for n in range(N):
        u[n+1, :] = u[n, :] + delta * f(u[n, :])
    return u

def crank_nicolson(f, u0, delta, N, tol=1e-6, max_iter=100):
    # Misma firma que euler y runge_kutta_4. En cada paso se resuelve
    # Y = u_n + delta/2 (f(u_n) + f(Y)) por iteración de punto fijo (Y <- Y - R).
    u0 = np.asarray(u0, dtype=float)
    u = np.zeros((N+1, u0.size))
    u[0, :] = u0
    for n in range(0, N):
        fn = f(u[n, :])
        Y = u[n, :].copy()
        for _ in range(max_iter):
            R = Y - u[n, :] - delta/2*(fn + f(Y))
            Y = Y - R
            if not norm(R) > tol:   # 'not >' para salir también si aparece NaN
                break
        u[n+1, :] = Y
    return u

def runge_kutta_4(f, u0, delta, N):
    u0 = np.asarray(u0, dtype=float)
    u = np.zeros((N+1, u0.size))
    u[0,:] = u0

    for n in range(0,N):
        k1=f(u[n,:])
        k2=f(u[n,:]+delta*k1/2)
        k3=f(u[n,:]+delta*k2/2)
        k4=f(u[n,:]+delta*k3)
        u[n+1,:]=u[n,:]+delta*(k1+2*k2+2*k3+k4)/6
    return u

def integration_final_state(scheme, f, u0, T, n):
    # Para la extrapolación de Richardson, lo que necesitamos es el estado final de la integración,
    # y resulta útil hacer explícito que llegamos hasta ese instante T utilizando n pasos.
    delta = T / n
    traj = scheme(f, u0, delta, n)
    return traj[-1,:]

def scheme_order(scheme, f, u0, T, n):
    u1 = integration_final_state(scheme, f, u0, T, n)
    u2 = integration_final_state(scheme, f, u0, T, 2*n)
    u3 = integration_final_state(scheme, f, u0, T, 4*n)
    e1 = np.linalg.norm(u1 - u2)
    e2 = np.linalg.norm(u2 - u3)
    return np.log2(e1/e2)


def richardson_error(scheme, f, u0, T, n, p):
    u_coarse = integration_final_state(scheme, f, u0, T, n)
    u_fine = integration_final_state(scheme, f, u0, T, 2*n)
    return np.linalg.norm(u_fine - u_coarse) / (2**p - 1)

def richardson_extrapolation(scheme, f, u0, T, n, p):
    w_coarse = integration_final_state(scheme, f, u0, T, n)
    w_fine = integration_final_state(scheme, f, u0, T, 2*n)
    return w_fine + (w_fine - w_coarse) / (2**p -1)
