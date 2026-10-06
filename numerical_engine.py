"""Integra esquemas temporales de paso simple y adaptativo (Euler, Crank-Nicolson, RK4),
algoritmos de estimación de error asintótico por
extrapolación de Richardson y cálculo de regiones de estabilidad numérica |R(z)| ≤ 1
para evaluar la conservación física de las integrales primeras."""

import numpy as np
from numpy.linalg import norm


def euler(f, u, delta):
    return u + delta * f(u)

def implicit_euler(f, u, delta, tol=1e-6, max_iter=100):
    Y = u + delta * f(u)    
    for _ in range(max_iter):
        R = Y - u - delta * f(Y)
        Y = Y - R
        if not norm(R) > tol:   
            break
    return Y

def crank_nicolson(f, u, delta, tol=1e-6, max_iter=100):
    fn = f(u)
    Y = u.copy()
    for _ in range(max_iter):
        R = Y - u - delta/2*(fn + f(Y))
        Y = Y - R
        if not norm(R) > tol:   
            break
    return Y

def runge_kutta_4(f, u, delta):
    k1=f(u)
    k2=f(u+delta*k1/2)
    k3=f(u+delta*k2/2)
    k4=f(u+delta*k3)
    return u+delta*(k1+2*k2+2*k3+k4)/6

def amplification_euler(z):
    return 1 + z

def amplification_implicit_euler(z):
    return 1 / (1 - z)

def amplification_crank_nicolson(z):
    return (1 + z/2) / (1 - z/2)

def amplification_runge_kutta_4(z):
    return 1 + z + z**2/2 + z**3/6 + z**4/24

def integrate(scheme, f, u0, delta, N):

    u0 = np.asarray(u0, dtype=float)
    u = np.zeros((N+1, u0.size))
    u[0, :] = u0
    for n in range(N):
        u[n+1, :] = scheme(f, u[n, :], delta)
    return u

def integration_final_state(scheme, f, u0, T, n):
    delta = T / n
    traj = integrate(scheme, f, u0, delta, n)
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
