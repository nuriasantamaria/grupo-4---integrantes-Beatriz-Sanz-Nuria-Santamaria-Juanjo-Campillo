"""Integra esquemas temporales de paso simple y adaptativo (Euler, Crank-Nicolson, RK4),
algoritmos de estimación de error asintótico por
extrapolación de Richardson y cálculo de regiones de estabilidad numérica |R(z)| ≤ 1
para evaluar la conservación física de las integrales primeras."""

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


def richardson_error(scheme, f, u0, T, N, p):
    u_coarse = final_state(scheme, f, u0, T, N)
    u_fine = final_state(scheme, f, u0, T, 2*N)
    return norm(u_fine - u_coarse) / (2**p - 1)

def richardson_extrapolation(scheme, f, u0, T, N, p):
    w_coarse = final_state(scheme, f, u0, T, N)
    w_fine = final_state(scheme, f, u0, T, 2*N)
    return w_fine + (w_fine - w_coarse) / (2**p -1)