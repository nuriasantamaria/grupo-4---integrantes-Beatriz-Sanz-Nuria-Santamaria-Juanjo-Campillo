"Integra esquemas temporales de paso simple y adaptativo (Euler, Crank-Nicolson, RK4)," 
"algoritmos de estimación de error asintótico por"
"extrapolación de Richardson y cálculo de regiones de estabilidad numérica |R(z)| ≤ 12" 
"para evaluar la conservación física de las integrales primeras"

"Milestone 1 : Prototypes to integrate orbits without functions."
"1. Write a script to integrate orbits with an Euler method"


import matplotlib.pyplot as plt
import numpy as np 
from matplotlib.pylab import norm

#Parametros de integración
N=100000
Nv=3
delta=0.01

Omega_mod0 = 1 # Valor inicial del módulo de Omega, en rad/s.
I1 = 1 #temporal
I2 = 2 #temporal
I3 = 3 #temporal 

u = np.zeros((N+1, Nv))
u[0,:]=np.array([0.01,Omega_mod0,0.01]) # Por ejemplo, puede ir en los términos 2 o 3 en su lugar.

def f(u):
    f1 = ((I2 - I3) / I1) * u[1] * u[2]
    f2 = ((I3 - I1) / I2) * u[2] * u[0]
    f3 = ((I1 - I2) / I3) * u[0] * u[1]
    return np.array([f1, f2, f3])

def euler(u,delta):
    for n in range(0,N):
        u[n+1,:]=u[n,:]+delta*f(u[n,:])
    return u 

def crank_nicolson(u,delta):
    for n in range(0,N):
        Y=u[n,:]
        while norm(Y-u[n,:]-delta/2*(f(u[n,:])+f(Y)))>1e-6:
            R=Y-u[n,:]-delta/2*(f(u[n,:])+f(Y))
            Y=Y-R
            print(norm(R))
            u[n+1,:]=Y
    return u

def runge_kutta_4(u,delta):
    for n in range(0,N):
        k1=f(u[n,:])
        k2=f(u[n,:]+delta*k1/2)
        k3=f(u[n,:]+delta*k2/2)
        k4=f(u[n,:]+delta*k3)
        u[n+1,:]=u[n,:]+delta*(k1+2*k2+2*k3+k4)/6
    return u

methods=(euler,crank_nicolson,runge_kutta_4)
names=("Euler","Crank-Nicolson","Runge-Kutta 4")
fig,axes=plt.subplots(1,3,figsize=(15,5))

for ax,integrate,name in zip(axes,methods,names):
    trajectory=integrate(u.copy(),delta)
    ax.plot(trajectory[:,0],trajectory[:,1])
    ax.set_title(name)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()

