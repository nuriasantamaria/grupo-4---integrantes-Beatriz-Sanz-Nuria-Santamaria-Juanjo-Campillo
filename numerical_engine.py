"Integra esquemas temporales de paso simple y adaptativo (Euler, Crank-Nicolson, RK4)," 
"algoritmos de estimación de error asintótico por"
"extrapolación de Richardson y cálculo de regiones de estabilidad numérica |R(z)| ≤ 12" 
"para evaluar la conservación física de las integrales primeras"

"Milestone 1 : Prototypes to integrate orbits without functions."
"1. Write a script to integrate orbits with an Euler method"

from numpy import array, zeros
import matplotlib.pyplot as plt
N=100000
Nv=2
delta=0.01
u=zeros((N+1,Nv))
u[0,:]=array((1,0))
def f(u):
    return array((u[1], -u[0]))
for n in range(0,N):
    u[n+1,:]=u[n,:]+delta*f(u[n,:])
plt.plot(u[:,0],u[:,1])
plt.show()



"Milestone 1 : Prototypes to integrate orbits without functions."
"1.2 Write a script to integrate orbits with a Crank-Nicolson method."

def F(u):
    return array([u[1], -u[0]])
N=1000000
At=0.001
Nv=2
u=zeros((N+1, Nv))
u[0,:]=array([1,0])
for n in range(0,N):
    k1=F(u[n,:])
    k2=F(u[n,:]+At*k1/2)
    k3=F(u[n,:]+At*k2/2)
    k4=F(u[n,:]+At*k3)
    u[n+1,:]=u[n,:]+At*(k1+2*k2+2*k3+k4)/6
plt.plot(u[:,0],u[:,1])
plt.axis('equal')
plt.show()
