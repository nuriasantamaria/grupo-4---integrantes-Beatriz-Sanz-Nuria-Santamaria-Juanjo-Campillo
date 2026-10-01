import numpy as np
from typing import callable
import matplotlib.pyplot as plt

Nv=2
N=1000
t=1
delta=0.001

U=np.zeros((N+1,Nv))
U[0,:]=np.array([1,0])

def F(U, t):
    return np.array([U[1], -U[0]])

def euler(U, t, delta, F):
    return U + delta * F(U, t)

for n in range (0,N):
    U[n+1,:]=euler(U[n,:], N*delta, delta, F)

plt.plot(U[:,0], U[:,1])
plt.axis('equal')
plt.show()
