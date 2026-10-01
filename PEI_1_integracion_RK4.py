from numpy import array, zeros, dot, linalg
import matplotlib.pyplot as plt

N = 100000
Nv = 3 # Numero de variables del vector de estado U (en este caso, Omega).
Dt = 0.001

Omega_mod0 = 1 # Valor inicial del módulo de Omega, en rad/s.
I1 = 1
I2 = 2
I3 = 3
I = [I1, I2, I3] # Momentos de inercia principales, en kg x m^2.

t = zeros(N+1)
t[0] = 0
U = zeros((N+1, Nv))
U[0,:] = array([0.01,Omega_mod0,0.01]) # Por ejemplo, puede ir en los términos 2 o 3 en su lugar.

Inv1 = zeros(N+1)
Inv1[0] = dot(U[0,:], I * U[0,:])
Inv2 = zeros(N+1)
Inv2[0] = linalg.norm(I * U[0,:]) ** 2

def F(U):
    f1 = ((I2 - I3) / I1) * U[1] * U[2]
    f2 = ((I3 - I1) / I2) * U[2] * U[0]
    f3 = ((I1 - I2) / I3) * U[0] * U[1]
    return array([f1, f2, f3])

for n in range(0,N):
    k1 = F(U[n,:])
    k2 = F(U[n,:] + Dt * k1/2)
    k3 = F(U[n,:] + Dt * k2/2)
    k4 = F(U[n,:] + Dt * k3)

    t[n+1] = t[n] + Dt
    U[n+1,:] = U[n,:] + Dt*(k1 + 2*k2 + 2*k3 + k4)/6

    Inv1[n+1] = dot(U[n+1,:], I * U[n+1,:]) # 2T, energía cinética.
    Inv2[n+1] = linalg.norm(I * U[n+1,:]) ** 2 # |L|^2, norma momento cinético.

print(Inv1[N], Inv2[N])
plt.plot(Inv2, t)
plt.show()