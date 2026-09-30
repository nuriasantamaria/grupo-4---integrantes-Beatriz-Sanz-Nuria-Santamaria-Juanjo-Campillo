from matplotlib.pylab import norm
import matplotlib.pyplot as plt #df is your dataframe
from numpy import array, zeros

def F(u):
    return array([u[1], -u[0]])

N=100
At=0.1
Nv=2
u=zeros((N+1, Nv))
u[0,:]=array([1,0])

"Milestone 1 : Prototypes to integrate orbits without functions."
"1.2 Write a script to integrate orbits with a Crank-Nicolson method."

for n in range(0,N):
	Y=u[n,:]
	while norm(Y-u[n,:]-At/2*(F(u[n,:])+F(Y)))>1e-6:
		R=Y-u[n,:]-At/2*(F(u[n,:])+F(Y))
		Y=Y-R
		print(norm(R))
	u[n+1,:]=Y
plt.plot(u[:,0],u[:,1])
plt.axis('equal')
plt.show()

"1.3 Write a script to integrate orbits with a Runge-Kutta fourth order."

for n in range(0,N):
    k1=F(u[n,:])
    k2=F(u[n,:]+At*k1/2)
    k3=F(u[n,:]+At*k2/2)
    k4=F(u[n,:]+At*k3)
    u[n+1,:]=u[n,:]+At*(k1+2*k2+2*k3+k4)/6
plt.plot(u[:,0],u[:,1])
plt.axis('equal')
plt.show()
