import numpy as np
from numerical_engine import runge_kutta_4, integration_final_state, richardson_error, scheme_order, f
from numerical_engine import richardson_extrapolation
from functools import partial
from space_physics import euler_equations

f_test = lambda u: -u
#traj = runge_kutta_4(f_test, [1.0], delta=0.1, N=10)
#print(traj[-1, 0], np.exp(-1.0))

#for n in [10, 20, 40, 80]:

#    traj = runge_kutta_4(f_test, [1.0], delta=1/n, N=n)
#    error = abs(traj[-1, 0] - np.exp(-1.0))
#   print(n, error)

#w = integration_final_state(runge_kutta_4, f_test, [1.0], T=1.0, n=10)
#print(w, np.exp(-1.0))

#est = richardson_error(runge_kutta_4, f_test, [1.0], T=1.0, n=10, p=4)
#real = abs(integration_final_state(runge_kutta_4, f_test, [1.0], T=1.0, n=20) - np.exp(-1.0))
#print(est, real)

#print(scheme_order(runge_kutta_4, f_test, [1.0], T=1.0, n=100))

#traj = runge_kutta_4(f, [1,0.01,0.01], delta=0.1, N=10)
#est = richardson_error(runge_kutta_4, f, [0.01,1,0.01], T=1.0, n=10, p=4)
#print(est)

#I = np.diag([1.0, 2.0, 3.0])
#f = partial(euler_equations, I=I)

#u0 = [0.1, 1.0, 0.1]
#print (scheme_order(runge_kutta_4, f, u0, T=10, n=800))

exacta = np.exp(-1.0); n=10
w_coarse = integration_final_state(runge_kutta_4, f_test, [1.0], 1.0, n)
w_fine = integration_final_state(runge_kutta_4, f_test, [1.0], 1.0, 2*n)
w_extrap = richardson_extrapolation(runge_kutta_4, f_test, [1.0], 1.0, n, p=4)

print("error n=10 :", abs(w_coarse - exacta))
print("error n=20 :", abs(w_fine - exacta))
print("error extrap :", abs(w_extrap - exacta))

for n in [5, 10, 20, 40]:
    w = richardson_extrapolation(runge_kutta_4, f_test, [1.0], 1.0, n, p=4)
    print(n, abs(w[0] - exacta))