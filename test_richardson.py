import numpy as np
from numerical_engine import runge_kutta_4, integration_final_state, richardson_error, scheme_order, f

#f_test = lambda u: -u
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

traj = runge_kutta_4(f, [1,0.01,0.01], delta=0.1, N=10)
est = richardson_error(runge_kutta_4, f, [0.01,1,0.01], T=1.0, n=10, p=4)
print(est)