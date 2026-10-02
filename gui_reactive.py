import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from space_physics import euler_equations
from numerical_engine import runge_kutta_4

def ellipsoid_grid(a, b, c, n_points=60):
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)
    THETA, PHI = np.meshgrid(theta, phi)
    # Matrices n_points x n_points que guardan el par de ángulos correspondientes
    # a cada punto de la malla elipsoidal.
    X = a * np.cos(THETA) * np.sin(PHI)
    Y = b * np.sin(THETA) * np.sin(PHI)
    Z = c * np.cos(PHI)
    # Matrices n_points x n_points que contienen las coordenadas cartesianas de cada
    # punto de la malla elipsoidal.
    return X, Y, Z

def poinsot_semiaxes(I, w0):
    I1, I2, I3 = I
    Inv1 = I1 * w0[0]**2 + I2 * w0[1]**2 + I3 * w0[2]**2 # Energía cinética rotacional.
    Inv2 = (I1*w0[0])**2 + (I2*w0[1])**2 + (I3*w0[2])**2 # Cuadrado del módulo del momento angular.

    a_E = np.sqrt(Inv1/I1); b_E = np.sqrt(Inv1/I2); c_E = np.sqrt(Inv1/I3)
    a_L = np.sqrt(Inv2)/I1; b_L = np.sqrt(Inv2)/I2; c_L = np.sqrt(Inv2)/I3

    return (a_E, b_E, c_E), (a_L, b_L, c_L)

def polhode_points(I, w0, n_points=400):
    I1, I2, I3 = I
    Inv1 = I1 * w0[0]**2 + I2 * w0[1]**2 + I3 * w0[2]**2 # Energía cinética rotacional.
    Inv2 = (I1*w0[0])**2 + (I2*w0[1])**2 + (I3*w0[2])**2 # Cuadrado del módulo del momento angular.

    ymax = min( (Inv1*I3 - Inv2) / (I2*(I3-I2)) , (Inv2 - Inv1*I1) / (I2*(I2-I1)) )         
    s = np.linspace(0, 2*np.pi, n_points)
    y = ymax * np.sin(s)**2

    x = (Inv1*I3 - Inv2 - I2*(I3 - I2)*y) / (I1*(I3 - I1))
    z = (Inv2 - Inv1*I1 - I2*(I2 - I1)*y) / (I3*(I3 - I1))

    x = np.maximum(x, 0)
    z = np.maximum(z, 0)

    w2 = np.sqrt(ymax) * np.sin(s)
    w1p = np.sqrt(x)
    w3p = np.sqrt(z)

    curves = []
    for s1 in (+1, -1):
        for s3 in (+1, -1):
            curves.append((s1*w1p, w2, s3*w3p))

    return curves

def integrate_trayectory(I, w0, T=50.0, N=10000):
    f = partial(euler_equations, I=np.diag(I))
    delta = T/N
    traj = runge_kutta_4(f, w0, delta, N)
    t = np.linspace(0, T, N+1)
    return t, traj

def poinsot_drawing(ax, I, w0):
    ax.cla()

    semiaxes_E, semiaxes_L = poinsot_semiaxes(I, w0)

    X_E, Y_E, Z_E = ellipsoid_grid(*semiaxes_E)
    X_L, Y_L, Z_L = ellipsoid_grid(*semiaxes_L)

    t, traj = integrate_trayectory(I, w0)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color="tab:red", lw=2)

    ax.plot_surface(X_E, Y_E, Z_E, alpha=0.3, shade=False, color="tab:green")
    ax.plot_surface(X_L, Y_L, Z_L, alpha=0.3, shade=False, color="tab:pink")
    for w1, w2, w3 in polhode_points(I, w0):
        ax.plot(w1, w2, w3, color="k", lw=1.5)

    lim = max(semiaxes_E + semiaxes_L)
    ax.set_xlim([-lim, lim]); ax.set_ylim([-lim, lim]); ax.set_zlim([-lim, lim])
    ax.set_box_aspect((1, 1, 1))

    ax.set_xlabel("omega1") ; ax.set_ylabel("omega2") ; ax.set_zlabel("omega3")


if __name__ == "__main__":

    I = (1.0, 2.0, 3.0)
    w0 = np.array([0.1, 1.0, 0.1])

    t, traj = integrate_trayectory(I, w0)

    # Geometría y trayectoria.
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    poinsot_drawing(ax, I, w0)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], color="tab:red", lw=2)

    # Vector omega(t)
    fig2, ax2 = plt.subplots()
    ax2.plot(t, traj[:,0], label="omega1")
    ax2.plot(t, traj[:,1], label="omega2")
    ax2.plot(t, traj[:,2], label="omega3")
    ax2.set_xlabel("t"); ax2.legend(); ax2.grid(True)

    plt.show()
    
    #I = (1.0, 2.0, 3.0)
    #w0 = np.array([0.1, 1.0, 0.1])
    #(a_E, b_E, c_E), (a_L, b_L, c_L) = poinsot_semiaxes(I, w0)
    #print("Energía :", a_E, b_E, c_E)
    #print("Momento :", a_L, b_L, c_L)

    #I1, I2, I3 = I
    #twoT = I1*w0[0]**2 + I2*w0[1]**2 + I3*w0[2]**2
    #L2 = (I1*w0[0])**2 + (I2*w0[1])**2 + (I3*w0[2])**2

    #for w1, w2, w3 in polhode_points(I, w0):
    #    print(np.max(np.abs(I1*w1**2 + I2*w2**2 + I3*w3**2 - twoT)),
    #np.max(np.abs((I1*w1)**2 + (I2*w2)**2 + (I3*w3)**2 - L2)))

    
    








