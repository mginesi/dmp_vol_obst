"""
Ginesi et al. 2020, fig 9
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':'14'})
rc('text', usetex=True)

from dmp.dmp import DMPs_cartesian
from dmp.obstacle_superquadric import Obstacle_Dynamic as sq_dyn

savefig = False

def plot_obsts(c, ax, coefs, color):
    x_plot = np.linspace(c[0] - ax[0]+0.0000000001, c[0] + ax[0]-0.0000000001, 1000)
    y_plot_p = ax[1] * np.power(1 - np.power((x_plot - c[0]) / ax[0], 2.0 * coeffs[1]) , 1.0 / 2.0 / coeffs[1]) + c[1]
    y_plot_m = - ax[1] * np.power(1 - np.power((x_plot - c[0]) / ax[0], 2.0 * coeffs[1]) , 1.0 / 2.0 / coeffs[1]) + c[1]
    plt.plot(x_plot, y_plot_p, color)
    plt.plot(x_plot, y_plot_m, color)

# Parameters
K = 1050.0
alpha = 4.0
tol = 0.02

lmbda = 10.0
eta = 2.0
beta = 2.0

c1 = np.array([0.5, 1.5])
c2 = np.array([2.0, 0.5])
c3 = np.array([3.5, 1.5])
ax1 = np.array([0.5, 1.5]) + 0.17
ax2 = np.array([1.0, 0.5]) + 0.17
ax3 = np.array([0.5, 1.5]) + 0.17
coeffs = np.array([2.0, 2.0])

c_hull = np.array([2.0, 1.5])
ax_hull = np.array([2.0, 1.5]) + 0.35

# MP and obstacle definition
MP = DMPs_cartesian(n_dmps = 2, K=K, alpha_s=alpha, tol=tol)
MP.x_0 = np.array([0.0, 3.5])
MP.x_goal = np.array([2.0, 1.5])

obst_1 = sq_dyn(center=c1, axis=ax1, coeffs=coeffs, lmbda=lmbda, beta=beta, eta=eta)
obst_2 = sq_dyn(center=c2, axis=ax2, coeffs=coeffs, lmbda=lmbda, beta=beta, eta=eta)
obst_3 = sq_dyn(center=c3, axis=ax3, coeffs=coeffs, lmbda=lmbda, beta=beta, eta=eta)

obst_hull = sq_dyn(center=c_hull, axis=ax_hull, coeffs=coeffs, lmbda=lmbda, beta=beta, eta=eta)

def f_obsts(x, v):
    f = obst_1.gen_external_force(x, v) + obst_2.gen_external_force(x, v) + obst_3.gen_external_force(x, v)
    return f

def f_hull(x, v):
    return obst_hull.gen_external_force(x, v)

# -------------------
# Convex components
# -------------------
MP.reset_state()
conv = False
x_track = np.array([MP.x_0])
while not conv:
    MP.step(external_force=f_obsts, adapt=True)
    x_track = np.append(x_track, np.array([MP.x]), axis=0)
    conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol

plt.figure()
plt.plot(x_track[:, 0], x_track[:, 1], 'k', linewidth=2)
plt.plot(MP.x_0[0], MP.x_0[1], '.k', markersize=8)
plt.plot(MP.x_goal[0], MP.x_goal[1], '.k', markersize=8)
plt.text(-0.3, 3.5, r'$ \mathbf{x}_0 $')
plt.text(2, 1.7, r'$ \mathbf{g} $')
plot_obsts(c1, ax1, coeffs, 'r')
plot_obsts(c2, ax2, coeffs, 'g')
plot_obsts(c3, ax3, coeffs, 'b')
plt.fill([0, 1, 1, 0], [0, 0, 3, 3], color='gray', alpha=0.4)
plt.fill([1, 3, 3, 1], [0, 0, 1, 1], color='gray', alpha=0.4)
plt.fill([3, 4, 4, 3], [0, 0, 3, 3], color='gray', alpha=0.4)
plt.plot([0, 1, 1, 0, 0], [0, 0, 3, 3, 0], '--',color='red')
plt.plot([1, 3, 3, 1, 1], [0, 0, 1, 1, 0], '--',color='green')
plt.plot([3, 4, 4, 3, 3], [0, 0, 3, 3, 0], '--',color='blue')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('scaled')
if savefig:
    plt.savefig('imgs/nonconv_comp.png')

# -------------
# Convex hull
# -------------
MP.x_goal = np.array([5.0, 1.5])
MP.reset_state()
conv = False
x_track = np.array([MP.x_0])
while not conv:
    MP.step(external_force=f_hull, adapt=True)
    x_track = np.append(x_track, np.array([MP.x]), axis=0)
    conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol

plt.figure()
plt.plot(x_track[:, 0], x_track[:, 1], 'k', linewidth=2)
plot_obsts(c_hull, ax_hull, coeffs, 'b')
plt.text(-0.3, 3.5, r'$ \mathbf{x}_0 $')
plt.text(5, 1.7, r'$ \mathbf{g} $')
plt.plot(MP.x_0[0], MP.x_0[1], '.k', markersize=8)
plt.plot(MP.x_goal[0], MP.x_goal[1], '.k', markersize=8)
plt.fill([0, 4, 4, 3, 3, 1, 1, 0, 0], [0, 0, 3, 3, 1, 1, 3, 3, 0], color='gray', alpha=0.4)
plt.plot([0, 4, 4, 0, 0], [0, 0, 3, 3, 0], '--', color='blue')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('scaled')
if savefig:
    plt.savefig('imgs/nonconv_hull.png')

plt.show()