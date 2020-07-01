"""
Ginesi et al. 2020, fig 1
"""

import numpy as np
from dmp.dmp_cartesian import DMPs_cartesian as dmp_cart

# Plot stuff
import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':'14'})
rc('text', usetex=True)

# Trajectory Generation
t_span = np.linspace(0.0, np.pi, 500)
gamma = np.transpose(np.array([t_span, np.sin(t_span) ** 2.0]))

# DMP
mp = dmp_cart(n_dmps=2, tol=0.05)
mp.imitate_path(x_des=gamma, t_des=t_span)
mp.x_goal += np.array([0.0, 0.5])

x_track = mp.rollout()[0]

plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--b')
plt.plot(x_track[:, 0], x_track[:, 1], '-r')
plt.plot(gamma[0][0], gamma[0][1], '.k', markersize=10)
plt.plot(gamma[-1][0], gamma[-1][1], '.k', markersize=10)
plt.plot(mp.x_goal[0], mp.x_goal[1], '.k', markersize=10)
plt.text(-0.3, -0.02,r'$\mathbf{x}_0$', fontsize=16)
plt.text(3.3, -0.02,r'$\mathbf{g}$', fontsize=16)
plt.text(3.3, 0.48,r"$\mathbf{g}'$", fontsize=16)
plt.xlim(-0.5, 3.5)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)

plt.show()