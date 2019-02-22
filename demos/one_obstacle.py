import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn

# To use the codes in the main folder
import sys
sys.path.insert(0, '../codes/')
sys.path.insert(0, 'codes/')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import pdb

import dmp_cartesian
import obstacle_ellipse
import obstacle

"""
Here we create the trajectory to learn
"""
t_f = 1. * np.pi # final time
t_steps = 10 ** 3 # time steps
t = np.linspace(0, t_f, t_steps)
a_x = 1. / np.pi
b_x = 1.
a_y = 1. / np.pi
b_y = 1.
x = a_x * t * np.cos(b_x*t)
y = a_y * t * np.sin(b_y*t)
x_des = np.ndarray([2, t_steps])
x_des[0,:] = x
x_des[1,:] = y
x_des -= x_des[:, 0][:, None]
# Learning of the trajectory
dmp = dmp_cartesian.DMPs_cartesian(n_dmps=2, n_bfs=40, K = 1050 * np.ones(2),dt = .01, alpha_s = 3.)
dmp.imitate_path(x_des=x_des)
x_track, dx_track, ddx_track = dmp.rollout()
x_classical = x_track
# Reset state
dmp.reset_state()
x_track = np.zeros((1, dmp.n_dmps))
dx_track = np.zeros((1, dmp.n_dmps))
ddx_track = np.zeros((1, dmp.n_dmps))

dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0
dmp.tol = 5e-02

"""
Volumetric Obstacle
"""
x_track_s = x_track[0]
x_c_1 = -0.5
y_c_1 = 0.7
n = 2
a_1 = .3
b_1 = .2
x_c_2 = 0.15
y_c_2 = 0.4
a_2 = 0.1
b_2 = 0.1
center_1 = np.array([x_c_1, y_c_1])
axis_1 = np.array([a_1, b_1])
center_2 = np.array([x_c_2, y_c_2])
axis_2 = np.array([a_2, b_2])
A = 50.
eta = 1
obst_volume_1 = obstacle_ellipse.Obstacle_Ellipse(n_dim = 2, n = 1, center = center_1, axis = axis_1)

while (not flag):
    if (dmp.t == 0):
        dmp.first = True
    else:
        dmp.first = False
    # run and record timestep
    F = (obst_volume_1.compute_forcing_term(x_track_s, A, eta))
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force=F)
    x_track = np.append(x_track, [x_track_s], axis=0)
    dx_track = np.append(dx_track, [dx_track_s],axis=0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis=0)
    dmp.t += 1
    flag = (dmp.t >= dmp.cs.timesteps) & (np.linalg.norm(x_track_s - dmp.goal) / np.linalg.norm(dmp.goal - dmp.x0) <= dmp.tol)
fig = plt.figure(1)
plt.clf()
plt.figure(1, figsize=(6,6))
plt.plot(x_classical[:,0], x_classical[:, 1], '--b', lw=2, label = 'without obstacle')
plt.plot(x_track[:,0], x_track[:,1], '-g', lw=2, label = 'with obstacle')

"""
Point cloud obstacle
"""

dmp.reset_state()
x_track = np.zeros((1, dmp.n_dmps))
dx_track = np.zeros((1, dmp.n_dmps))
ddx_track = np.zeros((1, dmp.n_dmps))

dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0
dmp.tol = 5e-02
# Obstacle definition
num_obst_1 = 50
t_1 = np.linspace(0., np.pi * 2., num_obst_1)
obst_list_1 = []
obst_list_2 = []
for n in range(num_obst_1):
    obst = obstacle.obstacle()
    obst.def_obstacle (x_obst = np.array([x_c_1 + a_1*np.cos(t_1[n]), y_c_1 + b_1*np.sin(t_1[n])]), dx_obst = np.zeros(2))
    obst_list_1.append(obst)

while (not flag):
    if (dmp.t == 0):
        dmp.first = True
    else:
        dmp.first = False
    # run and record timestep
    F_1 = np.zeros([2])
    F_2 = np.zeros([2])
    for n in range(num_obst_1):
        f_n = obst_list_1[n].gen_external_force(dmp.x, dmp.dx, dmp.goal, n_dim = 2)
        F_1 += f_n
    F = F_1
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force=F)
    x_track = np.append(x_track, [x_track_s], axis=0)
    dx_track = np.append(dx_track, [dx_track_s],axis=0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis=0)
    dmp.t += 1
    flag = (dmp.t >= dmp.cs.timesteps) & (np.linalg.norm(x_track_s - dmp.goal) / np.linalg.norm(dmp.goal - dmp.x0) <= dmp.tol)
plt.plot(x_track[:,0], x_track[:,1], color = 'orange', linestyle = '-.', lw=2, label = 'with obstacle')
x_plot_1 = x_c_1 + a_1*np.cos(t_1)
y_plot_1 = y_c_1 + b_1 * np.sin(t_1)
plt.plot (x_plot_1, y_plot_1, ':r', lw=2, label = 'obstacle')
plt.xlabel(r'$x_1$',fontsize=14)
plt.ylabel(r'$x_2$',fontsize=14)
plt.axis('equal')
plt.text(dmp.x0[0]-0.05, dmp.x0[1]-0.05, r'$\mathbf{x}_0$', fontsize = 16)
plt.text(dmp.goal[0]+0.01, dmp.goal[1]-0.05, r'$\mathbf{g}$', fontsize = 16)

plt.show()