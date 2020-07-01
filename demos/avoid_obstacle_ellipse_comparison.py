import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import pdb

from dmp import dmp_cartesian
from dmp import obstacle_superquadric
from dmp import point_obstacle

"""
Here we create the trajectory to learn
"""

t_f = 1.0 * np.pi # final time
t_steps = 10 ** 3 # time steps
t = np.linspace(0, t_f, t_steps)

a_x = 1.0 / np.pi
b_x = 1.0
a_y = 1.0 / np.pi
b_y = 1.0

x = a_x * t * np.cos(b_x*t)
y = a_y * t * np.sin(b_y*t)

x_des = np.ndarray([t_steps, 2])
x_des[:, 0] = x
x_des[:, 1] = y

x_des -= x_des[0]

# Learning of the trajectory
dmp = dmp_cartesian.DMPs_cartesian(n_dmps = 2, n_bfs = 40, K = 1000.0, dt = 0.01, alpha_s = 3.0, tol = 2e-02)
dmp.imitate_path(x_des = x_des)
x_track, _, _, _ = dmp.rollout()
x_classical = x_track
# Execution with the obstacles
dmp.reset_state()
x_track = np.zeros([1, dmp.n_dmps])
dx_track = np.zeros([1, dmp.n_dmps])
ddx_track = np.zeros([1, dmp.n_dmps])

dmp.dx_old = np.zeros(dmp.n_dmps)
dmp.ddx_old = np.zeros(dmp.n_dmps)
flag = False
dmp.t = 0

"""
Volumetric Obstacle
"""
x_track_s = x_track[0]
x_c_1 = - 0.5
y_c_1 = 0.7
n = 2
a_1 = 0.3
b_1 = 0.2
x_c_2 = 0.15
y_c_2 = 0.4
a_2 = 0.1
b_2 = 0.1
center_1 = np.array([x_c_1, y_c_1])
axis_1 = np.array([a_1, b_1])
center_2 = np.array([x_c_2, y_c_2])
axis_2 = np.array([a_2, b_2])
A = 50.0
eta = 1.0
obst_volume_1 = obstacle_superquadric.Obstacle_Static(center = center_1, axis = axis_1, A = A, eta = eta)
obst_volume_2 = obstacle_superquadric.Obstacle_Static(center = center_2, axis = axis_2, A = A, eta = eta)

def F_sq(x, v):
    return obst_volume_1.gen_external_force(x) + obst_volume_2.gen_external_force(x)

while (not flag):
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force = F_sq, adapt=False)
    x_track = np.append(x_track, [x_track_s], axis = 0)
    dx_track = np.append(dx_track, [dx_track_s],axis = 0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis = 0)
    dmp.t += 1
    flag = (np.linalg.norm(x_track_s - dmp.x_goal) / np.linalg.norm(dmp.x_goal - dmp.x_0) <= dmp.tol)
    fig = plt.figure(1)
    plt.clf()
plt.figure(1, figsize = (6,6))
plt.plot(x_classical[:,0], x_classical[:, 1], '--', color='orange', lw=2, label = 'without obstacle')
plt.plot(x_track[:,0], x_track[:,1], '-b', lw=2, label = 'static vol obst')

## Dynamic
x_track_d = np.array([dmp.x_0])
lmbda = 10.0
beta = 2.0
eta = 1.0
obst_volume_dyn_1 = obstacle_superquadric.Obstacle_Dynamic(center = center_1, axis = axis_1, lmbda=lmbda, beta=beta, eta=eta)
obst_volume_dyn_2 = obstacle_superquadric.Obstacle_Dynamic(center = center_2, axis = axis_2, lmbda=lmbda, beta=beta, eta=eta)

def F_sq_d(x, v):
    return obst_volume_dyn_1.gen_external_force(x, v) + obst_volume_dyn_2.gen_external_force(x, v)

flag = False
dmp.reset_state()
while (not flag):
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force = F_sq_d, adapt=False)
    x_track_d = np.append(x_track_d, [x_track_s], axis = 0)
    dx_track = np.append(dx_track, [dx_track_s],axis = 0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis = 0)
    dmp.t += 1
    flag = (np.linalg.norm(x_track_s - dmp.x_goal) / np.linalg.norm(dmp.x_goal - dmp.x_0) <= dmp.tol)
plt.plot(x_track_d[:,0], x_track_d[:,1], '-', color='magenta', lw=2, label = 'dynamic vol obst')

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
num_obst_2 = 50
t_1 = np.linspace(0.0, np.pi * 2.0, num_obst_1)
t_2 = np.linspace(0.0, np.pi * 2.0, num_obst_2)
obst_list_1 = []
obst_list_2 = []
for n in range(num_obst_1):
    obst = point_obstacle.Obstacle_Steering(x = np.array([x_c_1 + a_1*np.cos(t_1[n]), y_c_1 + b_1*np.sin(t_1[n])]))
    obst_list_1.append(obst)

for n in range(num_obst_2):
    obst = point_obstacle.Obstacle_Steering(x = np.array([x_c_2 + a_2*np.cos(t_2[n]), y_c_2 + b_2*np.sin(t_2[n])]))
    obst_list_2.append(obst)

def F_sa(x, v):
    f = np.zeros(2)
    for _n in range(num_obst_1):
        f += obst_list_1[_n]. gen_external_force(x, v, dmp.x_goal)
    for _n in range(num_obst_2):
        f += obst_list_2[_n]. gen_external_force(x, v, dmp.x_goal)
    return f

while (not flag):
    # run and record timestep
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force = F_sa, adapt=False)
    x_track = np.append(x_track, [x_track_s], axis = 0)
    dx_track = np.append(dx_track, [dx_track_s],axis = 0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis = 0)
    dmp.t += 1
    flag = (np.linalg.norm(x_track_s - dmp.x_goal) / np.linalg.norm(dmp.x_goal - dmp.x_0) <= dmp.tol)
plt.plot(x_track[:, 0], x_track[:, 1], '-r', lw = 2, label = 'Pastor (steering angle)')

# Obstacle plot
x_plot_1 = x_c_1 + a_1*np.cos(t_1)
y_plot_1 = y_c_1 + b_1 * np.sin(t_1)
plt.plot (x_plot_1, y_plot_1, '.k', lw = 2)
x_plot_2 = x_c_2 + a_2*np.cos(t_2)
y_plot_2 = y_c_2 + b_2 * np.sin(t_2)
plt.plot (x_plot_2, y_plot_2, '.k', lw = 2)
plt.xlabel(r'$x$',fontsize = 14)
plt.ylabel(r'$y$',fontsize = 14)
plt.axis('equal')
plt.legend(loc = 'best')
plt.show()