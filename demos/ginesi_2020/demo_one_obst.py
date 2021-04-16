"""
Ginesi et al. 2020, fig 5, fig 6
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':'14'})
rc('text', usetex=True)

from dmp.dmp import DMPs_cartesian
from dmp.obstacle_superquadric import Obstacle_Dynamic as sq_dyn
from dmp.obstacle_superquadric import Obstacle_Static as sq_stat
from dmp.point_obstacle import Obstacle_Steering as pt_steer
from dmp.point_obstacle import Obstacle_Potential_Static as pt_stat
from dmp.point_obstacle import Obstacle_Potential_Dynamic as pt_dyn

# ---------------------------------------------------------------------------- #
#                                  Setup                                       #
# ---------------------------------------------------------------------------- #

# Trajectory
t_des = np.linspace(0.0, 1.0, 500)
gamma = np.transpose(
    [t_des * np.cos(np.pi * t_des),
    t_des * np.sin(np.pi * t_des)])

# DMP
K = 1050.0
dt = 0.002
tol = 0.01
MP = DMPs_cartesian(n_dmps=2, K=K, dt=dt, tol=tol)
MP.imitate_path(t_des=t_des, x_des=gamma)
gamma, _, _, _ = MP.rollout()
t_des = np.linspace(0.0, 1.0, np.shape(gamma)[0])

# Obstacle
obst_center = np.array([-0.5, 0.7])
obst_axis = np.array([0.3, 0.2])
density_mesh = 50
x_obst = (obst_axis[0] * np.cos(np.linspace(0.0, 2*np.pi, density_mesh + 1)) + obst_center[0])[:-1]
y_obst = (obst_axis[1] * np.sin(np.linspace(0.0, 2*np.pi, density_mesh + 1)) + obst_center[1])[:-1]

# ---------------------------------------------------------------------------- #
#                             Rollout with obstacle                            #
# ---------------------------------------------------------------------------- #

## Point static obstacle

# Obstacle definition
p0 = 0.1
eta = 1.0
list_obst_static_point = []
for _i in range(density_mesh):
    list_obst_static_point.append(pt_stat(
        x=np.array([x_obst[_i], y_obst[_i]]), p0=p0, eta=eta))

def ext_f_psp(x, v):
    f = np.zeros(2)
    for obst in list_obst_static_point:
        f += obst.gen_external_force(x)
    return f

# Rollout with obstacles
MP.reset_state()
x_track_point_static = np.array([MP.x])
ddx_track_point_static = np.array([MP.ddx])
while(np.linalg.norm(MP.x - MP.x_goal) > MP.tol):
    MP.step(external_force=ext_f_psp)
    x_track_point_static = np.append(x_track_point_static, np.array([MP.x]), axis=0)
    ddx_track_point_static = np.append(ddx_track_point_static, np.array([MP.ddx]), axis=0)

## Point dynamic obstacle

# Obstacle definition
lmbda = 0.2
beta = 2.0
list_obst_dynamic_point = []
for _i in range(density_mesh):
    list_obst_dynamic_point.append(pt_dyn(
        x=np.array([x_obst[_i], y_obst[_i]]), lmbda=lmbda, beta=beta))

def ext_f_pdp(x, v):
    f = np.zeros(2)
    for obst in list_obst_dynamic_point:
        f += obst.gen_external_force(x, v)
    return f

# Rollout with obstacles
MP.reset_state()
x_track_point_dynamic = np.array([MP.x])
ddx_track_point_dynamic = np.array([MP.ddx])
while(np.linalg.norm(MP.x - MP.x_goal) > MP.tol):
    MP.step(external_force=ext_f_pdp)
    x_track_point_dynamic = np.append(x_track_point_dynamic, np.array([MP.x]), axis=0)
    ddx_track_point_dynamic = np.append(ddx_track_point_dynamic, np.array([MP.ddx]), axis=0)

## Point steering angle obstacle

# Obstacle definition
gma = 20.0
beta = 3.0
list_obst_steering_point = []
for _i in range(density_mesh):
    list_obst_steering_point.append(pt_steer(
        x=np.array([x_obst[_i], y_obst[_i]]), gamma=gma, beta=beta))

def ext_f_psa(x, v):
    f = np.zeros(2)
    for obst in list_obst_steering_point:
        f += obst.gen_external_force(x, v, MP.x_goal)
    return f

# Rollout with obstacles
MP.reset_state()
x_track_point_steering = np.array([MP.x])
ddx_track_point_steering = np.array([MP.ddx])
while(np.linalg.norm(MP.x - MP.x_goal) > MP.tol):
    MP.step(external_force=ext_f_psa)
    x_track_point_steering = np.append(x_track_point_steering, np.array([MP.x]), axis=0)
    ddx_track_point_steering = np.append(ddx_track_point_steering, np.array([MP.ddx]), axis=0)

## Volume static obstacle

# Obstacle definition
A = 10.0
eta = 1.0
obst_stat_vol = sq_stat(center=obst_center, axis=obst_axis, A=A, eta=eta)

def ext_f_vs(x, v):
    return obst_stat_vol.gen_external_force(x)

# Rollout with obstacles
MP.reset_state()
x_track_static_volume = np.array([MP.x])
ddx_track_static_volume = np.array([MP.ddx])
while(np.linalg.norm(MP.x - MP.x_goal) > MP.tol):
    MP.step(external_force=ext_f_vs)
    x_track_static_volume = np.append(x_track_static_volume, np.array([MP.x]), axis=0)
    ddx_track_static_volume = np.append(ddx_track_static_volume, np.array([MP.ddx]), axis=0)

## Volume dynamic obstacle

# Obstacle definition
lmbda = 10.0
beta = 2.0
eta = 0.5
obst_stat_vol = sq_dyn(center=obst_center, axis=obst_axis, lmbda=lmbda, beta=beta, eta=eta)

def ext_f_vd(x, v):
    return obst_stat_vol.gen_external_force(x, v)

# Rollout with obstacles
MP.reset_state()
x_track_dynamic_volume = np.array([MP.x])
ddx_track_dynamic_volume = np.array([MP.ddx])
while(np.linalg.norm(MP.x - MP.x_goal) > MP.tol):
    MP.step(external_force=ext_f_vd)
    x_track_dynamic_volume = np.append(x_track_dynamic_volume, np.array([MP.x]), axis=0)
    ddx_track_dynamic_volume = np.append(ddx_track_dynamic_volume, np.array([MP.ddx]), axis=0)

# ---------------------------------------------------------------------------- #
#                           Distance from original                             #
# ---------------------------------------------------------------------------- #
from scipy.interpolate import interp1d

static_point_interp = interp1d(
    np.linspace(0.0, 1.0, np.shape(x_track_point_static)[0]),
    np.transpose(x_track_point_static), kind='cubic')
static_point_interp = np.transpose(static_point_interp(t_des))
err_static_point = np.linalg.norm(static_point_interp - gamma, axis=1)

dynamic_point_interp = interp1d(
    np.linspace(0.0, 1.0, np.shape(x_track_point_dynamic)[0]),
    np.transpose(x_track_point_dynamic), kind='cubic')
dynamic_point_interp = np.transpose(dynamic_point_interp(t_des))
err_dynamic_point = np.linalg.norm(dynamic_point_interp - gamma, axis=1)

steering_point_interp = interp1d(
    np.linspace(0.0, 1.0, np.shape(x_track_point_steering)[0]),
    np.transpose(x_track_point_steering), kind='cubic')
steering_point_interp = np.transpose(steering_point_interp(t_des))
err_steering_point = np.linalg.norm(steering_point_interp - gamma, axis=1)

static_volume_interp = interp1d(
    np.linspace(0.0, 1.0, np.shape(x_track_static_volume)[0]),
    np.transpose(x_track_static_volume), kind='cubic')
static_volume_interp = np.transpose(static_volume_interp(t_des))
err_static_volume = np.linalg.norm(static_volume_interp - gamma, axis=1)

dynamic_volume_interp = interp1d(
    np.linspace(0.0, 1.0, np.shape(x_track_dynamic_volume)[0]),
    np.transpose(x_track_dynamic_volume), kind='cubic')
dynamic_volume_interp = np.transpose(dynamic_volume_interp(t_des))
err_dynamic_volume = np.linalg.norm(dynamic_volume_interp - gamma, axis=1)

# ---------------------------------------------------------------------------- #
#                                   Plot                                       #
# ---------------------------------------------------------------------------- #

# Show purposes
plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--',color='orange', label='desired')
plt.plot(x_obst, y_obst, '.k', label='obstacle')
plt.plot(x_track_point_static[:, 0], x_track_point_static[:, 1], ':r', label='point static')
plt.plot(x_track_point_dynamic[:, 0], x_track_point_dynamic[:, 1], linestyle=(0, (3, 2, 1, 2)), color = 'purple', label='point dynamic')
plt.plot(x_track_point_steering[:, 0], x_track_point_steering[:, 1], linestyle=(0, (2, 2, 1, 2, 1, 2)), color='green', label='point steering')
plt.plot(x_track_static_volume[:, 0], x_track_static_volume[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2)), color='blue', label='volume static')
plt.plot(x_track_dynamic_volume[:, 0], x_track_dynamic_volume[:, 1], linestyle=(0, (3, 2, 1, 2, 1, 2,1,2)), color='fuchsia', label='volume dynamic')
plt.plot(gamma[0][0], gamma[0][1] , '.k', markersize=10)
plt.text(gamma[0][0]-0.08, gamma[0][1]-0.01, r'$\mathbf{x}_0$', fontsize=16)
plt.plot(gamma[-1][0], gamma[-1][1], '.k', markersize=10)
plt.text(gamma[-1][0]-0.06, gamma[-1][1]-0.01, r'$\mathbf{g}$', fontsize=16)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xlim(-1.1, 0.25)
# plt.axis('equal')
# plt.legend(loc=(1.0, 0.5))
# plt.legend(loc='best')

# Print
plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--',color='orange', label='desired')
plt.plot(x_obst, y_obst, '.k', label='obstacle')
plt.plot(x_track_point_static[:, 0], x_track_point_static[:, 1], '-r', label='point static')
plt.plot(gamma[0][0], gamma[0][1] , '.k', markersize=10)
plt.text(gamma[0][0]-0.08, gamma[0][1]-0.01, r'$\mathbf{x}_0$', fontsize=16)
plt.plot(gamma[-1][0], gamma[-1][1], '.k', markersize=10)
plt.text(gamma[-1][0]-0.06, gamma[-1][1]-0.01, r'$\mathbf{g}$', fontsize=16)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xlim(-1.1, 0.25)

plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--',color='orange', label='desired')
plt.plot(x_obst, y_obst, '.k', label='obstacle')
plt.plot(x_track_point_dynamic[:, 0], x_track_point_dynamic[:, 1], color='purple', label='point static')
plt.plot(gamma[0][0], gamma[0][1] , '.k', markersize=10)
plt.text(gamma[0][0]-0.08, gamma[0][1]-0.01, r'$\mathbf{x}_0$', fontsize=16)
plt.plot(gamma[-1][0], gamma[-1][1], '.k', markersize=10)
plt.text(gamma[-1][0]-0.06, gamma[-1][1]-0.01, r'$\mathbf{g}$', fontsize=16)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xlim(-1.1, 0.25)

plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--',color='orange', label='desired')
plt.plot(x_obst, y_obst, '.k', label='obstacle')
plt.plot(x_track_point_steering[:, 0], x_track_point_steering[:, 1], color='green', label='point steering')
plt.plot(gamma[0][0], gamma[0][1] , '.k', markersize=10)
plt.text(gamma[0][0]-0.08, gamma[0][1]-0.01, r'$\mathbf{x}_0$', fontsize=16)
plt.plot(gamma[-1][0], gamma[-1][1], '.k', markersize=10)
plt.text(gamma[-1][0]-0.06, gamma[-1][1]-0.01, r'$\mathbf{g}$', fontsize=16)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xlim(-1.1, 0.25)

plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--',color='orange', label='desired')
plt.plot(np.append(x_obst,x_obst[0]), np.append(y_obst, y_obst[0]), '-k', label='obstacle')
plt.plot(x_track_static_volume[:, 0], x_track_static_volume[:, 1], color='blue', label='volume static')
plt.plot(gamma[0][0], gamma[0][1] , '.k', markersize=10)
plt.text(gamma[0][0]-0.08, gamma[0][1]-0.01, r'$\mathbf{x}_0$', fontsize=16)
plt.plot(gamma[-1][0], gamma[-1][1], '.k', markersize=10)
plt.text(gamma[-1][0]-0.06, gamma[-1][1]-0.01, r'$\mathbf{g}$', fontsize=16)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xlim(-1.1, 0.25)

plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--',color='orange', label='desired')
plt.plot(np.append(x_obst,x_obst[0]), np.append(y_obst, y_obst[0]), '-k', label='obstacle')
plt.plot(x_track_dynamic_volume[:, 0], x_track_dynamic_volume[:, 1], color='fuchsia', label='volume dynamic')
plt.plot(gamma[0][0], gamma[0][1] , '.k', markersize=10)
plt.text(gamma[0][0]-0.08, gamma[0][1]-0.01, r'$\mathbf{x}_0$', fontsize=16)
plt.plot(gamma[-1][0], gamma[-1][1], '.k', markersize=10)
plt.text(gamma[-1][0]-0.06, gamma[-1][1]-0.01, r'$\mathbf{g}$', fontsize=16)
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xlim(-1.1, 0.25)

# Trajectory error
plt.figure()
plt.plot(t_des, err_static_point, linestyle=(0, (1, 1)), color = 'red', label = '(00)')
plt.plot(t_des, err_dynamic_point, linestyle=(0, (3, 2)), color = 'purple', label = '(00)')
plt.plot(t_des, err_steering_point, linestyle=(0, (3, 2, 1, 2)), color = 'green', label = '(00)')
plt.plot(t_des, err_static_volume, linestyle=(0, (3, 2, 1, 2, 1, 2)), color = 'blue', label = '(00)')
plt.plot(t_des, err_dynamic_volume, '-', color = 'fuchsia', label = '(00)')
plt.xlabel(r'$t$', fontsize = 14)
plt.ylabel(r'err', fontsize = 14)
plt.legend(loc='best', fontsize=12)

# Acceleration
plt.figure()
t_sp = np.linspace(0.0, 1.0, np.shape(ddx_track_point_static)[0])
t_range = np.logical_and(t_sp < 0.9, t_sp > 0.4)
plt.plot(t_sp[t_range], np.linalg.norm(ddx_track_point_static, axis=1)[t_range], linestyle=(0, (1, 1)), color = 'red', label = '(00)')

t_sp = np.linspace(0.0, 1.0, np.shape(ddx_track_point_dynamic)[0])
t_range = np.logical_and(t_sp < 0.9, t_sp > 0.4)
plt.plot(t_sp[t_range], np.linalg.norm(ddx_track_point_dynamic, axis=1)[t_range], linestyle=(0, (3, 2)), color = 'purple', label = '(00)')

t_sp = np.linspace(0.0, 1.0, np.shape(ddx_track_point_steering)[0])
t_range = np.logical_and(t_sp < 0.9, t_sp > 0.4)
plt.plot(t_sp[t_range], np.linalg.norm(ddx_track_point_steering, axis=1)[t_range], linestyle=(0, (3, 2, 1, 2)), color = 'green', label = '(00)')

t_sp = np.linspace(0.0, 1.0, np.shape(ddx_track_static_volume)[0])
t_range = np.logical_and(t_sp < 0.9, t_sp > 0.4)
plt.plot(t_sp[t_range], np.linalg.norm(ddx_track_static_volume, axis=1)[t_range], linestyle=(0, (3, 2, 1, 2, 1, 2)), color = 'blue', label = '(00)')

t_sp = np.linspace(0.0, 1.0, np.shape(ddx_track_dynamic_volume)[0])
t_range = np.logical_and(t_sp < 0.9, t_sp > 0.4)
plt.plot(t_sp[t_range], np.linalg.norm(ddx_track_dynamic_volume, axis=1)[t_range], '-', color = 'fuchsia', label = '(00)')

plt.xlabel(r'$t$', fontsize = 14)
plt.ylabel(r'$\| \ddot{\mathbf{x}} (t) \|$', fontsize = 14)
plt.legend(loc='best', fontsize=12)
plt.ylim(0, 60)

plt.show()