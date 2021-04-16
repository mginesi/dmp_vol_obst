"""
Ginesi et al. 2020, fig 22
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

from dmp.dmp import DMPs_cartesian as dmp_c
from dmp.obstacle_superquadric import Obstacle_Static, Obstacle_Dynamic

fs = 14
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':fs})
rc('text', usetex=True)

## Setup
# Parameters
x0 = np.array([0.0, 0.0])
xg = np.array([-0.1, 2.0])

obst = np.array([0.0, 1.0])
axes = np.array([0.4, 0.5])

K = 500.0
alpha = 4.0

A_stat = 1.0
eta_stat = 1.0

lmbda_d = 1.0
beta_d = 1.0
eta_d = 1.0

tau = 100

t_presence = [0.2, 0.5]
t_presence = [t_presence[i] * tau for i in range(len(t_presence))]

print_err = False
savefig = False
savetraj = False

saveframe = True

# MP
MP = dmp_c(n_dmps=2, x_0=x0, x_goal=xg, K=K, alpha_s=alpha)
x_des = np.array([np.linspace(x0[0], xg[0], 200),
                    np.linspace(x0[1], xg[1], 200)]).transpose()
t_des = np.linspace(0, 1, 200)
MP.imitate_path(x_des=x_des, t_des=t_des)
MP.cs.dt *= tau
x_track_no_obst, _, _, t_track_no_obst = MP.rollout(tau=tau)

# Obstacle
obst_stat = Obstacle_Static(center=obst, axis=axes, A=A_stat, eta=eta_stat)
def perturb_stat(x, v):
    return obst_stat.gen_external_force(x)

obst_dyn = Obstacle_Dynamic(center=obst, axis=axes, lmbda=lmbda_d, beta=beta_d, eta=eta_d)
def perturb_dyn(x, v):
    return obst_dyn.gen_external_force(x, v)

## Simulations
# Static - always present
MP.reset_state()
x_track_static = np.array([MP.x_0])
t_static = [0]
t = 0
while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
    if t <= t_presence[0]:
        MP.step(adapt=True, tau=tau)
    else:
        MP.step(external_force=perturb_stat, adapt=True, tau=tau)
    t += MP.cs.dt
    x_track_static = np.append(x_track_static, np.array([MP.x]), axis=0)
    t_static.append(t_static[-1] + MP.cs.dt)
    if print_err:
        print(np.linalg.norm(MP.x - MP.x_goal))

t_static_temporal_always = np.asarray(t_static)
t_range_static_always = np.where(t_static_temporal_always > t_presence[0])[0]
x_static_range_always = x_track_static[t_range_static_always]

# Dynamic - always present
MP.reset_state()
x_track_dynamic = np.array([MP.x_0])
t_dynamic = [0]
t = 0
while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
    if t <= t_presence[0]:
        MP.step(adapt=True, tau=tau)
    else:
        MP.step(external_force=perturb_dyn, adapt=True, tau=tau)
    t += MP.cs.dt
    x_track_dynamic = np.append(x_track_dynamic, np.array([MP.x]), axis=0)
    t_dynamic.append(t_dynamic[-1] + MP.cs.dt)
    if print_err:
        print(np.linalg.norm(MP.x - MP.x_goal))

t_dynamic_temporal_always = np.asarray(t_dynamic)
t_range_dynamic_always = np.where(t_dynamic_temporal_always > t_presence[0])[0]
x_dynamic_range_always = x_track_dynamic[t_range_dynamic_always]

# Static - temporal presence
MP.reset_state()
x_track_static_temporal = np.array([MP.x_0])
t = 0
t_static_temporal = [0]
while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
    if t > t_presence[0] and t < t_presence[1]:
        MP.step(external_force=perturb_stat, adapt=True, tau=tau)
    else:
        MP.step(adapt=True, tau=tau)
    t += MP.cs.dt
    t_static_temporal.append(t)
    x_track_static_temporal = np.append(x_track_static_temporal, np.array([MP.x]), axis=0)
    if print_err:
        print(np.linalg.norm(MP.x - MP.x_goal))

t_static_temporal = np.asarray(t_static_temporal)
t_range_static = np.where(np.logical_and((t_static_temporal > t_presence[0]) , (t_static_temporal < t_presence[1])))[0]
x_static_range = x_track_static_temporal[t_range_static]

# Dynamic - temporal presence
MP.reset_state()
x_track_dynamic_temporal = np.array([MP.x_0])
t = 0
t_dynamic_temporal = [0]
while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
    if t > t_presence[0] and t < t_presence[1]:
        MP.step(external_force=perturb_dyn, adapt=True, tau=tau)
    else:
        MP.step(adapt=True, tau=tau)
    t += MP.cs.dt
    t_dynamic_temporal.append(t)
    x_track_dynamic_temporal = np.append(x_track_dynamic_temporal, np.array([MP.x]), axis=0)
    if print_err:
        print(np.linalg.norm(MP.x - MP.x_goal))

t_dynamic_temporal = np.asarray(t_dynamic_temporal)
t_range_dynamic = np.where(np.logical_and((t_dynamic_temporal > t_presence[0]) , (t_dynamic_temporal < t_presence[1])))[0]
x_dynamic_range = x_track_dynamic_temporal[t_range_dynamic]

## Plot
from matplotlib.patches import Ellipse as plot_ellipse

plt.figure()
plt.gca().add_patch(plot_ellipse(obst, axes[0], axes[1], edgecolor='black', facecolor='gray'))
plt.plot(x_track_static[:, 0], x_track_static[:, 1], 'r', linewidth=2)
plt.plot(x_static_range_always[:, 0], x_static_range_always[:, 1], 'g', linewidth=2)
plt.plot(x_track_no_obst[:, 0], x_track_no_obst[:, 1], '--b', linewidth=1)
plt.plot(x0[0], x0[1], '.k')
plt.plot(xg[0], xg[1], '.k')
plt.text(x0[0] + 0.1, x0[1], r'$ \mathbf{x}_0 $', fontsize=fs)
plt.text(xg[0] + 0.1, xg[1], r'$ \mathbf{g} $', fontsize=fs)
plt.xlabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.axis('scaled')
plt.xlim(-0.75, 0.75)
if savefig:
    plt.savefig('imgs/youbot_static_vconst_always.eps')

plt.figure()
plt.subplot(211)
plt.plot(t_static, x_track_static[:, 0], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 0], '--b')
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.ylabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.subplot(212)
plt.plot(t_static, x_track_static[:, 1], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 1], '--b')
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.xlabel(r'$t[\texttt{s}]$', fontsize=fs)
if savefig:
    plt.savefig('imgs/youbot_static_vconst_always_sol.eps')

plt.figure()
plt.gca().add_patch(plot_ellipse(obst, axes[0], axes[1], edgecolor='black', facecolor='gray'))
plt.plot(x_track_dynamic[:, 0], x_track_dynamic[:, 1], 'r', linewidth=2)
plt.plot(x_dynamic_range_always[:, 0], x_dynamic_range_always[:, 1], 'g', linewidth=2)
plt.plot(x_track_no_obst[:, 0], x_track_no_obst[:, 1], '--b', linewidth=1)
plt.plot(x0[0], x0[1], '.k')
plt.plot(xg[0], xg[1], '.k')
plt.text(x0[0] + 0.1, x0[1], r'$ \mathbf{x}_0 $', fontsize=fs)
plt.text(xg[0] + 0.1, xg[1], r'$ \mathbf{g} $', fontsize=fs)
plt.xlabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.axis('scaled')
plt.xlim(-0.75, 0.75)
if savefig:
    plt.savefig('imgs/youbot_dynamic_vconst_always.eps')

plt.figure()
plt.subplot(211)
plt.plot(t_dynamic, x_track_dynamic[:, 0], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 0], '--b')
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.ylabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.subplot(212)
plt.plot(t_dynamic, x_track_dynamic[:, 1], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 1], '--b')
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.xlabel(r'$t[\texttt{s}]$', fontsize=fs)
if savefig:
    plt.savefig('imgs/youbot_dynamic_vconst_always_sol.eps')

if False:
    for n in range(int(x_track_dynamic.shape[0] / 10) - 1):
        plt.figure()
        plt.gca().add_patch(plot_ellipse(obst, axes[0], axes[1], edgecolor='black', facecolor='gray'))
        plt.plot(x_track_dynamic[:10*n, 0], x_track_dynamic[:10*n, 1], 'r', linewidth=2)
        plt.plot(x_track_no_obst[:, 0], x_track_no_obst[:, 1], '--b', linewidth=1)
        plt.plot(x0[0], x0[1], '.k')
        plt.plot(xg[0], xg[1], '.k')
        plt.text(x0[0] + 0.1, x0[1], r'$ \mathbf{x}_0 $', fontsize=fs)
        plt.text(xg[0] + 0.1, xg[1], r'$ \mathbf{g} $', fontsize=fs)
        plt.xlabel(r'$x[\texttt{m}]$', fontsize=fs)
        plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
        plt.axis('scaled')
        plt.xlim(-0.75, 0.75)
        plt.savefig('frames/youbot_dynamic_vconst_always' + str(1000 + n) + '.png')
        plt.close()

plt.figure()
plt.gca().add_patch(plot_ellipse(obst, axes[0], axes[1], edgecolor='black', facecolor='gray'))
plt.plot(x_track_static_temporal[:, 0], x_track_static_temporal[:, 1], 'r', linewidth=2)
plt.plot(x_static_range[:, 0], x_static_range[:, 1], 'g', linewidth=2)
plt.plot(x_track_no_obst[:, 0], x_track_no_obst[:, 1], '--b', linewidth=1)
plt.plot(x0[0], x0[1], '.k')
plt.plot(xg[0], xg[1], '.k')
plt.text(x0[0] + 0.1, x0[1], r'$ \mathbf{x}_0 $', fontsize=fs)
plt.text(xg[0] + 0.1, xg[1], r'$ \mathbf{g} $', fontsize=fs)
plt.xlabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.axis('scaled')
plt.xlim(-0.75, 0.75)
if savefig:
    plt.savefig('imgs/youbot_static_vconst_temporal.eps')

plt.figure()
plt.subplot(211)
plt.plot(t_static_temporal, x_track_static_temporal[:, 0], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 0], '--b')
plt.ylabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.axvline(x=t_presence[1], color='k', linestyle='--')
plt.subplot(212)
plt.plot(t_static_temporal, x_track_static_temporal[:, 1], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 1], '--b')
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.axvline(x=t_presence[1], color='k', linestyle='--')
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.xlabel(r'$t[\texttt{s}]$', fontsize=fs)
if savefig:
    plt.savefig('imgs/youbot_static_vconst_temporal_sol.eps')

plt.figure()
plt.gca().add_patch(plot_ellipse(obst, axes[0], axes[1], edgecolor='black', facecolor='gray'))
plt.plot(x_track_dynamic_temporal[:, 0], x_track_dynamic_temporal[:, 1], 'r', linewidth=2)
plt.plot(x_dynamic_range[:, 0], x_dynamic_range[:, 1], 'g', linewidth=2)
plt.plot(x_track_no_obst[:, 0], x_track_no_obst[:, 1], '--b', linewidth=1)
plt.plot(x0[0], x0[1], '.k')
plt.plot(xg[0], xg[1], '.k')
plt.text(x0[0] + 0.1, x0[1], r'$ \mathbf{x}_0 $', fontsize=fs)
plt.text(xg[0] + 0.1, xg[1], r'$ \mathbf{g} $', fontsize=fs)
plt.xlabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.axis('scaled')
plt.xlim(-0.75, 0.75)
if savefig:
    plt.savefig('imgs/youbot_dynamic_vconst_temporal.eps')

if saveframe:
    for n in range(int(x_track_dynamic_temporal.shape[0] / 10) - 1):
        plt.figure()
        plt.gca().add_patch(plot_ellipse(obst, axes[0], axes[1], edgecolor='black', facecolor='gray'))
        plt.plot(x_track_dynamic_temporal[:10*n, 0], x_track_dynamic_temporal[:10*n, 1], 'r', linewidth=2)
        plt.plot(x_track_no_obst[:, 0], x_track_no_obst[:, 1], '--b', linewidth=1)
        plt.plot(x0[0], x0[1], '.k')
        plt.plot(xg[0], xg[1], '.k')
        plt.text(x0[0] + 0.1, x0[1], r'$ \mathbf{x}_0 $', fontsize=fs)
        plt.text(xg[0] + 0.1, xg[1], r'$ \mathbf{g} $', fontsize=fs)
        plt.xlabel(r'$x[\texttt{m}]$', fontsize=fs)
        plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
        plt.axis('scaled')
        plt.xlim(-0.75, 0.75)
        plt.savefig('frames/youbot_dynamic_vconst_temporal' + str(1000 + n) + '.png')
        plt.close()

plt.figure()
plt.subplot(211)
plt.plot(t_dynamic_temporal, x_track_dynamic_temporal[:, 0], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 0], '--b')
plt.ylabel(r'$x[\texttt{m}]$', fontsize=fs)
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.axvline(x=t_presence[1], color='k', linestyle='--')
plt.subplot(212)
plt.plot(t_dynamic_temporal, x_track_dynamic_temporal[:, 1], 'r', linewidth=2)
plt.plot(t_track_no_obst, x_track_no_obst[:, 1], '--b')
plt.ylabel(r'$y[\texttt{m}]$', fontsize=fs)
plt.xlabel(r'$t[\texttt{s}]$', fontsize=fs)
plt.axvline(x=t_presence[0], color='k', linestyle='--')
plt.axvline(x=t_presence[1], color='k', linestyle='--')
if savefig:
    plt.savefig('imgs/youbot_dynamic_vconst_temporal_sol.eps')

if not saveframe:
    plt.show()

# Trajectroy expor
if savetraj:
    from scipy.interpolate import interp1d
    traj_interp_v_const_static_always = interp1d(t_static, x_track_static.transpose())
    traj_v_const_static_always = np.transpose(traj_interp_v_const_static_always(np.linspace(t_static[0], t_static[-1], 100)))
    np.save('v_const_static_always', traj_v_const_static_always)
    traj_interp_v_const_dynamic_always = interp1d(t_dynamic, x_track_dynamic.transpose())
    traj_v_const_dynamic_always = np.transpose(traj_interp_v_const_dynamic_always(np.linspace(t_dynamic[0], t_dynamic[-1], 100)))
    np.save('v_const_dynamic_always', traj_v_const_dynamic_always)
    traj_interp_v_const_static_range = interp1d(t_static_temporal, x_track_static_temporal.transpose())
    traj_v_const_static_range = np.transpose(traj_interp_v_const_static_range(np.linspace(t_static_temporal[0], t_static_temporal[-1], 100)))
    np.save('v_const_static_range', traj_v_const_static_range)
    traj_interp_v_const_dynamic_range = interp1d(t_dynamic_temporal, x_track_dynamic_temporal.transpose())
    traj_v_const_dynamic_range = np.transpose(traj_interp_v_const_dynamic_range(np.linspace(t_dynamic_temporal[0], t_dynamic_temporal[-1], 100)))
    np.save('v_const_dynamic_range', traj_v_const_dynamic_range)