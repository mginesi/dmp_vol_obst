import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn

# To use the codes in the main folder
import sys
sys.path.insert(0, 'codes/')
sys.path.insert(0, '../codes/')
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

import pdb

import dmp_cartesian

# DMP initialization
num_ts = 300
dmp = dmp_cartesian.DMPs_cartesian(n_dmps=3, n_bfs=40, K = 3050 * np.ones(3), dt = .001, alpha_s = 3.)
dmp.w = np.zeros([dmp.n_dmps, dmp.n_bfs + 1])
dmp.tol = 1e-02
x0 = np.array([0.519, -0.327, 0.084])
goal = np.array([0.614, 0.307, 0.083])
dmp.x0 = x0
dmp.goal = goal
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

x_track_s = dmp.x0
x_track[0, :] = dmp.x0

# Obstacles definition
x_c_1 = 0.512
y_c_1 = -0.152
z_c_1 = 0.179 - 0.063
a_1 = 0.09
b_1 = 0.02
c_1 = 0.07
x_c_2 = 0.659
x_c_2 = 0.512
y_c_2 = 0.12
z_c_2 = 0.12
a_2 = a_1
b_2 = b_1
c_2 = c_1
center_1 = np.array([x_c_1, y_c_1, z_c_1])
radii_1 = np.array([a_1, b_1, c_1])
coeffs_1 = np.array([1, 1, 2])
center_2 = np.array([x_c_2, y_c_2, z_c_2])
radii_2 = np.array([a_2, b_2, c_2])
coeffs_2 = np.array([1, 1, 2])
# Parameter of the potential
A = 50.0
eta = 1.0

# Function which generates the perturbation term
def perturbation(center, radii, coeffs, position, A, eta):
    isopotential = np.sum(((position - center) / radii) ** (2 * coeffs)) - 1.0 # C(x)
    disopotential = 2 * coeffs * ((position - center) / radii) ** (2 * coeffs - 1) / radii # gradient of C(x)
    phi = (A * np.exp(- eta * isopotential) * (- eta * disopotential) * isopotential - A * np.exp(- eta * isopotential) * disopotential) / isopotential / isopotential
    return - phi

x_track_s = dmp.x0
while (not flag):
    if (dmp.t == 0):
        dmp.first = True
    else:
        dmp.first = False
    # run and record timestep
    phi1 = perturbation(center_1, radii_1, coeffs_1, x_track_s, A, eta)
    phi2 = perturbation(center_2, radii_2, coeffs_2, x_track_s, A, eta)
    F = phi1 + phi2
    x_track_s, dx_track_s, ddx_track_s = dmp.step(external_force=F)
    x_track = np.append(x_track, [x_track_s], axis=0)
    dx_track = np.append(dx_track, [dx_track_s],axis=0)
    ddx_track = np.append(ddx_track, [ddx_track_s],axis=0)
    dmp.t += 1
    flag = (dmp.t >= dmp.cs.timesteps) & (np.linalg.norm(x_track_s - dmp.goal) / np.linalg.norm(dmp.goal - dmp.x0) <= dmp.tol)

## Figure with subplots

plt.figure()

plt.subplot(121)

# Plot of the E-E traj (using that it is a straight line)
x_plot_ee = np.array([x0[0] - radii_1[0] * np.sqrt(3) / 2, x0[0] + radii_1[0] * np.sqrt(3) / 2, goal[0] + radii_1[0] * np.sqrt(3) / 2, goal[0] - radii_1[0] * np.sqrt(3) / 2, x0[0] - radii_1[0] * np.sqrt(3) / 2])
y_plot_ee = np.array([x0[1] + radii_1[1] * np.sqrt(3) / 2, x0[1] - radii_1[1] * np.sqrt(3) / 2, goal[1] - radii_1[1] * np.sqrt(3) / 2, goal[1] + radii_1[1] * np.sqrt(3) / 2, x0[1] + radii_1[1] * np.sqrt(3) / 2])
plt.fill(x_plot_ee, y_plot_ee, 'k', alpha = 0.2)
# Plot of the E-E
x_rect = np.array([-1, -1, 1, 1]) * radii_1[0] * np.sqrt(3) / 2 + x0[0]
y_rect = np.array([-1, 1, 1, -1]) * radii_1[1] * np.sqrt(3) / 2 + x0[1]
plt.fill(x_rect, y_rect, color = 'gray')
x_rect = np.array([-1, -1, 1, 1]) * radii_2[0] * np.sqrt(3) / 2 + goal[0]
y_rect = np.array([-1, 1, 1, -1]) * radii_2[1] * np.sqrt(3) / 2 + goal[1]
plt.fill(x_rect, y_rect, color = 'gray')

# 2D plot (project on x_1 x_2 plane)
n_mesh = 200
theta = np.linspace(0, 2 * np.pi, n_mesh)
# plot the actual obstacles
plt.plot (center_1[0] + 0.01 * np.cos(theta), center_1[1] + 0.01 * np.sin(theta), '-k')
plt.plot (center_2[0] + 0.01 * np.cos(theta), center_2[1] + 0.01 * np.sin(theta), '-k')
# plot of the surrounding superquadric
plt.plot (center_1[0] + radii_1[0] * np.cos(theta), center_1[1] + radii_1[1] * np.sin(theta), ':r')
plt.plot (center_2[0] + radii_2[0] * np.cos(theta), center_2[1] + radii_2[1] * np.sin(theta), ':r')

plt.plot(x_classical[:, 0], x_classical[:, 1], '--b')

plt.plot(goal[0], goal[1], '.b')
plt.xlabel(r'$x_1$')
plt.ylabel(r'$x_2$')
plt.axis('equal')

plt.subplot(122)


# Plot of the E-E traj
x_plot_ee_1 = x_track[:, 0] + radii_1[0] * np.sqrt(3) / 2
y_plot_ee_1 = x_track[:, 1] - + radii_1[1] * np.sqrt(3) / 2
x_plot_ee_2 = np.array([goal[0] + radii_1[0]* np.sqrt(3) / 2, goal[0] - radii_1[0]* np.sqrt(3) / 2])
y_plot_ee_2 = np.array([goal[1] - radii_1[1] * np.sqrt(3) / 2, goal[1] - radii_1[1] * np.sqrt(3) / 2])
x_plot_ee_3 = np.flip(x_track[:,0]) - radii_1[0] * np.sqrt(3) / 2
y_plot_ee_3 = np.flip(x_track[:, 1]) + radii_1[1] * np.sqrt(3) / 2
x_plot_ee_4 = np.array([x0[0] - radii_1[0] * np.sqrt(3) / 2, x0[0] + radii_1[0] * np.sqrt(3) / 2])
y_plot_ee_4 = np.array([x0[1], x0[1]]) + radii_1[1] * np.sqrt(3) / 2
x_plot_ee = np.concatenate((x_plot_ee_1, x_plot_ee_2, x_plot_ee_3, x_plot_ee_4))
y_plot_ee = np.concatenate((y_plot_ee_1, y_plot_ee_2, y_plot_ee_3, y_plot_ee_4))
plt.fill(x_plot_ee, y_plot_ee, 'k', alpha = 0.2)

# Plot of the E-E
x_rect = np.array([-1, -1, 1, 1]) * radii_1[0] * np.sqrt(3) / 2 + x0[0]
y_rect = np.array([-1, 1, 1, -1]) * radii_1[1] * np.sqrt(3) / 2 + x0[1]
plt.fill(x_rect, y_rect, color = 'gray')
x_rect = np.array([-1, -1, 1, 1]) * radii_2[0] * np.sqrt(3) / 2 + goal[0]
y_rect = np.array([-1, 1, 1, -1]) * radii_2[1] * np.sqrt(3) / 2 + goal[1]
plt.fill(x_rect, y_rect, color = 'gray')
# 2D plot (project on x_1 x_2 plane)
n_mesh = 200
theta = np.linspace(0, 2 * np.pi, n_mesh)
# plot the actual obstacles
plt.plot (center_1[0] + 0.01 * np.cos(theta), center_1[1] + 0.01 * np.sin(theta), '-k')
plt.plot (center_2[0] + 0.01 * np.cos(theta), center_2[1] + 0.01 * np.sin(theta), '-k')
# plot of the surrounding superquadric
plt.plot (center_1[0] + radii_1[0] * np.cos(theta), center_1[1] + radii_1[1] * np.sin(theta), ':r')
plt.plot (center_2[0] + radii_2[0] * np.cos(theta), center_2[1] + radii_2[1] * np.sin(theta), ':r')
plt.plot(x_track[:, 0], x_track[:, 1], '-g')
plt.plot(goal[0], goal[1], '.b')
plt.xlabel(r'$x_1$')
plt.axis('equal')
plt.show()