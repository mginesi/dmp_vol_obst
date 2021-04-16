"""
Ginesi et al. 2020, fig 13
"""


import numpy as np
from dmp.dmp import DMPs_cartesian as dmp_cart
from dmp.obstacle_superquadric import Obstacle_Static, Obstacle_Dynamic

# Plot stuff
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':14})
rc('text', usetex=True)

savefig = False
saveframes = False

# DMP parameters
K = 1050.0
alpha_s = 4.0
tol = 0.01

goal_move_to_peg = np.array([0.488574563751,
                            -0.184923096437,
                            0.262908357465])

goal_move_to_ring = np.array([0.505577739181,
                              0.310174815877,
                              0.045214734967])

x0_move_to_peg = goal_move_to_ring

x0_move_to_ring = np.array([0.307, 0, 0.483])

MP = dmp_cart(K=K, alpha_s=alpha_s, tol=tol)

# Obstacle parameters
peg_1_c = np.array([0.487796955824,
                    -0.238666226814,
                    0.153926015924 - 0.065])
peg_2_c = np.array([0.487796955824,
                    -0.238666226814+0.315,
                    0.153926015924 - 0.065])
peg_3_c = np.array([0.487796955824+0.28,
                    -0.238666226814+0.15,
                    0.153926015924 - 0.065])
peg_4_c = np.array([0.487796955824+0.28,
                    -0.238666226814+0.48,
                    0.153926015924 - 0.065])
base = np.array([0.487796955824+0.14,
                 -0.238666226814+0.24,
                 0.153926015924-0.13-0.02])

axes_move_to_ring = np.array([0.008, 0.008, 0.065])
axes_move_to_peg = np.array([0.128, 0.128, 0.065])
axes_base = np.array([0.35, 0.35, 0.02])

dyn_lambda = 10.0
dyn_eta = 1.0
dyn_beta = 1.0

stat_A = 10.0
stat_eta = 1.0

# ---------------------------------------------------------------------------------------------
#  Obstacles and potentials creations
# ---------------------------------------------------------------------------------------------

# to-ring , Static
peg_1_obst_static_toring = Obstacle_Static(center=peg_1_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
peg_2_obst_static_toring = Obstacle_Static(center=peg_2_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
peg_3_obst_static_toring = Obstacle_Static(center=peg_3_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
peg_4_obst_static_toring = Obstacle_Static(center=peg_4_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
base_obst_static_toring = Obstacle_Static(center=base, axis=axes_base, coeffs=np.array([1, 1, 2]), A=stat_A, eta=stat_eta)

def perturb_static_toring(x, v):
    out = peg_1_obst_static_toring.gen_external_force(x) + \
        peg_2_obst_static_toring.gen_external_force(x) + \
        peg_3_obst_static_toring.gen_external_force(x) + \
        peg_4_obst_static_toring.gen_external_force(x) + \
        base_obst_static_toring.gen_external_force(x)
    return out

# to-ring , Dynamic
peg_1_obst_dynamic_toring = Obstacle_Dynamic(center=peg_1_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
peg_2_obst_dynamic_toring = Obstacle_Dynamic(center=peg_2_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
peg_3_obst_dynamic_toring = Obstacle_Dynamic(center=peg_3_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
peg_4_obst_dynamic_toring = Obstacle_Dynamic(center=peg_4_c, axis=axes_move_to_ring, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
base_obst_dynamic_toring = Obstacle_Dynamic(center=base, axis=axes_base, coeffs=np.array([1, 1, 2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)

def perturb_dynamic_toring(x, v):
    out = peg_1_obst_dynamic_toring.gen_external_force(x, v) + \
        peg_2_obst_dynamic_toring.gen_external_force(x, v) + \
        peg_3_obst_dynamic_toring.gen_external_force(x, v) + \
        peg_4_obst_dynamic_toring.gen_external_force(x, v) + \
        base_obst_dynamic_toring.gen_external_force(x, v)
    return out

# to-peg , Static
peg_1_obst_static_topeg = Obstacle_Static(center=peg_1_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
peg_2_obst_static_topeg = Obstacle_Static(center=peg_2_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
peg_3_obst_static_topeg = Obstacle_Static(center=peg_3_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
peg_4_obst_static_topeg = Obstacle_Static(center=peg_4_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), A=stat_A, eta=stat_eta)
base_obst_static_topeg = Obstacle_Static(center=base, axis=axes_base, coeffs=np.array([1, 1, 2]), A=stat_A, eta=stat_eta)

def perturb_static_topeg(x, v):
    out = peg_1_obst_static_topeg.gen_external_force(x) + \
        peg_2_obst_static_topeg.gen_external_force(x) + \
        peg_3_obst_static_topeg.gen_external_force(x) + \
        peg_4_obst_static_topeg.gen_external_force(x) + \
        base_obst_static_topeg.gen_external_force(x)
    return out

# to-peg , Dynamic
peg_1_obst_dynamic_topeg = Obstacle_Dynamic(center=peg_1_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
peg_2_obst_dynamic_topeg = Obstacle_Dynamic(center=peg_2_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
peg_3_obst_dynamic_topeg = Obstacle_Dynamic(center=peg_3_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
peg_4_obst_dynamic_topeg = Obstacle_Dynamic(center=peg_4_c, axis=axes_move_to_peg, coeffs=np.array([1,1,2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)
base_obst_dynamic_topeg = Obstacle_Dynamic(center=base, axis=axes_base, coeffs=np.array([1, 1, 2]), lmbda=dyn_lambda, beta=dyn_beta, eta=dyn_eta)

def perturb_dynamic_topeg(x, v):
    out = peg_1_obst_dynamic_topeg.gen_external_force(x, v) + \
        peg_2_obst_dynamic_topeg.gen_external_force(x, v) + \
        peg_3_obst_dynamic_topeg.gen_external_force(x, v) + \
        peg_4_obst_dynamic_topeg.gen_external_force(x, v) + \
        base_obst_dynamic_topeg.gen_external_force(x, v)
    return out

# ---------------------------------------------------------------------------------------------
#  DMP evolution
# ---------------------------------------------------------------------------------------------

# to-ring, static
MP.x_0 = x0_move_to_ring
MP.x_goal = goal_move_to_ring
MP.reset_state()
x_track_static_toring = np.array([MP.x_0])
flag_conv = False
while not flag_conv:
    MP.step(external_force=perturb_static_toring)
    x_track_static_toring = np.append(x_track_static_toring, np.array([MP.x]), axis=0)
    flag_conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol

# to-ring, dynamic
MP.x_0 = x0_move_to_ring
MP.x_goal = goal_move_to_ring
MP.reset_state()
x_track_dynamic_toring = np.array([MP.x_0])
flag_conv = False
while not flag_conv:
    MP.step(external_force=perturb_dynamic_toring)
    x_track_dynamic_toring = np.append(x_track_dynamic_toring, np.array([MP.x]), axis=0)
    flag_conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol

# to-peg, static
MP.x_0 = x0_move_to_peg
MP.x_goal = goal_move_to_peg
MP.reset_state()
x_track_static_topeg = np.array([MP.x_0])
ddx_track_static_topeg = np.array(np.zeros([0, 3]))
flag_conv = False
while not flag_conv:
    MP.step(external_force=perturb_static_topeg)
    x_track_static_topeg = np.append(x_track_static_topeg, np.array([MP.x]), axis=0)
    ddx_track_static_topeg = np.append(ddx_track_static_topeg, np.array([MP.ddx]), axis=0)
    flag_conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol

# to-peg, dynamic
MP.x_0 = x0_move_to_peg
MP.x_goal = goal_move_to_peg
MP.reset_state()
x_track_dynamic_topeg = np.array([MP.x_0])
ddx_track_dynamic_topeg = np.array(np.zeros([0, 3]))
flag_conv = False
while not flag_conv:
    MP.step(external_force=perturb_dynamic_topeg)
    x_track_dynamic_topeg = np.append(x_track_dynamic_topeg, np.array([MP.x]), axis=0)
    ddx_track_dynamic_topeg = np.append(ddx_track_dynamic_topeg, np.array([MP.ddx]), axis=0)
    flag_conv = np.linalg.norm(MP.x - MP.x_goal) < MP.tol

# ---------------------------------------------------------------------------------------------
#  Plotting
# ---------------------------------------------------------------------------------------------

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid

# Static to ring
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.auto_scale_xyz
Xc,Yc,Zc = data_for_cylinder_along_z(base[0], base[1], axes_base[0], 2*axes_base[2])
Zc += 0.5 * base[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color='gray')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_1_c[0], peg_1_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_1_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'green')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_2_c[0], peg_2_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_2_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'red')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_3_c[0], peg_3_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_3_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'yellow')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_4_c[0], peg_4_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_4_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'blue')

ax.plot(x_track_static_toring[:, 0], x_track_static_toring[:, 1], x_track_static_toring[:, 2],'-k', linewidth=1.5)

ax.plot(x0_move_to_ring[0], x0_move_to_ring[1], x0_move_to_ring[2], '.k', markersize=5)
ax.plot(goal_move_to_ring[0], goal_move_to_ring[1], goal_move_to_ring[2], '.k', markersize=5)
ax.text(0.3, 0.0, 0.5, r'$\mathbf{x}_0$')
ax.text(0.5, 0.3, 0.1, r'$\mathbf{g}$')

ax.set_xlabel(r'$x[\texttt{m}]$')
ax.set_ylabel(r'$y[\texttt{m}]$')
ax.set_zlabel(r'$z[\texttt{m}]$')

ax.view_init(elev=34, azim=16)

if savefig:
    plt.savefig('imgs/panda_static_toring.png')

# Dynamic to ring
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.auto_scale_xyz
Xc,Yc,Zc = data_for_cylinder_along_z(base[0], base[1], axes_base[0], 2*axes_base[2])
Zc += 0.5 * base[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color='gray')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_1_c[0], peg_1_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_1_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'green')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_2_c[0], peg_2_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_2_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'red')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_3_c[0], peg_3_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_3_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'yellow')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_4_c[0], peg_4_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_4_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'blue')

ax.plot(x_track_dynamic_toring[:, 0], x_track_dynamic_toring[:, 1], x_track_dynamic_toring[:, 2],'-k', linewidth=1.5)

ax.plot(x0_move_to_ring[0], x0_move_to_ring[1], x0_move_to_ring[2], '.k', markersize=5)
ax.plot(goal_move_to_ring[0], goal_move_to_ring[1], goal_move_to_ring[2], '.k', markersize=5)
ax.text(0.3, 0.0, 0.5, r'$\mathbf{x}_0$')
ax.text(0.5, 0.3, 0.1, r'$\mathbf{g}$')

ax.set_xlabel(r'$x[\texttt{m}]$')
ax.set_ylabel(r'$y[\texttt{m}]$')
ax.set_zlabel(r'$z[\texttt{m}]$')

ax.view_init(elev=34, azim=16)

if savefig:
    plt.savefig('imgs/panda_dynamic_toring.png')

if saveframes:
    for n_fr in range(x_track_dynamic_toring.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.auto_scale_xyz
        Xc,Yc,Zc = data_for_cylinder_along_z(base[0], base[1], axes_base[0], 2*axes_base[2])
        Zc += 0.5 * base[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color='gray')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_1_c[0], peg_1_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_1_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'green')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_2_c[0], peg_2_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_2_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'red')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_3_c[0], peg_3_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_3_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'yellow')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_4_c[0], peg_4_c[1], axes_move_to_ring[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_4_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'blue')

        ax.plot(x_track_dynamic_toring[:n_fr, 0], x_track_dynamic_toring[:n_fr, 1], x_track_dynamic_toring[:n_fr, 2],'-k', linewidth=1.5)

        ax.plot(x0_move_to_ring[0], x0_move_to_ring[1], x0_move_to_ring[2], '.k', markersize=5)
        ax.plot(goal_move_to_ring[0], goal_move_to_ring[1], goal_move_to_ring[2], '.k', markersize=5)
        ax.text(0.3, 0.0, 0.5, r'$\mathbf{x}_0$')
        ax.text(0.5, 0.3, 0.1, r'$\mathbf{g}$')

        ax.set_xlabel(r'$x[\texttt{m}]$')
        ax.set_ylabel(r'$y[\texttt{m}]$')
        ax.set_zlabel(r'$z[\texttt{m}]$')

        ax.view_init(elev=34, azim=16)
        plt.savefig('frames/panda_dynamic_toring'+ str(1000 + n_fr) +'.png')

# Static to peg
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.auto_scale_xyz
Xc,Yc,Zc = data_for_cylinder_along_z(base[0], base[1], axes_base[0], 2*axes_base[2])
Zc += 0.5 * base[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color='gray')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_1_c[0], peg_1_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_1_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'green')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_2_c[0], peg_2_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_2_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'red')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_3_c[0], peg_3_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_3_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'yellow')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_4_c[0], peg_4_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_4_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'blue')

ax.plot(x_track_static_topeg[:, 0], x_track_static_topeg[:, 1], x_track_static_topeg[:, 2],'-k', linewidth=1.5)

ax.plot(x0_move_to_peg[0], x0_move_to_peg[1], x0_move_to_peg[2], '.k', markersize=5)
ax.plot(goal_move_to_peg[0], goal_move_to_peg[1], goal_move_to_peg[2], '.k', markersize=5)

ax.text(0.5, 0.3, 0.1, r'$\mathbf{x}_0$')
ax.text(0.5, -0.2, 0.3, r'$\mathbf{g}$')

ax.set_xlabel(r'$x[\texttt{m}]$')
ax.set_ylabel(r'$y[\texttt{m}]$')
ax.set_zlabel(r'$z[\texttt{m}]$')

ax.view_init(elev=34, azim=16)

if savefig:
    plt.savefig('imgs/panda_static_topeg.png')

# Dynamic to peg
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.auto_scale_xyz
Xc,Yc,Zc = data_for_cylinder_along_z(base[0], base[1], axes_base[0], 2*axes_base[2])
Zc += 0.5 * base[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color='gray')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_1_c[0], peg_1_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_1_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'green')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_2_c[0], peg_2_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_2_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'red')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_3_c[0], peg_3_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_3_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'yellow')
Xc,Yc,Zc = data_for_cylinder_along_z(peg_4_c[0], peg_4_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
Zc += 0.5 * peg_4_c[2]
ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'blue')

ax.plot(x_track_dynamic_topeg[:, 0], x_track_dynamic_topeg[:, 1], x_track_dynamic_topeg[:, 2],'-k', linewidth=1.5)

ax.plot(x0_move_to_peg[0], x0_move_to_peg[1], x0_move_to_peg[2], '.k', markersize=5)
ax.plot(goal_move_to_peg[0], goal_move_to_peg[1], goal_move_to_peg[2], '.k', markersize=5)

ax.text(0.5, 0.3, 0.1, r'$\mathbf{x}_0$')
ax.text(0.5, -0.2, 0.3, r'$\mathbf{g}$')

ax.set_xlabel(r'$x[\texttt{m}]$')
ax.set_ylabel(r'$y[\texttt{m}]$')
ax.set_zlabel(r'$z[\texttt{m}]$')

ax.view_init(elev=34, azim=16)

if savefig:
    plt.savefig('imgs/panda_dynamic_topeg.png')

if saveframes:
    for n_fr in range(x_track_dynamic_toring.shape[0]):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.auto_scale_xyz
        Xc,Yc,Zc = data_for_cylinder_along_z(base[0], base[1], axes_base[0], 2*axes_base[2])
        Zc += 0.5 * base[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color='gray')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_1_c[0], peg_1_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_1_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'green')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_2_c[0], peg_2_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_2_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'red')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_3_c[0], peg_3_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_3_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'yellow')
        Xc,Yc,Zc = data_for_cylinder_along_z(peg_4_c[0], peg_4_c[1], axes_move_to_peg[0], 2*axes_move_to_peg[2])
        Zc += 0.5 * peg_4_c[2]
        ax.plot_surface(Xc, Yc, Zc, alpha=0.5, color = 'blue')

        ax.plot(x_track_dynamic_topeg[:n_fr, 0], x_track_dynamic_topeg[:n_fr, 1], x_track_dynamic_topeg[:n_fr, 2],'-k', linewidth=1.5)

        ax.plot(x0_move_to_peg[0], x0_move_to_peg[1], x0_move_to_peg[2], '.k', markersize=5)
        ax.plot(goal_move_to_peg[0], goal_move_to_peg[1], goal_move_to_peg[2], '.k', markersize=5)

        ax.text(0.5, 0.3, 0.1, r'$\mathbf{x}_0$')
        ax.text(0.5, -0.2, 0.3, r'$\mathbf{g}$')

        ax.set_xlabel(r'$x[\texttt{m}]$')
        ax.set_ylabel(r'$y[\texttt{m}]$')
        ax.set_zlabel(r'$z[\texttt{m}]$')

        ax.view_init(elev=34, azim=16)
        plt.savefig('frames/panda_dynamic_topeg'+ str(1000+n_fr) +'.png')
        plt.close()

# plt.figure()
# plt.subplot(311)
# plt.plot(ddx_track_static_topeg[:, 0])
# plt.subplot(312)
# plt.plot(ddx_track_static_topeg[:, 1])
# plt.subplot(313)
# plt.plot(ddx_track_static_topeg[:, 2])

# plt.figure()
# plt.subplot(311)
# plt.plot(ddx_track_dynamic_topeg[:, 0])
# plt.subplot(312)
# plt.plot(ddx_track_dynamic_topeg[:, 1])
# plt.subplot(313)
# plt.plot(ddx_track_dynamic_topeg[:, 2])

if not saveframes:
    plt.show()