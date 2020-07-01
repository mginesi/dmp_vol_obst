"""
Ginesi et al. 2019, fig 2
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.pyplot as plt
import pdb

from mpl_toolkits.mplot3d import Axes3D
import seaborn

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# To use the codes in the main folder
import sys
sys.path.insert(0, 'codes/')
sys.path.insert(0, '../codes/')

from dmp.dmp_cartesian import DMPs_cartesian as dyn_mp
from dmp.obstacle_superquadric import Obstacle_Static as obst
from dmp.ellipsoid import EllipsoidTool
ET = EllipsoidTool()

# Initializing the DMP
dmp = dyn_mp(n_dmps=2, n_bfs=50, K = 1000 * np.ones(2),dt = .01, alpha_s = 3.)

def rotation_matrix(u):
    """
    Compute the roation matrix of a rotation of pi/2 around the direction given
    by u (if n_dim = 3)
    """
    c = 0.0
    s = 1.0
    # u has to be a versor in the l2-norm
    u = u / np.linalg.norm(u)
    x = u[0]
    y = u[1]
    z = u[2]
    C = 1. - c
    R = np.array([
        [(x * x * C + c), (x * y * C - z * s), (x * z * C + y * s)],
        [(y * x * C + z * s), (y * y * C + c), (y * z * C - x * s)],
        [(z * x * C - y * s), (z * y * C + x * s), (z * z * C + c)]])
    return R

def mesh_parallelepid(x_length, y_length, z_length, num_points, noise_var = 0.0):
    # The following code make a rotated version. To solve it we just re-arrange the parameters
    width = y_length
    height = x_length
    depth = z_length
    # First step: create a uniform mesh on the 2D area
    # To do so, we consider the fact that the 2D "cover" is composed of two copies
    # for each shape combination (width x height, height x depth, depth x width)
    area_wh = width * height
    area_hd = height * depth
    area_dw = depth * width
    tot_surface_area = 2 * (area_dw + area_hd + area_wh)
    # We simulate a uniform distribution for each face, with the number of point
    # proportional to its face
    n_point_wh = int(num_points * (area_wh / tot_surface_area))
    n_point_hd = int(num_points * (area_hd / tot_surface_area))
    n_point_dw = int(num_points / 2 - (n_point_wh + n_point_hd)) # we discard a point if num_points is odd :(
    # Create the distributions
    # Uniform in [0, 1] x [0,1]
    distr_wh_1 = np.random.rand(n_point_wh, 2)
    distr_wh_2 = np.random.rand(n_point_wh, 2)
    distr_hd_1 = np.random.rand(n_point_hd, 2)
    distr_hd_2 = np.random.rand(n_point_hd, 2)
    distr_dw_1 = np.random.rand(n_point_dw, 2)
    distr_dw_2 = np.random.rand(n_point_dw, 2)
    # Rescale
    distr_wh_1 *= np.array([width, height])
    distr_wh_2 *= np.array([width, height])
    distr_hd_1 *= np.array([height, depth])
    distr_hd_2 *= np.array([height, depth])
    distr_dw_1 *= np.array([depth, width])
    distr_dw_2 *= np.array([depth, width])
    # Now the sets of randomly sampled points need to be mapped onto the relative faces
    # Initialize the six faces
    face_wh_1 = np.zeros([n_point_wh, 3])
    face_wh_2 = np.zeros([n_point_wh, 3])
    face_hd_1 = np.zeros([n_point_hd, 3])
    face_hd_2 = np.zeros([n_point_hd, 3])
    face_dw_1 = np.zeros([n_point_dw, 3])
    face_dw_2 = np.zeros([n_point_dw, 3])
    # Map the point to the faces
    face_wh_1[:, [1, 0]] = distr_wh_1
    face_wh_2[:, [1, 0]] = distr_wh_2
    face_wh_2[:, 2] = depth
    face_hd_1[:, [0,2]] = distr_hd_1
    face_hd_2[:, [0,2]] = distr_hd_2
    face_hd_2[:, 1] = width
    face_dw_1[:, [2, 1]] = distr_dw_1
    face_dw_2[:, [2, 1]] = distr_dw_2
    face_dw_2[:, 0] = height
    point_set = np.zeros([0, 3])
    point_set = np.append(point_set, face_wh_1, axis = 0)
    point_set = np.append(point_set, face_wh_2, axis = 0)
    point_set = np.append(point_set, face_hd_1, axis = 0)
    point_set = np.append(point_set, face_hd_2, axis = 0)
    point_set = np.append(point_set, face_dw_1, axis = 0)
    point_set = np.append(point_set, face_dw_2, axis = 0)
    return point_set

x_length = 2
y_length = 3
z_length = 1

solid = mesh_parallelepid(x_length, y_length, z_length, 200)
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(solid[:, 0], solid[:, 1], solid[:, 2])
ax.set_xlabel (r'$x$')
ax.set_ylabel (r'$y$')
ax.set_zlabel (r'$z$')
ax.set_xlim(0, max([x_length, y_length, z_length]))
ax.set_ylim(0, max([x_length, y_length, z_length]))
ax.set_zlim(0, max([x_length, y_length, z_length]))
# Test on the ellipse generator
center, radii, rotation = ET.getMinVolEllipse(solid, 0.01)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# plot points
ax.scatter(solid[:,0], solid[:,1], solid[:,2], color='g', marker='*', s=100)
# plot ellipsoid
ET.plotEllipsoid(center, radii, rotation, ax=ax, plotAxes=True)


range_n_p = (np.ceil(np.power(2, np.linspace(3, 9, 12)))).astype(int)
time_point_obst = np.zeros(len(range_n_p))
time_vol_obst = np.zeros(len(range_n_p))
time_vol_obst_no_ellipse = np.zeros(len(range_n_p))
# Position and velocities: only for testing purpose
pos = np.array([x_length, y_length, z_length]) * 2.0 # position at which computing the forcing term
vel = np.random.rand(3) # velocity
# Parameters for the potential
# solid
param_n = 1.0
param_A = 1.0
param_eta = 1.0
# steering angle
param_gamma = 1.0
param_beta = 1.0
import time
count = 0
tot_num_test = 30
for num_points in range_n_p:
    for num_test in range(tot_num_test):
        # Create the parallelipedic obstacle
        solid = mesh_parallelepid(x_length, y_length, z_length, num_points)
        # Computation time for volumetric obstacle
        t_vol_ellipse = time.time()
        # The computation of the smallest enclosing ellipse does not use any closef formula.
        # For generality, the point cloud is considered generic.
        center, radii, rotation = ET.getMinVolEllipse(solid, 0.01)
        # Compute forcing term using superquadric
        t_vol = time.time()
        phi = ((pos - center) ** (2 * param_n - 1)) / (radii ** (2 * param_n))
        K = np.sum((pos - center / radii) ** (2 * param_n)) - 1
        phi *= (param_A * np.exp(- param_eta * K)) * (param_eta / K + 1. / K / K) * (2 * param_n)
        time_vol_obst[count] += time.time() - t_vol_ellipse # in seconds
        time_vol_obst_no_ellipse[count] += time.time() - t_vol # in seconds
        # Computation time for volumetric obstacle
        t = time.time()
        # Compute the forcing term
        phi = np.zeros(3)
        for i in range(len(solid)):
            # Rotation matrix
            u = (np.cross(solid[i] - pos, vel))
            R = rotation_matrix(u)
            theta = np.arccos(np.dot(solid[i] - pos, vel) / np.linalg.norm(solid[i] - pos) / np.linalg.norm(vel))
            phi += param_gamma * np.dot(R, vel) * theta * np.exp (- param_beta * theta)
        time_point_obst[count] += time.time() - t # in seconds
        # Increase the counter
    count += 1

# To obtain the mean time, divide by the number of tests
time_vol_obst /= tot_num_test
time_vol_obst_no_ellipse /= tot_num_test
time_point_obst /= tot_num_test

# Plot
lw = 1
ms = 5
fs = 14
plt.figure()
plt.loglog(range_n_p, time_vol_obst, 'ob--', lw = lw, markersize = ms, label = 'Volumetric computing the ellipse')
plt.loglog(range_n_p, time_vol_obst_no_ellipse, '^g--', lw = lw, markersize = ms, label = 'Volumetric not computing the ellipse')
plt.loglog(range_n_p, time_point_obst, '*r--', lw = lw, markersize = ms, label = 'Point sum')
plt.legend(loc = 'best')
plt.show()