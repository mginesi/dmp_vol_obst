"""
Ginesi et al. fig 4
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':'12'})
rc('text', usetex=True)

# Ellipse parameters
axis = [2.0, 1.0]
center = [0.5, 1.0]
coeffs = [1.0, 1.0]

# Particle velocity
vel = np.array([1.0, 1.0])

# Potential parameter
lmbda = 2.0
beta = 2.0

# Domain
x_span = np.linspace(-7, 5, 500)
y_span = np.linspace(-4, 3, 500)
[X, Y] = np.meshgrid(x_span, y_span)

isopotential = ((X - center[0]) / axis[0]) ** (2 * coeffs[0]) + ((Y - center[1]) / axis[1]) ** (2 * coeffs[1]) - 1.0
gradient_isopot = np.zeros([np.shape(X)[0], np.shape(X)[1], 2])
gradient_isopot[:, :, 0] = 2.0 * coeffs[0] * (X - center[0]) ** (2.0 * coeffs[0] - 1) / axis[0] ** (2 * coeffs[0])
gradient_isopot[:, :, 1] = 2.0 * coeffs[1] * (Y - center[1]) ** (2.0 * coeffs[1] - 1) / axis[1] ** (2 * coeffs[1])

cos_theta = np.dot(gradient_isopot, vel) / np.linalg.norm(vel) / np.linalg.norm(gradient_isopot, axis = 2)
theta = np.nan_to_num(np.arccos(cos_theta))

potential = lmbda * (- cos_theta) ** beta * np.linalg.norm(vel) / isopotential
potential[theta < np.pi / 2.0] = 0.0
potential[potential<0] = 0.0
potential[potential>1.0] = 1.0

# Plot
t_span = np.linspace(0.0, 2*np.pi)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, potential, cmap=cm.plasma)
ax.plot([-6, -4],[-4,-2],[0.8,0.8], '-k')
ax.set_zlim(0, 1)
ax.view_init(azim=-52, elev=56)
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
ax.set_zlabel(r'$U_D(\mathbf{x}, \mathbf{v})$', fontsize=14)
# plt.title('Dynamic Obstacle - potential')

plt.figure()
plt.contour(X, Y, potential, cmap=cm.plasma)
plt.plot(axis[0] * np.cos(t_span) + center[0], axis[1] * np.sin(t_span) + center[1], '-k')
plt.quiver(-1, -4, 3, 3, scale=25)
plt.xlabel(r'$x_1$', fontsize=14)
plt.ylabel(r'$x_2$', fontsize=14)
plt.axis('equal')
plt.show()
