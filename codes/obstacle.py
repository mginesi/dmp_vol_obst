"""
Copyright (C) 2018 Michele Ginesi
Copyright (C) 2018 Daniele Meli

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.	If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from dmp_cartesian import DMPs_cartesian

class obstacle(DMPs_cartesian):

	"""
	Implementation of an obstacle for Dynamic Movement Primitives as described in

	[1] Hoffmann, H., Pastor, P., Park, D. H., & Schaal, S. (2009, May).
	Biologically-inspired dynamical systems for movement generation: automatic
	real-time goal adaptation and obstacle avoidance. In Robotics and Automation,
	2009. ICRA'09. IEEE International Conference on (pp. 2587-2592). IEEE.
	"""

	def __init__(self, **kwargs):
		return

	def def_obstacle (self, x_obst = np.zeros(3), dx_obst = np.zeros(3), gamma_obst = 20., beta_obst = 10. / np.pi):
		"""
		Defining position, velocity and parameters of the repulsive force given by
		the obstacle [1]
		\dot \varphi = \gamma \phi ( -\beta |\varphi| )
		"""
		self.x_obst = x_obst
		self.dx_obst = dx_obst
		self.gamma_obst = gamma_obst
		self.beta_obst = beta_obst

	def rotation_matrix(self, theta, u, n_dim = 3):
		"""
		Compute the roation matrix of a rotation of theta around the direction given
		by u (if n_dim = 3)
		"""
		c = np.cos(theta)
		s = np.sin(theta)
		if (n_dim == 2):
			R = np.array([[c, -s], [s, c]])
		elif (n_dim == 3):
			if (np.shape(u)[0] != n_dim):
				raise ValueError ('dimension of u incompatible with n_dim')
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
		else:
			raise ValueError ('Invalid dimension')
		return R

	def gen_external_force(self, x, dx, goal, n_dim = 3):
		"""
		Computes the quantity p(x,v) which describes the perturbation of the DMP
		system.
		"""
		pval = 0. * self.x_obst
		# The repulsive force has to be computed only if we are moving
		if np.linalg.norm(dx) > 1e-5:
			# Computing the steering angle phi_i
			pos = self.x_obst - x # o_i - x
			# Rotate it by the direction we're going
			vel = dx - self.dx_obst # v - \dot o_i
			# Calculate the steering angle
			phi = np.arccos(np.dot(pos, vel) /
			                            (np.linalg.norm(pos) * np.linalg.norm(vel)))
			dphi = self.gamma_obst * phi * np.exp(-self.beta_obst * phi)
			r = np.cross(pos, dx)
			R = self.rotation_matrix(np.pi/2, r, n_dim)
			pval = np.dot(R, vel) * dphi
		return pval