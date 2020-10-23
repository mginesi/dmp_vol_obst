'''
Copyright (C) 2020 Michele Ginesi
Copyright (C) 2018 Daniele Meli

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
'''

import numpy as np
import copy

class Obstacle_Potential_Static():
    '''
    Implementation of an obstacle for Dynamic Movement Primitives as described
    in

    [1] O. Khatib,
        “Real-time obstacle avoidance for manipulators and fast mobile robots,” 
        International Journal of Robotics Research, vol. 5, p. 90, 1986.
    '''

    def __init__(self, x = np.zeros(3), p0 = 1.0, eta = 1.0, **kwargs):
        self.x_obst = copy.deepcopy(x)
        self.radius = copy.deepcopy(p0)
        self.gain = copy.deepcopy(eta)
        return

    def gen_external_force(self, x):
        p_x = np.linalg.norm(self.x_obst - x)
        if p_x > self.radius:
            phi_x = np.zeros_like(x)
        else:
            phi_x = self.gain * (1.0 / p_x - 1.0 / self.radius) * \
                (x - self.x_obst) / p_x ** 3.0
        return phi_x

class Obstacle_Potential_Dynamic():
    '''
    Implementation of an obstacle for Dynamic Movement Primitives as described
    in

    [1] Park, D.H., Hoffmann, H., Pastor, P., Schaal, S.
        Movement reproduction and obstacleavoidance with dynamic movement
        primitives and potential fields.
        In: Humanoid Robots,2008. Humanoids 2008.
        8th IEEE-RAS International Conference on, pp. 91–98. IEEE(2008)
    '''

    def __init__(self, x = np.zeros(3), lmbda = 1.0, beta = 2.0, **kwargs):
        self.x_obst = copy.deepcopy(x)
        self.gain = copy.deepcopy(lmbda)
        self.rate = copy.deepcopy(beta)
        return

    def gen_external_force(self, x, v):
        if np.linalg.norm(v) < 1e-14:
            phi_x = np.zeros_like(x)
        else:
            p_x = np.linalg.norm(x - self.x_obst)
            cos_theta = np.dot(v, x - self.x_obst) / np.linalg.norm(v) / p_x
            theta = np.arccos(cos_theta)
            if theta < np.pi / 2.0:
                phi_x = np.zeros_like(x)
            else:
                nabla_p = (x - self.x_obst) / p_x
                nabla_dot = v
                dot_p = np.dot(v, x - self.x_obst)
                nabla_cos = (p_x * nabla_dot - dot_p * nabla_p) / \
                    np.linalg.norm(v) / p_x / p_x
                phi_x = self.gain * (-cos_theta) ** self.rate * np.linalg.norm(v) * \
                    (self.gain * nabla_cos - cos_theta * nabla_p / p_x) / p_x
        return phi_x

class Obstacle_Steering():
    '''
    Implementation of an obstacle for Dynamic Movement Primitives as described
    in

    [1] Hoffmann, H., Pastor, P., Park, D. H., & Schaal, S. (2009, May).
        Biologically-inspired dynamical systems for movement generation:
        automatic real-time goal adaptation and obstacle avoidance.
        In Robotics and Automation, 2009. ICRA'09. IEEE International Conference
        on (pp. 2587-2592). IEEE.
    '''

    def __init__(self, x = np.zeros(3), gamma = 20.0, beta = 10.0 / np.pi, **kwargs):
        '''
        Initialize the obstacle object
        '''

        self.x_obst = copy.deepcopy(x)
        self.gamma = copy.deepcopy(gamma)
        self.beta = copy.deepcopy(beta)
        self.n_dim = np.size(self.x_obst)

    def rotation_matrix(self, theta, u):
        '''
        Compute the rotation matrix of a rotation of theta around the direction
        given by u (if self.n_dim = 3)
        '''

        c = np.cos(theta)
        s = np.sin(theta)
        if (self.n_dim == 2):
            R = np.array([[c, -s], [s, c]])
        elif (self.n_dim == 3):
            if (np.shape(u)[0] != self.n_dim):
                raise ValueError ('dimension of u incompatible with self.n_dim')
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

    def gen_external_force(self, x, dx, goal):
        '''
        Computes the quantity p(x,v) which describes the perturbation of the DMP
        system.
        '''

        pval = 0. * self.x_obst

        # The repulsive force has to be computed only if we are moving
        if np.linalg.norm(dx) > 1e-5:

            # Computing the steering angle phi_i
            pos = self.x_obst - x # o_i - x

            # Calculate the steering angle
            phi = np.arccos(np.dot(pos, dx) /
                (np.linalg.norm(pos) * np.linalg.norm(dx)))
            dphi = self.gamma * phi * np.exp(-self.beta * phi)
            r = np.cross(pos, dx)
            R = self.rotation_matrix(np.pi/2, r)
            pval = np.dot(R, dx) * dphi
            if not (np.abs(phi) < np.pi / 2.0):
                pval *= 0.0
            """
            # Check to see if the distance to the obstacle is further than
            # the distance to the target. If it is, ignore the obstacle
            if np.linalg.norm(pos) > np.linalg.norm(goal - x):
                pval = 0. * self.x_obst
            # Check if we are risking to collide with the obstacle. If not,
            # ignore the obstacle
            if not ((phi <= np.pi / 2) & (phi >= - np.pi/2)):
                pval = 0 * self.x_obst
            """
        return pval