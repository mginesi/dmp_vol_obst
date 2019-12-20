"""
Copyright (C) 2018 Michele Ginesi
Copyright (C) 2018 Daniele Meli

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
from dmp.dmp_cartesian import DMPs_cartesian

class obstacle(DMPs_cartesian):

    """
    Implementation of an obstacle for Dynamic Movement Primitives as described
    in

    [1] Hoffmann, H., Pastor, P., Park, D. H., & Schaal, S. (2009, May).
        Biologically-inspired dynamical systems for movement generation:
        automatic real-time goal adaptation and obstacle avoidance.
        In Robotics and Automation, 2009. ICRA'09. IEEE International Conference
        on (pp. 2587-2592). IEEE.
    """

    def __init__(self, x_obst = np.zeros(3), dx_obst = np.zeros(3),
            gamma_obst = 20.0, beta_obst = 10.0 / np.pi, **kwargs):
        """
        Initialize the obstacle object
        """
        self.x_obst = x_obst
        self.dx_obst = dx_obst
        self.gamma_obst = gamma_obst
        self.beta_obst = beta_obst
        self.n_dim = np.size(self.x_obst)

    def rotation_matrix(self, theta, u):
        """
        Compute the roation matrix of a rotation of theta around the direction
        given by u (if self.n_dim = 3)
        """
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
            R = self.rotation_matrix(np.pi/2, r)
            pval = np.dot(R, vel) * dphi
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

    if __name__ is "__main__":
        import matplotlib.pyplot as plt
        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
        rc('text', usetex=True)

        from dmp_cartesian import DMPs_cartesian as dmp
        from point_obstacle import obstacle as obstacle
        import copy

        ## Create the trajectory to learn
        t = np.linspace(0, np.pi, 1000)
        x_des = np.transpose(
            np.array([
            t * np.cos(t) / np.pi,
            t * np.sin(t) / np.pi
            ]))

        ## Create the DMP
        dmp_1 = dmp(K=1000 * np.ones(2), n_bfs=40, n_dmps=2, dt=0.005, alpha_s=3.0, tol=2e-02)
        dmp_1.imitate_path(t_des=t, x_des=x_des)
        dmp_1.reset_state()

        ## Create the obstacles
        t_s = np.linspace(0, 2 * np.pi, 50)
        obst_list_1 = []
        obst_list_2 = []
        for n in range(50):
            obst = obstacle(x_obst = np.array([-0.5 + 0.3*np.cos(t_s[n]), 0.7 + 0.2*np.sin(t_s[n])]), dx_obst = np.zeros(2))
            obst_list_1.append(obst)

        for n in range(50):
            obst = obstacle(x_obst = np.array([0.15 + 0.1*np.cos(t_s[n]), 0.4 + 0.1*np.sin(t_s[n])]), dx_obst = np.zeros(2))
            obst_list_2.append(obst)
        x_plot_obst_1 = -0.5 + 0.3 * np.cos(np.linspace(0, 2*np.pi, 50))
        y_plot_obst_1 = 0.7 + 0.2 * np.sin(np.linspace(0, 2*np.pi, 50))
        x_plot_obst_2 = 0.15 + 0.1 * np.cos(np.linspace(0, 2*np.pi, 50))
        y_plot_obst_2 = 0.4 + 0.1 * np.sin(np.linspace(0, 2*np.pi, 50))

        x_track = np.zeros([0, 2])
        F_1_track = np.zeros([0, 2])
        F_2_track = np.zeros([0, 2])
        ## Execution
        flag_convergence = False
        while (not flag_convergence):
            # Integration step
            F_1 = np.zeros(2)
            F_2 = np.zeros(2)
            for i in range(50):
                F_1 += obst_list_1[i].gen_external_force(dmp_1.x, dmp_1.dx, dmp_1.goal)
                F_2 += obst_list_2[i].gen_external_force(dmp_1.x, dmp_1.dx, dmp_1.goal)
            F_1_track = np.append(F_1_track, [copy.deepcopy(F_1)], axis = 0)
            F_2_track = np.append(F_2_track, [copy.deepcopy(F_2)], axis = 0)
            dmp_1.step(external_force=F_1 + F_2)
            x_track = np.append(x_track, [dmp_1.x], axis=0)
            # Plotting
            plt.clf()
            plt.subplot(122)
            plt.plot(x_des[:, 0], x_des[:, 1], '--b')
            plt.plot(x_track[:, 0], x_track[:, 1], '--r')
            plt.plot(x_plot_obst_1, y_plot_obst_1, '.k')
            plt.plot(x_plot_obst_2, y_plot_obst_2, '.k')
            plt.plot(dmp_1.x[0], dmp_1.x[1], '.r')
            norm_v = dmp_1.dx / np.linalg.norm(dmp_1.dx) * 0.1
            plt.arrow(dmp_1.x[0], dmp_1.x[1], norm_v[0], norm_v[1],
                color = 'r', lw = 2)
            plt.subplot(221)
            plt.plot(F_1_track[:, 0], label = 'x')
            plt.plot(F_1_track[:, 1], label = 'y')
            plt.legend(loc = 'best')
            plt.grid()
            plt.subplot(223)
            plt.plot(F_2_track[:, 0], label = 'x')
            plt.plot(F_2_track[:, 1], label = 'y')
            plt.legend(loc = 'best')
            plt.grid()
            plt.pause(0.01)
            # Stopping condition
            flag_convergence = np.linalg.norm(dmp_1.x - dmp_1.goal) <= dmp_1.tol \
                and dmp_1.cs.s <= np.exp(-dmp_1.cs.alpha_s * dmp_1.cs.run_time) + dmp_1.tol

        plt.show()