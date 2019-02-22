"""
Copyright (C) 2018 Michele Ginesi
Copyright (C) 2018 Daniele Meli
Copyright (C) 2013 Travis DeWolf

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.    See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.    If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
from scipy import integrate
import scipy.sparse as sparse
import scipy.interpolate
import pdb

from cs import CanonicalSystem
from exponential_integration import exp_eul_step

class DMPs_cartesian(object):
    """
    Implementation of discrete dxnamic Movement Primitives in cartesian space,as
    described in
    [1] Park, D. H., Hoffmann, H., Pastor, P., & Schaal, S. (2008, December).
    Movement reproduction and obstacle avoidance with dxnamic movement primitives
    and potential fields. In Humanoid Robots, 2008. Humanoids 2008. 8th IEEE-RAS
    International Conference on (pp. 91-98). IEEE.
    [2] Hoffmann, H., Pastor, P., Park, D. H., & Schaal, S. (2009, May).
    Biologically-inspired dxnamical systems for movement generation: automatic
    real-time goal adaptation and obstacle avoidance. In Robotics and Automation,
    2009. ICRA'09. IEEE International Conference on (pp. 2587-2592). IEEE.
    """

    def __init__(self, n_dmps = 3, n_bfs = 50, dt = .01, x0 = None, goal = None, T = 1., K = None, D = None, tol = 0.1, alpha_s = 1., **kwargs):
        """
        n_dmps int   : number of dynamic movement primitives (i.e. dimensions)
        n_bfs int    : number of basis functions per DMP (actually, they will be one more)
        dt float     : timestep for simulation
        x0 np.array  : initial state of DMPs
        goal np.array: goal state of DMPs
        T float      : final time
        K float      : elastic parameter in the dynamical system
        D float      : damping parameter in the dynamical system
        tol float    : tolerance
        alpha_s float: constant of the Canonical System
        """
        # Tolerance for the accuracy of the movement: the trajectory will stop when
        # || x - g || <= tol
        self.tol = tol
        self.n_dmps = n_dmps
        self.n_bfs = n_bfs
        # Default values give as in [2]
        self.K = np.ones(n_dmps) * 150. if K is None else K
        self.D = 2 * np.sqrt(self.K) if D is None else D
        # Create the matrix of the linear component of the problem
        self.linear_part = np.zeros([2 * self.n_dmps, 2 * self.n_dmps])
        for d in range(n_dmps):
            self.linear_part[2 * d, 2 * d] = - self.D[d]
            self.linear_part[2 * d, 2 * d + 1] = - self.K[d]
            self.linear_part[2 * d + 1, 2 * d] = 1.
        # Set up the CS
        self.cs = CanonicalSystem(dt = dt, run_time = T, alpha_s = alpha_s)
        # Set up the DMP system
        if x0 is None:
            x0 = np.zeros(self.n_dmps)
        if goal is None:
            goal = np.zeros(self.n_dmps)
        self.x0 = x0
        self.goal = goal
        self.reset_state()
        self.gen_centers()
        self.gen_width()

    def compute_derivative_matrices(self):
        """
        Compute the matrices used to compute the derivatives
        """
        n = self.cs.timesteps
        ## D1 computation
        d1_p = np.ones([n - 1])
        d1_p[0] = 4.
        d1_m = - np.ones([n-1])
        d1_m[-1] = - 4.
        D1 = sparse.diags(np.array([d1_p, d1_m]), [1, -1]).toarray()
        D1[0,0] = - 3.
        D1[0, 2] = -1.
        D1[-1, -3] = 1.
        D1[-1,-1] = 3.
        D1 /= 2 * self.cs.dt
        ## D2 coputation
        d2_p = np.ones([n-1])
        d2_p[0] = -5.
        d2_m = np.ones([n-1])
        d2_m[-1] = -5.
        d2_c = - 2. * np.ones([n])
        d2_c[0] = 2.
        d2_c[-1] = 2.
        D2 = sparse.diags(np.array([d2_p, d2_c, d2_m]), [1, 0, -1]).toarray()
        D2[0, 2] = 4.
        D2[0, 3] = -1.
        D2[-1, -3] = 4.
        D2[-1, -4] = -1.
        D2 /= self.cs.dt * self.cs.dt
        return [D1, D2]

    def ext_forces(F):
        """
        Sums all the external forces.
        F: F_{i,j} = F ^ {(i)} _ {x_j}
        """
        tot_F = np.sum(F,0)
        return tot_F

    def gen_centers(self):
        """
        Set the centres of the basis functions to be spaced evenly throughout run
        time
        """
        # Desired activations throughout time
        self.c = np.exp(-self.cs.alpha_s * self.cs.run_time * ((np.cumsum(np.ones([1, self.n_bfs + 1])) - 1) / self.n_bfs))

    def gen_goal(self, x_des):
        """
        Generate the goal for path imitation.
        x_des np.array: the desired trajectory to follow
        """
        gl = np.copy(x_des[:, -1])
        return gl

    def gen_psi(self, s):
        """
        Generates the activity of the basis functions for a given
        canonical system rollout.
        x float, array: the canonical system state or path
        """
        c = np.transpose(np.array([self.c]))
        w = np.transpose(np.array([self.width]))
        xi = w * np.power((s - c), 2.0)
        psi_set = np.exp(- xi)
        return psi_set

    def gen_width(self):
        """
        Set the widths for the basis functions, given in such a way that each basis
        function is supported in only two consecutive intervals. This width does
        not depend on the number of basis functions
        """
        self.width = np.power(np.diff(self.c), -2.0)
        self.width = np.append(self.width, self.width[-1])

    def gen_weights(self, f_target):
            """Generate a set of weights over the basis functions such
            that the target forcing term trajectory is matched.
            f_target np.array: the desired forcing term trajectory
            """
            # calculate x and psi
            x_track = self.cs.rollout()
            psi_track = self.gen_psi(x_track)
            # efficiently calculate BF weights using weighted linear regression
            self.w = np.zeros((self.n_dmps, self.n_bfs + 1))
            for d in range(self.n_dmps):
                    # spatial scaling term
                    for b in range(self.n_bfs + 1):
                            numer = np.sum(x_track * psi_track[b, :] * f_target[d, :])
                            denom = np.sum(x_track**2 * psi_track[b, :])
                            self.w[d, b] = numer / denom
            self.w = np.nan_to_num(self.w)

    def imitate_path(self, x_des, t_des = None):
        """
        Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.
        x_des list/array: the desired trajectories of each DMP
                          should be shaped [n_dmps, run_time]
        """
        # Set initial state and goal
        if x_des.ndim == 1:
            x_des = x_des.reshape(1, len(x_des))
        self.x0 = x_des[:, 0].copy()
        self.goal = self.gen_goal(x_des)
        self.x_des = x_des.copy()
        # Generate function to interpolate the desired trajectory
        path = np.zeros((self.n_dmps, self.cs.timesteps))
        if t_des is None:
            t_des = np.linspace(0, self.cs.run_time, x_des.shape[1])
        else:
            t_des -= t_des[0]
            t_des /= t_des[-1]
            t_des *= self.cs.run_time
        time = np.linspace(0., self.cs.run_time, self.cs.timesteps)
        for d in range(self.n_dmps):
            # Piecewise linear interpolation
            path_gen = scipy.interpolate.interp1d(t_des, x_des[d]) # this is a function
            path[d, :] = path_gen(time)
        x_des = path
        # Second order estimates of the derivatives (the last non centered, all the
        # others centered)
        [D1, D2] = self.compute_derivative_matrices()
        dx_des = np.transpose(np.dot(D1, np.transpose(x_des)))
        ddx_des = np.transpose(np.dot(D2, np.transpose(x_des)))
        f_target = np.zeros([self.n_dmps, self.cs.timesteps])
        # Find the force required to move along this trajectory
        s_track = self.cs.rollout()
        for d in range(self.n_dmps):
            f_target[d, :] = ddx_des[d, :] / self.K[d] - (self.goal[d] - x_des[d, :]) + self.D[d] / self.K[d] * dx_des[d, :] + (self.goal[d] - self.x0[d]) * s_track
        self.gen_weights(f_target)
        self.reset_state()
        self.learned_position = self.goal - self.x0
        return f_target

    def reset_state(self):
        """
        Reset the system state
        """
        self.x = self.x0.copy()
        self.dx = np.zeros(self.n_dmps)
        self.ddx = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def rollout(self, tau = 1., **kwargs):
        """
        Generate a system trial, no feedback is incorporated.
        """
        self.reset_state()
        # Set up tracking vectors
        x_track = np.zeros((0, self.n_dmps))
        dx_track = np.zeros((0, self.n_dmps))
        ddx_track = np.zeros((0, self.n_dmps))
        x_track = np.append(x_track, [self.x0], axis = 0)
        dx_track = np.append(dx_track, [0.0 * self.x0], axis = 0)
        dx_track = np.append(ddx_track, [0.0 * self.x0], axis = 0)
        flag = False
        t = 0
        while (not flag):
            # Run and record timestep
            x_track_s, dx_track_s, ddx_track_s = self.step(tau = tau)
            x_track = np.append(x_track, [x_track_s], axis=0)
            dx_track = np.append(dx_track, [dx_track_s],axis=0)
            ddx_track = np.append(ddx_track, [ddx_track_s],axis=0)
            t += 1
            err_abs = np.linalg.norm(x_track_s - self.goal)
            err_rel = err_abs / (np.linalg.norm(self.goal - self.x0) + 1e-14)
            flag = ((t >= self.cs.timesteps) and err_rel <= self.tol)
        return x_track, dx_track, ddx_track

    def step(self, tau=1, error=0.0, external_force=None, **kwargs):
        """
        Run the DMP system for a single timestep.

        tau float: scales the timestep
                   increase tau to make the system execute faster
        error float: optional system feedback
        """
        error_coupling = 1.0 / (1.0 + error)
        # Run canonical system
        s = self.cs.step(tau = tau, error_coupling = error_coupling)
        # Generate basis function activation
        psi = self.gen_psi(s)
        f = np.zeros(self.n_dmps)
        state = np.zeros(2 * self.n_dmps)
        affine_part = np.zeros(2 * self.n_dmps)
        for d in range(self.n_dmps):
            # Generate the forcing term
            # FIXME In a brute force way, I set f = 0 when sum(psi) = 0, is there a
            # better way?
            if (np.sum(psi) <= 1e-15):
                f[d] = 0.
            else:
                f[d] = (np.dot(psi[:, 0], self.w[d, :])) / (np.sum(psi[:, 0])) * s
        for d in range(self.n_dmps):
            state[2*d] = self.dx[d]
            state[2*d+1] = self.x[d]
            affine_part[2*d] = self.K[d] * (self.goal[d] * (1. - s) + self.x0[d] * s + f[d])
            if external_force is not None:
                affine_part[2*d] += external_force[d]
            affine_part[2*d+1] = 0.
        state = exp_eul_step(state, self.linear_part, affine_part, self.cs.dt)
        for d in range(self.n_dmps):
            self.x[d] = state[2*d+1]
            self.dx[d] = state[2*d]
            self.ddx[d] = self.K[d] / (tau ** 2) * ((self.goal[d] - self.x[d]) - (self.goal[d] - self.x0[d]) * s + f[d]) - self.D[d] / tau * self.dx[d]
        return self.x, self.dx, self.ddx