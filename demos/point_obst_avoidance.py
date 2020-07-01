import numpy as np
from dmp.dmp import DMPs_cartesian as dmp_c
from dmp.point_obstacle import Obstacle_Potential_Static as obst_static
from dmp.point_obstacle import Obstacle_Potential_Dynamic as obst_dyn
from dmp.point_obstacle import Obstacle_Steering as obst_steer
import matplotlib.pyplot as plt

# Obstacle properties
obst_pos = np.array([0, 1])

# Static obstacle
obst_s = obst_static(x = obst_pos, p0 = 1.0, eta = 1.0)

# Dynamic obstacle
obst_d = obst_dyn(x = obst_pos, lmbda=1.0, beta=2.0)

# Steering obstacle
obst_sa = obst_steer(x=obst_pos, gamma=50.0, beta=1.0)

# Trajectory
t = np.linspace(-1.0, 1.0, 300)
gamma = np.transpose([t, (1 + t) * (1 - t)])

MP = dmp_c(n_dmps=2, K = 1050.0, dt = 0.002)
MP.imitate_path(t_des=t, x_des=gamma)

## Execution with obstacle avoidance
# Static Potential
MP.reset_state()
x_track_static = np.zeros([0, 2])
def f_sp(x, v):
    return obst_s.gen_external_force(x)
while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
    MP.step(external_force=f_sp, adapt=True)
    x_track_static = np.append(x_track_static, np.array([MP.x]), axis=0)

# Dynamic Potential
MP.reset_state()
x_track_dynamic = np.zeros([0, 2])
def f_dp(x, v):
    return obst_d.gen_external_force(x, v)
while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
    MP.step(external_force=f_dp, adapt=True)
    x_track_dynamic = np.append(x_track_dynamic, np.array([MP.x]), axis=0)

# Steering Angle
MP.reset_state()
x_track_steering = np.zeros([0, 2])
def f_sa(x, v):
    return obst_sa.gen_external_force(x, v, MP.x_goal)
while np.linalg.norm(MP.x - MP.x_goal) > MP.tol:
    MP.step(external_force=f_sa, adapt=True)
    x_track_steering = np.append(x_track_steering, np.array([MP.x]), axis=0)

plt.figure()
plt.plot(gamma[:, 0], gamma[:, 1], '--k', label='desired')
plt.plot(x_track_static[:, 0], x_track_static[:, 1], '-r', label='static pot')
plt.plot(x_track_dynamic[:, 0], x_track_dynamic[:, 1], 'g', label='dynamic pot')
plt.plot(x_track_steering[:, 0], x_track_steering[:, 1], 'b', label='steering angle')
plt.plot(obst_pos[0], obst_pos[1], '.k', markersize=10)
plt.legend(loc='best')
plt.show()