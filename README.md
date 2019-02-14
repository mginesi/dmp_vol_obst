# README #

This repository contains the implementation of Dynamic Movement Primitives (DMPs), together with two different approaches to obstacle avoidance, in Python 3.5.

In particular, this repository contains all the synthetic tests done for the work (currently under revision):

_Ginesi M., Meli D., Calanca A., Dall'Alba D., Sansonetto N., and Fiorini P._; **Dynamic Movement Primitives: Volumetric Obstacle Avoidance**

## Repository Contents ##

The repository contains two folders: _codes_ and _demos_.

### The _codes_ folder ###

The _codes_ folder contains all the scripts performing the basis functions needed to implement DMPs (together with obstacle avoidance).
In particular:
* _cs.py_ implements the so-called "Canonical System";
* _dmp_cartesian.py_ is the class that generates and handle the DMP, able to handle both the learning of the weights given a trajectory and to execute a trajectory given the set of weights;
* _ellipsoid.py_ implements the method using to extract the minimum volume enclosing ellipsoid algorithm [1];
* _exponential_integration.py_ contains the functions needed to perform the numerical integration method "Exponential Euler";
* _obstacle.py_ is the class which implements obstacle avoidance for point-like obstacles using the method presented in [2];
* _obstacle_ellipse.py_ is the class which implements volumentric obstacle avoidance for ellipses (2 dimensions) and ellipsoids (3 dimensions)

### The _demos_ folder ###

The _demos_ folder contains all the scripts performing the tests proposed in our work.
In particular:
* _one_obstacle.py_ test both the steering angle method [2] and our proposed approach are with a single obstacle;
* _plot_ellipsoid_to_rectangle.py_ shows how a rectangle can be approximated using an $n$-ellipsoid
* _point_cloud_test.py_ tests the computational times between:
* * using each point of a point cloud as obstacle,
* * extract the minimum volume enclosing ellipsoid and using it as a single volumetric obstacle;
* _synthetic_test.py_ tests our approach considering the volume occupied by the end-effector of the robot;
* _two_obstacles.py_ tests both the steering angle method [2] and our proposed approach with two obstacles;

## References ##

[1] https://github.com/minillinim/ellipsoid

[2] _Hoffmann, H., Pastor, P., Park, D. H., and Schaal, S._. **Biologically-inspired dynamical systems for movement generation: automatic real-time goal adaptation and obstacle avoidance**. In Robotics and Automation, 2009. ICRA'09. IEEE International Conference on (pp. 2587-2592). IEEE.