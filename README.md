# README #

This repository contains the implementation of Dynamic Movement Primitives (DMPs), together with different approaches to obstacle avoidance, in Python 3.5.

In particular, this repository contains all the synthetic tests done for the following two publications:

_M. Ginesi, D. Meli, A. Calanca, D. Dall'Alba, N. Sansonetto and P. Fiorini_, **Dynamic Movement Primitives: Volumetric Obstacle Avoidance,** 2019 19th International Conference on Advanced Robotics (ICAR), Belo Horizonte, Brazil, 2019, pp. 234-239.

_Ginesi, M., Meli, D., Roberti, A. et al._, **Dynamic Movement Primitives: Volumetric Obstacle Avoidance Using Dynamic Potential Functions** J Intell Robot Syst 101, 79 (2021). https://doi.org/10.1007/s10846-021-01344-y

File _biblio.bib_ contains the bibentries of the works.
## Install ##

The package can be installed by running

```
pip install .
```

or

```
pip3 install .
```

## Repository Contents ##

The repository contains two folders: _codes_ and _demos_.

The _codes_ folder contains all the scripts performing the basic functions needed to implement DMPs (together with obstacle avoidance).

The _demos_ folder contains all the scripts performing the tests proposed in our works.
The folder contains two sub-folders, one for our work with static potentials, and one of the dynamic potentials.
See the comments at the beginning of the code to associate the test to the figure in the papers

## Theory: quick recall ##

_Dynamic Movement Primitives_ are a framework for trajectory learning. It is based on a second order differential equation of spring-mass-damper type:
$$
\begin{cases}
    \tau \dot{\mathbf{v}} = \mathbf{K} (\mathbf{g} - \mathbf{x}) - \mathbf{D} \mathbf{v} - \mathbf{K} (\mathbf{g} - \mathbf{x}_0) s + \mathbf{K} \mathbf{f}(s) \\
    \tau \dot{\mathbf{x}} = \mathbf{v}
\end{cases},
$$

where $\mathbf{x}, \mathbf{v} \in \mathbb{R}^d$ are position and velocity of the system; $\mathbf{x}_0 , \mathbf{g} \in \mathbb{R}^d$ are initial and goal position, respectively; $\mathbf{K}, \mathbf{D} \in \mathbb{R}^{d \times d}$ are diagonal matrices, representing the elastic and damping terms, respectively; $\mathbf{f} (s) \in \mathbb{R}^d $ is the "forcing term"; $\tau \in \mathbb{R}^+$ is a parameter used to make the execution faster or slower. Parameter $s \in \mathbb{R}$ is a re-parametrization of time, governed by the _canonical system_
$$ \tau \dot{s} = -\alpha s, \quad \alpha \in \mathbb{R}^+ . $$

### Learning Phase ###

During the learning phase, a desired curve $\tilde\mathbf{x}(t)$ is shown. This permit to compute the forcing term $\mathbf{f}(s)$. Thus, the forcing term is approximated using Gaussian Radial Basis functions $\{ \psi_i(s) \}_{i=0,1,\ldots, N}$ as
$$ \mathbf{f}(s) = \frac{ \sum_{i=0}^N \omega_i \psi_i(s) }{ \sum_{i=0}^N \psi_i(s) } s , $$
with $\omega_i \in \mathbb{R}$, $i=0,1,\ldots, N.$

### Execution Phase ###

Once the _weights_ $\omega_i$ have been learned, the forcing term can be computed. The dynamical system can be integrated changing $\mathbf{x}_0, \mathbf{g}$, and $\tau$, thus being able to generalize the trajectory changing initial and final positions, and execution time.

### Obstacle Avoidance ###

To avoid obstacles, a _copupling term_ $\bm{\varphi} (\mathbf{x}, \mathbf{v})$ is added to the first equation of the DMP system.
In our approaches, an isopotential
$$ C(\mathbf{x}) = \left( \left( \frac{x_1}{f_1(\mathbf{x})} \right) ^ {2n} + \left( \frac{x_2}{f_2(\mathbf{x})} \right) ^ {2n} \right) ^ \frac{2m}{2n} + \left( \frac{x_3}{f_3(\mathbf{x})} \right) ^ {2m} - 1 $$
that vanishes on the boundary of the obstacle is created by setting parameters $m,n \in \mathbb{N}$ and functions $f_i(\mathbf{x})$.

The **static potential** is created as
$$ U_S(\mathbf{x}) = \frac{A \exp (-\eta C(\mathbf{x})) }{ C(\mathbf{x}) }, $$
with $A, \eta \in \mathbb{R}^+$.
The coupling term is computed as
$$ \bm{\varphi} (\mathbf{x}, \mathbf{v}) \equiv \bm{\varphi} (\mathbf{x}) = - \nabla_\mathbf{x} U_S(\mathbf{x}) . $$

The *****dynamic potential***** is defined as
$$
U(\mathbf{x}) (\mathbf{x}, \mathbf{v}) =
\begin{cases}
\lambda ( -\cos \theta ) ^ \beta \dfrac{\Vert \mathbf{v} \Vert}{C^\eta (\mathbf{x})}
& \text{if } \theta \in \left( \frac{\pi}{2}, \pi \right] \\
0
& \text{if } \theta \in \left[0, \frac{\pi}{2} \right]
\end{cases},
$$
where $\theta$ is defined as
$$
\cos \theta = \frac{ \langle \nabla_\mathbf{x} C (\mathbf{x}) , \mathbf{v} \rangle }{ \Vert \nabla_\mathbf{x} C (\mathbf{x}) \Vert \, \Vert \mathbf{v} \Vert }.
$$
The coupling term is again defined as
$$ \bm{\varphi} (\mathbf{x}, \mathbf{v}) = - \nabla_\mathbf{x} U_D(\mathbf{x}) .$$

## References ##

[1] https://github.com/minillinim/ellipsoid

[2] _Hoffmann, H., Pastor, P., Park, D. H., and Schaal, S._. **Biologically-inspired dynamical systems for movement generation: automatic real-time goal adaptation and obstacle avoidance**. In Robotics and Automation, 2009. ICRA'09. IEEE International Conference on (pp. 2587-2592). IEEE.