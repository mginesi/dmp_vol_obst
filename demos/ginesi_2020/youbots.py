"""
Ginesi et al. 2020, fig 11
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc

fs = 16
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
    'size':fs})
rc('text', usetex=True)

savefig = False

c_1 = [0.7 , 0.15]
c_2 = [0.7 , -1.25]
ax = [0.6 , 0.45]

x_obst_1 = np.linspace(c_1[0] - ax[0], c_1[0] + ax[0])
y_obst_1_p = (1 - (( ( x_obst_1 - c_1[0] ) / ax[0] ) ** 4.0)) ** (1.0/4.0) * ax[1] + c_1[1]
y_obst_1_m = - ((1 - (( ( x_obst_1 - c_1[0] ) / ax[0] ) ** 4.0)) ** (1.0/4.0) * ax[1]) + c_1[1]

x_obst_2 = np.linspace(c_2[0] - ax[0], c_2[0] + ax[0])
y_obst_2_p = (1 - (( ( x_obst_2 - c_2[0] ) / ax[0] ) ** 4.0)) ** (1.0/4.0) * ax[1] + c_2[1]
y_obst_2_m = - ((1 - (( ( x_obst_2 - c_2[0] ) / ax[0] ) ** 4.0)) ** (1.0/4.0) * ax[1]) + c_2[1]

# ------------------------------------------------------------------------------------------- #
#  Static - null weights
# ------------------------------------------------------------------------------------------- #
data_nullw_static = np.load('data/null_static.npy')

trj_1 = data_nullw_static[:, [1, 2]]
trj_2 = data_nullw_static[:, [3, 4]]
trj_3 = data_nullw_static[:, [5, 6]]

plt.figure()
plt.plot(trj_1[:, 1], -trj_1[:, 0], 'r', label='YouBot0')
plt.plot(trj_2[:, 1], -trj_2[:, 0], 'g', label='YouBot1')
plt.plot(trj_3[:, 1], -trj_3[:, 0], 'b', label='YouBot2')
plt.plot(trj_1[0][1], -trj_1[0][0], '.r', markersize=7)
plt.plot(trj_2[0][1], -trj_2[0][0], '.g', markersize=7)
plt.plot(trj_3[0][1], -trj_3[0][0], '.b', markersize=7)
plt.plot(trj_1[-1][1], -trj_1[-1][0], '.r', markersize=7)
plt.plot(trj_2[-1][1], -trj_2[-1][0], '.g', markersize=7)
plt.plot(trj_3[-1][1], -trj_3[-1][0], '.b', markersize=7)
plt.plot([-2.0, 2.0, 2.0, -2.0, -2.0], [1.1, 1.1, -2.0, -2.0, 1.1], '-k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_p, 'k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_m, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_p, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_m, 'k', linewidth=1)
plt.text(-1.7, -1.5, r'$\mathbf{x}_0$', color='b')
plt.text(-1.7, -0.5, r'$\mathbf{x}_0$', color='g')
plt.text(-1.7, 0.5, r'$\mathbf{x}_0$', color='r')
plt.text(1.7, -1.5, r'$\mathbf{g}$', color='g')
plt.text(1.7, -0.5, r'$\mathbf{g}$', color='r')
plt.text(1.7, 0.5, r'$\mathbf{g}$', color='b')
plt.xlabel(r'$x[ \texttt{m} ]$')
plt.ylabel(r'$y[ \texttt{m} ]$')
plt.axis('scaled')
# plt.legend(loc='best')
if savefig:
    plt.savefig('imgs/youbots_static_nullw.eps')

t = np.linspace(0.0, 1.0, trj_1.shape[0])
plt.figure()
plt.subplot(211)
plt.plot(t, trj_1[:, 1], 'r')
plt.plot(t, trj_2[:, 1], 'g')
plt.plot(t, trj_3[:, 1], 'b')
plt.ylabel(r'$ x_1 $')
plt.gca().axes.xaxis.set_ticklabels([])
plt.subplot(212)
plt.plot(t, - trj_1[:, 0], 'r')
plt.plot(t, - trj_2[:, 0], 'g')
plt.plot(t, - trj_3[:, 0], 'b')
plt.ylabel(r'$ x_2 $')
plt.xlabel(r'$t$')
if savefig:
    plt.savefig('imgs/youbots_static_nullw_sol.eps')

# ------------------------------------------------------------------------------------------- #
#  Dynamic - null weights
# ------------------------------------------------------------------------------------------- #
data_nullw_dynamic = np.load('data/null_dynamic.npy')

trj_1 = data_nullw_dynamic[:, [1, 2]]
trj_2 = data_nullw_dynamic[:, [3, 4]]
trj_3 = data_nullw_dynamic[:, [5, 6]]

plt.figure()
plt.plot(trj_1[:, 1], -trj_1[:, 0], 'r', label='YouBot0')
plt.plot(trj_2[:, 1], -trj_2[:, 0], 'g', label='YouBot1')
plt.plot(trj_3[:, 1], -trj_3[:, 0], 'b', label='YouBot2')
plt.plot(trj_1[0][1], -trj_1[0][0], '.r', markersize=7)
plt.plot(trj_2[0][1], -trj_2[0][0], '.g', markersize=7)
plt.plot(trj_3[0][1], -trj_3[0][0], '.b', markersize=7)
plt.plot(trj_1[-1][1], -trj_1[-1][0], '.r', markersize=7)
plt.plot(trj_2[-1][1], -trj_2[-1][0], '.g', markersize=7)
plt.plot(trj_3[-1][1], -trj_3[-1][0], '.b', markersize=7)
plt.plot([-2.0, 2.0, 2.0, -2.0, -2.0], [1.1, 1.1, -2.0, -2.0, 1.1], '-k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_p, 'k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_m, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_p, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_m, 'k', linewidth=1)
plt.text(-1.7, -1.5, r'$\mathbf{x}_0$', color='b')
plt.text(-1.7, -0.5, r'$\mathbf{x}_0$', color='g')
plt.text(-1.7, 0.5, r'$\mathbf{x}_0$', color='r')
plt.text(1.7, -1.5, r'$\mathbf{g}$', color='g')
plt.text(1.7, -0.5, r'$\mathbf{g}$', color='r')
plt.text(1.7, 0.5, r'$\mathbf{g}$', color='b')
plt.xlabel(r'$x[ \texttt{m} ]$')
plt.ylabel(r'$y[ \texttt{m} ]$')
plt.axis('scaled')
# plt.legend(loc='best')
if savefig:
    plt.savefig('imgs/youbots_dynamic_nullw.eps')

t = np.linspace(0.0, 1.0, trj_1.shape[0])
plt.figure()
plt.subplot(211)
plt.plot(t, trj_1[:, 1], 'r')
plt.plot(t, trj_2[:, 1], 'g')
plt.plot(t, trj_3[:, 1], 'b')
plt.ylabel(r'$ x_1 $')
plt.gca().axes.xaxis.set_ticklabels([])
plt.subplot(212)
plt.plot(t, - trj_1[:, 0], 'r')
plt.plot(t, - trj_2[:, 0], 'g')
plt.plot(t, - trj_3[:, 0], 'b')
plt.ylabel(r'$ x_2 $')
plt.xlabel(r'$t$')
if savefig:
    plt.savefig('imgs/youbots_dynamic_nullw_sol.eps')

# ------------------------------------------------------------------------------------------- #
#  Static - constant velocity
# ------------------------------------------------------------------------------------------- #
data_const_static = np.load('data/const_static.npy')

trj_1 = data_const_static[:, [1, 2]]
trj_2 = data_const_static[:, [3, 4]]
trj_3 = data_const_static[:, [5, 6]]

plt.figure()
plt.plot(trj_1[:, 1], -trj_1[:, 0], 'r', label='YouBot0')
plt.plot(trj_2[:, 1], -trj_2[:, 0], 'g', label='YouBot1')
plt.plot(trj_3[:, 1], -trj_3[:, 0], 'b', label='YouBot2')
plt.plot(trj_1[0][1], -trj_1[0][0], '.r', markersize=7)
plt.plot(trj_2[0][1], -trj_2[0][0], '.g', markersize=7)
plt.plot(trj_3[0][1], -trj_3[0][0], '.b', markersize=7)
plt.plot(trj_1[-1][1], -trj_1[-1][0], '.r', markersize=7)
plt.plot(trj_2[-1][1], -trj_2[-1][0], '.g', markersize=7)
plt.plot(trj_3[-1][1], -trj_3[-1][0], '.b', markersize=7)
plt.plot([-2.0, 2.0, 2.0, -2.0, -2.0], [1.1, 1.1, -2.0, -2.0, 1.1], '-k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_p, 'k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_m, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_p, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_m, 'k', linewidth=1)
plt.text(-1.7, -1.5, r'$\mathbf{x}_0$', color='b')
plt.text(-1.7, -0.5, r'$\mathbf{x}_0$', color='g')
plt.text(-1.7, 0.5, r'$\mathbf{x}_0$', color='r')
plt.text(1.7, -1.5, r'$\mathbf{g}$', color='g')
plt.text(1.7, -0.5, r'$\mathbf{g}$', color='r')
plt.text(1.7, 0.5, r'$\mathbf{g}$', color='b')
plt.xlabel(r'$x[ \texttt{m} ]$')
plt.ylabel(r'$y[ \texttt{m} ]$')
plt.axis('scaled')
# plt.legend(loc='best')
if savefig:
    plt.savefig('imgs/youbots_static_const.eps')

t = np.linspace(0.0, 1.0, trj_1.shape[0])
plt.figure()
plt.subplot(211)
plt.plot(t, trj_1[:, 1], 'r')
plt.plot(t, trj_2[:, 1], 'g')
plt.plot(t, trj_3[:, 1], 'b')
plt.ylabel(r'$ x_1 $')
plt.gca().axes.xaxis.set_ticklabels([])
plt.subplot(212)
plt.plot(t, - trj_1[:, 0], 'r')
plt.plot(t, - trj_2[:, 0], 'g')
plt.plot(t, - trj_3[:, 0], 'b')
plt.ylabel(r'$ x_2 $')
plt.xlabel(r'$t$')
if savefig:
    plt.savefig('imgs/youbots_static_const_sol.eps')

# ------------------------------------------------------------------------------------------- #
#  Dynamic - constant velocity
# ------------------------------------------------------------------------------------------- #
data_const_dynamic = np.load('data/const_dynamic.npy')

trj_1 = data_const_dynamic[:, [1, 2]]
trj_2 = data_const_dynamic[:, [3, 4]]
trj_3 = data_const_dynamic[:, [5, 6]]

plt.figure()
plt.plot(trj_1[:, 1], -trj_1[:, 0], 'r', label='YouBot0')
plt.plot(trj_2[:, 1], -trj_2[:, 0], 'g', label='YouBot1')
plt.plot(trj_3[:, 1], -trj_3[:, 0], 'b', label='YouBot2')
plt.plot(trj_1[0][1], -trj_1[0][0], '.r', markersize=7)
plt.plot(trj_2[0][1], -trj_2[0][0], '.g', markersize=7)
plt.plot(trj_3[0][1], -trj_3[0][0], '.b', markersize=7)
plt.plot(trj_1[-1][1], -trj_1[-1][0], '.r', markersize=7)
plt.plot(trj_2[-1][1], -trj_2[-1][0], '.g', markersize=7)
plt.plot(trj_3[-1][1], -trj_3[-1][0], '.b', markersize=7)
plt.plot([-2.0, 2.0, 2.0, -2.0, -2.0], [1.1, 1.1, -2.0, -2.0, 1.1], '-k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_p, 'k', linewidth=1)
plt.plot(x_obst_1, y_obst_1_m, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_p, 'k', linewidth=1)
plt.plot(x_obst_2, y_obst_2_m, 'k', linewidth=1)
plt.text(-1.7, -1.5, r'$\mathbf{x}_0$', color='b')
plt.text(-1.7, -0.5, r'$\mathbf{x}_0$', color='g')
plt.text(-1.7, 0.5, r'$\mathbf{x}_0$', color='r')
plt.text(1.7, -1.5, r'$\mathbf{g}$', color='g')
plt.text(1.7, -0.5, r'$\mathbf{g}$', color='r')
plt.text(1.7, 0.5, r'$\mathbf{g}$', color='b')
plt.xlabel(r'$x[ \texttt{m} ]$')
plt.ylabel(r'$y[ \texttt{m} ]$')
plt.axis('scaled')
# plt.legend(loc='best')
if savefig:
    plt.savefig('imgs/youbots_dynamic_const.eps')

t = np.linspace(0.0, 1.0, trj_1.shape[0])
plt.figure()
plt.subplot(211)
plt.plot(t, trj_1[:, 1], 'r')
plt.plot(t, trj_2[:, 1], 'g')
plt.plot(t, trj_3[:, 1], 'b')
plt.ylabel(r'$ x_1 $')
plt.gca().axes.xaxis.set_ticklabels([])
plt.subplot(212)
plt.plot(t, - trj_1[:, 0], 'r')
plt.plot(t, - trj_2[:, 0], 'g')
plt.plot(t, - trj_3[:, 0], 'b')
plt.ylabel(r'$ x_2 $')
plt.xlabel(r'$t$')
if savefig:
    plt.savefig('imgs/youbots_dynamic_const_sol.eps')

plt.show()