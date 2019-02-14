import numpy as np
import matplotlib.pyplot as plt
import pdb

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

basis = 4.
height = 3.

rectangle_x = np.array([-basis, basis, basis, -basis, -basis]) / 2.
rectangle_y = np.array([-height, -height, height, height, -height]) / 2.

n_plot_points = 1000
x_plot = np.linspace(-basis/2, basis/2, n_plot_points)
ellipse_set = np.zeros([4, 2, n_plot_points])

for n in range(4):
	order = 2 * (n+1)
	ellipse_set[n][0] = height/2 * np.power(1 - np.power(x_plot/(basis/2), order), 1./order)
	ellipse_set[n][1] = - height/2 * np.power(1 - np.power(x_plot/(basis/2), order), 1./order)

plt.figure(1)
plt.plot(rectangle_x, rectangle_y, 'r')
plt.plot(x_plot, ellipse_set[0][0], '--b')
plt.plot(x_plot, ellipse_set[0][1], '--b')
plt.title(r'$n = 1$', fontsize=14)

plt.figure(2)
plt.plot(rectangle_x, rectangle_y, 'r')
plt.plot(x_plot, ellipse_set[1][0], '--b')
plt.plot(x_plot, ellipse_set[1][1], '--b')
plt.title(r'$n = 2$', fontsize=14)

plt.figure(3)
plt.plot(rectangle_x, rectangle_y, 'r')
plt.plot(x_plot, ellipse_set[2][0], '--b')
plt.plot(x_plot, ellipse_set[2][1], '--b')
plt.title(r'$n = 3$', fontsize=14)

plt.figure(4)
plt.plot(rectangle_x, rectangle_y, 'r')
plt.plot(x_plot, ellipse_set[3][0], '--b')
plt.plot(x_plot, ellipse_set[3][1], '--b')
plt.title(r'$n = 4$', fontsize=14)

plt.show()