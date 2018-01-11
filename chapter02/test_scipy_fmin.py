import numpy as np
from scipy.optimize import fmin


def absolute_error(v, e):
    return np.mean(np.abs(v - e))


def squared_error(v, e):
    return np.sum(np.square(v - e))

x = np.array([9.5, 8.5, 8.0, 7.0, 6.0])
xopt = fmin(absolute_error, x0=0, xtol=1e-8, args=(x,))

print xopt
print 'Optimization result: %f' % xopt[0]
print 'Mean: %f' % np.mean(x)
print 'Median: %f' % np.median(x)
