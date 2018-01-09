import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

x = np.linspace(-4, 4, 100)
print x
for mean, variance in [(0, 0.7), (0, 1), (1, 1.5), (-2, 0.5)]:
    plt.plot(x, mlab.normpdf(x, mean, variance))
plt.show()
