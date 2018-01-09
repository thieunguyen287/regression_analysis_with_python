import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
import matplotlib.mlab as mlab
import math


# boston = load_boston()
# california = fetch_california_housing()
# dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
# dataset['target'] = boston.target

x = np.linspace(-4, 4, 100)
for mean, variance in [(0, 0.7), (0, 1), (1, 1.5), (-2, 0.5)]:
    plt.plot(x, mlab.normpdf(x, mean, variance))
plt.show()
