import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
import matplotlib.mlab as mlab
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf


def covariance(variable_1, variable_2, bias=0):
    observations = float(len(variable_1))
    return np.sum((variable_1 - np.mean(variable_1)) * (variable_2 - np.mean(variable_2))) / (observations + min(bias, 1))


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


def correlation(variable_1, variable_2, bias=0):
    return covariance(standardize(variable_1), standardize(variable_2), bias)


boston = load_boston()
california = fetch_california_housing()
dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
dataset['target'] = boston.target
mean = np.mean(dataset['target'])

#
# squared_errors = pd.Series(mean - dataset['target']) ** 2
# print squared_errors
# mse = np.mean(squared_errors)
# print "MSE: %.4f" % mse
# squared_errors.plot('hist')
# plt.show()

# print 'Our correlation estimation: %f' % correlation(dataset['RM'], dataset['target'])
# print 'Correlation from Scipy pearsonr: %f' % pearsonr(dataset['RM'], dataset['target'])[0]
# x_range = [dataset['RM'].min(), dataset['RM'].max()]
# y_range = [dataset['target'].min(), dataset['target'].max()]
# scatter_plot = dataset.plot(kind='scatter', x='RM', y='target', xlim=x_range, ylim=y_range)
# mean_y = scatter_plot.plot(x_range, [dataset['target'].mean(), dataset['target'].mean()], '--',
#                            color='red', linewidth=1)
# mean_x = scatter_plot.plot([dataset['RM'].mean(), dataset['RM'].mean()], y_range, '--',
#                            color='red', linewidth=1)
# plt.show()


y = dataset['target']
X = dataset['RM']
X = sm.add_constant(X)
print X.head()
# linear_regression = sm.OLS(y, X)
linear_regression = smf.ols(formula='target ~ RM', data=dataset)
fitted_model = linear_regression.fit()
summary = fitted_model.summary()
print summary

