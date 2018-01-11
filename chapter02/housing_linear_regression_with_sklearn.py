import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
from sklearn import linear_model


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

linear_regression = linear_model.LinearRegression(fit_intercept=True)
X = dataset['RM'].values.reshape((-1, 1))
y = dataset['target'].values
linear_regression.fit(X, y)
print linear_regression.coef_
print linear_regression.intercept_
print linear_regression.predict(X).shape
print (np.dot(X, linear_regression.coef_) + linear_regression.intercept_)[:10]
