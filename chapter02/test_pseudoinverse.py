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

X = dataset['RM'].values.reshape(-1, 1)
Xb = np.column_stack((X, np.ones_like(X)))
y = dataset['target'].values


def matrix_inverse(X, y, pseudo=False):
    if pseudo:
        return np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
    else:
        return np.dot(np.linalg.inv(np.dot(X.T, X)), np.dot(X.T, y))


def normal_equation(X, y):
    return np.linalg.solve(np.dot(X.T, X), np.dot(X.T, y))

print matrix_inverse(Xb, y)
print matrix_inverse(Xb, y, True)
print normal_equation(Xb, y)


