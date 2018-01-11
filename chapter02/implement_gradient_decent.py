import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.datasets import fetch_california_housing
from sklearn.datasets import load_boston
from sklearn import linear_model
import random


def covariance(variable_1, variable_2, bias=0):
    observations = float(len(variable_1))
    return np.sum((variable_1 - np.mean(variable_1)) * (variable_2 - np.mean(variable_2))) / (observations + min(bias, 1))


def standardize(x):
    return (x - np.mean(x)) / np.std(x)


def correlation(variable_1, variable_2, bias=0):
    return covariance(standardize(variable_1), standardize(variable_2), bias)


def hypothesis(X, w):
    return np.dot(X, w)


def loss(X, w, y):
    return hypothesis(X, w) - y


def gradient(X, w, y):
    gradients = [np.mean(loss(X, w, y) * X[:, j]) for j in range(len(w))]
    return gradients
boston = load_boston()
california = fetch_california_housing()
dataset = pd.DataFrame(boston.data, columns=boston.feature_names)
dataset['target'] = boston.target
mean = np.mean(dataset['target'])

X = dataset['RM'].values.reshape((-1, 1))
y = dataset['target'].values

X = np.column_stack((X, np.ones_like()))
