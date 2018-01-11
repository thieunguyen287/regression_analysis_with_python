import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

X, y = make_regression(1000, n_features=1, n_targets=1, noise=20.4, bias=9.5)
model = LinearRegression()
model.fit(X, y)
print model.coef_
print model.intercept_
predictions = model.predict(X)
plt.plot(X)
plt.plot(y)
plt.plot(predictions)
plt.show()
