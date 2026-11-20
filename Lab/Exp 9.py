import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Sample dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 5, 10, 17, 26, 37])

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X, y)
y_linear = linear_model.predict(X)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

poly_model = LinearRegression()
poly_model.fit(X_poly, y)
y_poly = poly_model.predict(X_poly)

# Plot the results
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X, y_linear, color='red', label='Linear Regression')
plt.plot(X, y_poly, color='green', label='Polynomial Regression')

plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear vs Polynomial Regression')
plt.legend()
plt.show()

# Print results
print("Linear Regression Predictions:", y_linear)
print("Polynomial Regression Predictions:", y_poly)
