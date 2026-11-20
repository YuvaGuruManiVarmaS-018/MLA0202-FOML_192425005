import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 5, 7])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

print("Linear Regression")
print("Predicted values:", np.round(y_pred, 2))
print("R2 Score:", round(model.score(X, y), 2))

# Plot graph
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, y_pred, color="red", label="Regression Line")
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression")
plt.legend()
plt.grid()
plt.show()
