
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([
    [1.0], [1.5], [2.0], [2.5], [3.0],
    [8.0], [8.5], [9.0], [9.5], [10.0]
])

# Create Gaussian Mixture Model
gmm = GaussianMixture(
    n_components=2,
    random_state=42
)

# Fit the model using EM algorithm
gmm.fit(X)

# Predict cluster labels
labels = gmm.predict(X)

# Display results
print("Cluster Labels:", labels)
print("Means:", gmm.means_.flatten())
print("Variances:", gmm.covariances_.flatten())
print("Mixing Weights:", gmm.weights_)

# Visualize clusters
plt.scatter(X.flatten(), np.zeros(len(X)),
            c=labels, cmap='viridis')
plt.title("Expectation-Maximization Algorithm")
plt.xlabel("Data Points")
plt.yticks([])
plt.show()
