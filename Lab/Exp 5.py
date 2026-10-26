import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create and train K-NN model
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

print("Actual Values:", y_test)
print("Predicted Values:", y_pred)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Plot classification graph
plt.scatter(
    X_test[:, 0], X_test[:, 1],
    c=y_pred, cmap="viridis", edgecolors="black"
)

plt.xlabel(data.feature_names[0])
plt.ylabel(data.feature_names[1])
plt.title("K-NN Classification")
plt.colorbar(label="Predicted Class")
plt.show()
