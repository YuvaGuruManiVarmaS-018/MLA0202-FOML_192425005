
# Import required libraries
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create Perceptron model
model = Perceptron(max_iter=1000, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Perceptron-Based Iris Classification")
print("Accuracy:", accuracy * 100, "%")

# Classification report
print(classification_report(
    y_test, y_pred, target_names=iris.target_names
))

# Predict a new flower
sample = [[5.1, 3.5, 1.4, 0.2]]
sample = scaler.transform(sample)

prediction = model.predict(sample)

print("Predicted Flower:", iris.target_names[prediction[0]])
