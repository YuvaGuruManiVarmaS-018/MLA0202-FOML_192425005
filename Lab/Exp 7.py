# Logistic Regression with Matplotlib Graph

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load dataset
data = load_breast_cancer()
X = data.data[:, :2]
y = data.target

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 4. Predict the results
y_pred = model.predict(X_test)

# 5. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Logistic Regression Results")
print("---------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 6. Plot graphs
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Graph 1: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)
display.plot(ax=ax[0], cmap="Blues", colorbar=False)
ax[0].set_title("Confusion Matrix")

# Graph 2: Actual vs Predicted
ax[1].scatter(
    range(len(y_test)), y_test,
    label="Actual", marker="o"
)
ax[1].scatter(
    range(len(y_pred)), y_pred,
    label="Predicted", marker="x"
)
ax[1].set_title("Actual vs Predicted Results")
ax[1].set_xlabel("Test Sample")
ax[1].set_ylabel("Class (0 or 1)")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

# 7. Display accuracy graph
plt.figure(figsize=(5, 4))
plt.bar(["Logistic Regression"], [accuracy * 100])
plt.ylabel("Accuracy (%)")
plt.title("Logistic Regression Accuracy")
plt.ylim(0, 100)
plt.text(
    0, accuracy * 100 + 1,
    f"{accuracy * 100:.2f}%",
    ha="center"
)
plt.tight_layout()
plt.show()
