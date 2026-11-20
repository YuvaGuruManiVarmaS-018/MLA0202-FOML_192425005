# Naive Bayes Algorithm with Confusion Matrix and Accuracy Graph

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train the Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# 4. Predict the test data
y_pred = model.predict(X_test)

# 5. Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("Naive Bayes Classification Results")
print("----------------------------------")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(cm)

# 6. Display graphs using Matplotlib
fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Confusion Matrix Graph
ax[0].imshow(cm, interpolation='nearest', cmap='Blues')
ax[0].set_title("Confusion Matrix")
ax[0].set_xlabel("Predicted Class")
ax[0].set_ylabel("Actual Class")

ax[0].set_xticks(np.arange(3))
ax[0].set_yticks(np.arange(3))
ax[0].set_xticklabels(iris.target_names)
ax[0].set_yticklabels(iris.target_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax[0].text(j, i, cm[i, j],
                   ha="center", va="center", color="black",
                   fontsize=12)

# Accuracy Graph
ax[1].bar(["Naive Bayes"], [accuracy * 100])
ax[1].set_title("Model Accuracy")
ax[1].set_ylabel("Accuracy (%)")
ax[1].set_ylim(0, 100)

ax[1].text(0, accuracy * 100 + 2,
           f"{accuracy * 100:.2f}%",
           ha="center", fontsize=12)

plt.tight_layout()
plt.show()
