
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'RAM': [2, 3, 4, 6, 8, 12, 4, 6, 8, 12],
    'Battery': [2000, 2500, 3000, 3500, 4000, 5000, 2800, 3800, 4500, 5500],
    'Memory': [16, 32, 32, 64, 128, 256, 64, 128, 256, 512],
    'Price_Range': [0, 0, 1, 1, 2, 3, 1, 2, 3, 3]
}

df = pd.DataFrame(data)

# Split input and output
X = df[['RAM', 'Battery', 'Memory']]
y = df['Price_Range']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Predict mobile price range
y_pred = model.predict(X_test)

# Display accuracy
print("Mobile Price Prediction")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Predict a new mobile
new_mobile = pd.DataFrame({
    'RAM': [8],
    'Battery': [4500],
    'Memory': [128]
})

prediction = model.predict(new_mobile)

print("Predicted Price Range:", prediction[0])
