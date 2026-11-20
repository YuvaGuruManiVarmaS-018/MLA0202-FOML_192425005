
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Create sales dataset
data = {
    'Month': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales': [10000, 12000, 15000, 18000, 20000,
              22000, 25000, 28000, 30000, 35000]
}

df = pd.DataFrame(data)

# Split input and output
X = df[['Month']]
y = df['Sales']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Display performance
print("Future Sales Prediction")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict future sales
future_month = pd.DataFrame({'Month': [11]})
prediction = model.predict(future_month)

print("Predicted Sales for Month 11: Rs.",
      round(prediction[0], 2))

