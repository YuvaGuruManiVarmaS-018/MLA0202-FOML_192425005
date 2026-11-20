
# Import required libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Create dataset
data = {
    'Income': [30000, 40000, 50000, 60000, 25000,
               70000, 80000, 35000, 90000, 45000],
    'Credit_Score': [600, 650, 700, 750, 550,
                     800, 850, 620, 900, 680],
    'Loan_Amount': [100000, 150000, 200000, 250000,
                    80000, 300000, 350000, 120000,
                    400000, 180000],
    'Approved': [0, 0, 1, 1, 0, 1, 1, 0, 1, 1]
}

df = pd.DataFrame(data)

# Split input and output
X = df[['Income', 'Credit_Score', 'Loan_Amount']]
y = df['Approved']

# Split training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Naive Bayes model
model = GaussianNB()

# Train the model
model.fit(X_train, y_train)

# Predict loan approval
y_pred = model.predict(X_test)

# Calculate accuracy
print("Bank Loan Prediction using Naive Bayes")
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

# Predict a new applicant
new_applicant = pd.DataFrame({
    'Income': [60000],
    'Credit_Score': [750],
    'Loan_Amount': [200000]
})

prediction = model.predict(new_applicant)

if prediction[0] == 1:
    print("Loan Status: Approved")
else:
    print("Loan Status: Rejected")
