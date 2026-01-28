import pickle
from statistics import mean
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Simple Linear Regression from scratch
X = np.array([1, 2, 3, 4, 5], dtype=np.float64)

y = np.array([5, 4, 6, 5, 6], dtype=np.float64)

# splitting it into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of the linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train.reshape(-1, 1), y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test.reshape(-1, 1))

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Print the coefficients
print(f"Coefficient (slope): {model.coef_[0]}")
print(f"Intercept: {model.intercept_}")

# Save the model to a file
with open("linear_regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

# plotting actual data points
plt.scatter(X_test, y_test, color='blue', label='Actual')
# Plot the actual data points
plt.plot(X_test, y_pred, color='red', label='Predicted') 
# Add labels and title
plt.xlabel('X')
plt.ylabel('y')
# Add legend
plt.legend()
# Display the plot
plt.show()