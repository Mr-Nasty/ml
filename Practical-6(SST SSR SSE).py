import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Independent variables
x = pd.DataFrame({
    'x1': [52, 23, 33, 44, 52],
    'x2': [121, 143, 135, 174, 205]
})

# Dependent variable
y = pd.DataFrame({
    'y': [103, 126, 119, 121, 125]
})

# Create and train model
model1 = LinearRegression()
model1.fit(x, y)

# Model parameters
print("Coefficients:", model1.coef_)
print("Intercept:", model1.intercept_)

# Predictions
y_pred = model1.predict(x)
print("Predicted y:\n", y_pred)

# R² score
r2 = r2_score(y, y_pred)
print("R2 Score (Accuracy):", r2)

# Sum of Squares
y_mean = np.mean(y.values)

SST = np.sum((y.values - y_mean) ** 2)
SSR = r2 * SST
SSE = SST - SSR

print("SST:", SST)
print("SSR:", SSR)
print("SSE:", SSE)
