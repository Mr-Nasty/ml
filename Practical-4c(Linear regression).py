from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

X,y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model=LinearRegression()
model.fit(X_train, y_train)

print(model.coef_)
print(model.intercept_)

y_pred=model.predict(X_test)
print(y_test, y_pred)
accuracy=r2_score(y_test, y_pred)
print("R2 Score (Accuracy) : ", accuracy)
