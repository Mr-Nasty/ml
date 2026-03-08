import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle

data = pd.read_csv("student.csv")

X = data[['Hours']]
y = data['Result']

model = LogisticRegression()
model.fit(X, y)
pickle.dump(model, open("model.pkl", "wb"))
print("Model saved successfully!")
