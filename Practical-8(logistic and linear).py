from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np 
import matplotlib.pyplot as plt

x=np.array([[1],[2],[3],[4],[5]])
y = np.array([0,0,0,1,1])

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)

model = LogisticRegression()
model.fit(x_train,y_train)

model1=LinearRegression()
model1.fit(x_train, y_train)

print("coef", model.coef_)
print("intercept",model.intercept_)

y_pred = model.predict(x_test)
print("x_test", x_test)
print("y_test,",y_test)
print("y_pred",y_pred)

accuracy=accuracy_score(y_test,y_pred)
print("accuracy",accuracy)

x_curve=np.linspace(1,5,100).reshape(-1,1)
y_curve=model.predict_proba(x_curve.reshape(-1,1))[:,1]
y_curve1=model1.predict(x_curve)

plt.scatter(x,y)
plt.plot(x_curve,y_curve,color = 'red')
plt.scatter(x,y)
plt.plot(x_curve,y_curve1, color='blue')
plt.show()
