import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score,confusion_matrix,classification_report)
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x=iris.data[:,:2]
y=iris.target
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print("accuracy",accuracy_score(y_test,y_pred))
print("confusion matrix",confusion_matrix(y_test,y_pred))
print("classification report",classification_report(y_test,y_pred))
h=0.02
x_min, x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min, y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('SVM Decision Boundary on Iris Dataset')
plt.show()
