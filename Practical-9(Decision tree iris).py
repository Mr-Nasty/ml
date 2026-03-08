import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.tree import export_text
from sklearn.datasets import load_iris

iris = load_iris()
col_names = iris.feature_names + ["target"]
df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
df["target"] = iris.target
data = df.values
x = data[:, :-1]
y = data[:, -1]

print("Input: ", x.shape)
print("Output: ", y.shape)

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Input: ", x.shape)
print(x[:5, :])
print("Output: ", y.shape)
print(y[:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Predictions:", y_pred)
print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
text_representation = export_text(model, feature_names=iris.feature_names)
print("Decision Tree Structure:")
print(text_representation)
