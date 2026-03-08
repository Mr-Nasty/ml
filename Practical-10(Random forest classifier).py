from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score,recall_score, precision_score, accuracy_score, confusion_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()

x = data.data
y = data.target

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.3, random_state=42
)

forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(x_train, y_train)

y_pred = forest.predict(x_test)
print("Y Prediciton: ", y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)

precision = precision_score(y_test, y_pred, average='weighted')
print("Precision: ", precision)

f1 = f1_score(y_test, y_pred, average='weighted')
print("F1 Score: ", f1)

recall = recall_score(y_test, y_pred, average='weighted')
print("Recall score: ", recall)

cm = confusion_matrix(y_test, y_pred)
print("Confution Matrix: \n", cm)

classification = classification_report(y_test, y_pred, target_names = data.target_names)
print("CLassificiation: ", classification)