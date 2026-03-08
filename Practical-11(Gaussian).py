import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, precision_score,
                             recall_score, f1_score, ConfusionMatrixDisplay)

data = {
    'color':['red','red','red','yellow','yellow','yellow','yellow','yellow','red','red'],
    'origin':['sports','sports','sports','sports','sports','suv','suv','suv','suv','sports'],
    'type':['Domestic','Domestic','Domestic','Domestic','Imported','Imported','Imported','Domestic','Imported','Imported'],
    'stolen':['yes','no','yes','no','yes','no','yes','no','no','yes']
}

df = pd.DataFrame(data)

# Encode categorical values
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])

x = df.drop('stolen', axis=1)
y = df['stolen']

# Model
model = GaussianNB()
model.fit(x, y)

y_pred = model.predict(x)   

# Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred, average='weighted')
recall = recall_score(y, y_pred, average='weighted')
f1 = f1_score(y, y_pred, average='weighted')

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Classification Report
print("\nClassification Report:")
print(classification_report(y, y_pred))

# Confusion Matrix
cm = confusion_matrix(y, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot Confusion Matrix using ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(plt.cm.Blues)
plt.show()

plt.title("Confusion Matrix")
plt.show()

# Simple bar chart for metrics
metrics = ['Accuracy','Precision','Recall','F1']
values = [accuracy, precision, recall, f1]

plt.bar(metrics, values)
plt.title("Model Performance")
plt.ylabel("Score")
plt.show()