import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
col_names = ["outlook","temperature","humidity","windy","play_golf"]
df = pd.read_csv(r"C:\Users\devan\OneDrive\Desktop\ML\data - data.csv", header=None, names=col_names)

# Split features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode categorical data
encoder = OrdinalEncoder()
X = encoder.fit_transform(X)

label = LabelEncoder()
y = label.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# Train model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

plot_tree(
    model,
    feature_names=col_names[:-1],
    class_names=label.classes_,
    filled=True
)
plt.show()