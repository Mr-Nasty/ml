from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

X,y=load_iris(return_X_y=True)

model=LinearRegression()
selected_features=[]
remaining_features=list(range(X.shape[1]))

while remaining_features:
    scores=[]

    for feature in remaining_features:
        temp_features=selected_features+[feature]
        score=cross_val_score(
            model,
            X[:,temp_features],
            y,
            cv=5,scoring='r2'
        ).mean()
        scores.append(score)

    best_feature=remaining_features[np.argmax(scores)]
    selected_features.append(best_feature)
    remaining_features.remove(best_feature)

    print("Selected Features: ",selected_features)
