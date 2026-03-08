from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

X,y = load_iris(return_X_y=True)
model=LinearRegression()
selected_features=[]
features=list(range(X.shape[1]))

while len(features) > 1:
    scores=[]
    for feature in features:
        temp_features=features.copy()
        temp_features.remove(feature)
        score=cross_val_score(model,X[:,temp_features],y,cv=5, scoring='r2').mean()
        scores.append(score)
    worst_feature=features[np.argmin(scores)]
    features.remove(worst_feature)

    print("Remaining Features : ", features)
