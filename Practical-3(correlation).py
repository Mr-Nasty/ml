import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

data = load_iris()

df = pd.DataFrame(data.data, columns = data.feature_names)
df['Target'] = data.target

print(data.keys())
print(df.shape)
print("Feature Names: ", data.feature_names)
print("Target Names: ", data.target_names)

corr = df.corr()
print("\nCorrelation Matrix\n", corr)

strong_corr = corr.abs()
strong_corr = strong_corr[strong_corr>0.8]
print("\nStrong Correlation Matrix: \n",strong_corr)

sns.heatmap(corr,annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

sns.heatmap(strong_corr, annot=True, cmap="coolwarm")
plt.title("Strong correlation matrix")
plt.show()

upper = corr.where(np.triu(np.ones(corr.shape),k=1).astype(bool))
to_drop = [col for col in upper.columns if any (upper[col]>0.8)]

df_selected = df.drop(columns = to_drop)

print(upper)
print(df_selected)


