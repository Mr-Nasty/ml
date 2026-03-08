import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

np.random.seed(1)

data = {
    "Customer_Age": np.random.randint(20,75,200),
    "Annual_Income": np.random.randint(30000,90000,200),
    "Gender": np.random.choice(["Male","Female"],200),
    "City": np.random.choice(["Mumbai","Delhi","Pune","Bangalore","Chennai"],200),
    "Purchase_Amount": np.random.randint(800,7000,200),
    "Visit_Frequency": np.random.randint(1,20,200)
}

df = pd.DataFrame(data)

for col in ['Annual_Income','Purchase_Amount']:
    df.loc[df.sample(frac=0.12).index,col] = np.nan

df.loc[df.sample(frac=0.04).index,'Annual_Income'] = df['Annual_Income'] * 6

print(df.isnull().sum())

imputer = SimpleImputer(strategy='median')
df['Annual_Income'] = imputer.fit_transform(df[['Annual_Income']])
df['Purchase_Amount'] = imputer.fit_transform(df[['Purchase_Amount']])

Q1 = df["Annual_Income"].quantile(0.25)
Q3 = df["Annual_Income"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df["Annual_Income"] = df["Annual_Income"].clip(lower,upper)

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
df["City"] = le.fit_transform(df["City"])

plt.hist(df["Purchase_Amount"],bins=12)
plt.title("Purchase Amount Histogram")
plt.xlabel("Purchase Amount")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x=df["Annual_Income"])
plt.title("Annual Income Boxplot")
plt.show()

sns.boxplot(x=df["Purchase_Amount"])
plt.title("Purchase Amount Boxplot")
plt.show()

sns.heatmap(df.corr(),annot=True,cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

gender_mean = df.groupby("Gender")["Purchase_Amount"].mean()
print(gender_mean)

city_mean = df.groupby("City")["Purchase_Amount"].mean()
print(city_mean)