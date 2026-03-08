import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt

data = {
    'Age':[25,30,np.nan,40,1000,35,29],
    'Salary':[50000,60000,55000,np.nan,80000,10000000,45000],
    'Department':['IT','HR','IT','Finance', None,'HR','IT']
}

df = pd.DataFrame(data)
print(df)

print("Number of NAN", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Salary'] = df['Salary'].fillna(df['Salary'].median())
df['Department'] = df['Department'].fillna(df['Department'].mode()[0])

def outliner(column):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return column.clip(lower_bound, upper_bound)

df['Age'] = outliner(df['Age'])
df['Salary'] = outliner(df['Salary'])

scaler = StandardScaler()
df[['Age_scaled', 'Salary_scaled']] = scaler.fit_transform(df[['Age', 'Salary']])

minmax = MinMaxScaler()
df[['Age_MinMax', 'Salary_MinMax']] = minmax.fit_transform(df[['Age', 'Salary']])

print(df)