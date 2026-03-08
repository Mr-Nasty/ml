import pandas as pd
import numpy as np

data = {
    "age": [24,27,30,35,29,32,26,40,38,28],
    "salary": [35000,45000,60000,90000,55000,70000,40000,120000,110000,50000],
    "exp": [1,3,5,10,4,7,2,12,11,3],
    "dept": ["Software Dev","Data Analyst","Network Eng","Cloud Eng","QA Tester","DevOps Eng","Support Eng","Data Scientist","AI Engineer","Web Dev"],
    "city": ["Mumbai","Pune","Nagpur","Mumbai","Nashik","Pune","Aurangabad","Mumbai","Pune","Thane"]
}

df = pd.DataFrame(data)

df = pd.get_dummies(df, drop_first=True)

corr = df.corr().abs()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]

df = df.drop(columns=to_drop)

print("Dropped columns:", to_drop)
print("\nReduced Dataset:\n")
print(df)