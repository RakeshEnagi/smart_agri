import pandas as pd
df = pd.read_csv('data/potato_disease.csv')
print(df["Disease"].unique())