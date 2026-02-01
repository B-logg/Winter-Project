import pandas as pd

df = pd.read_csv("NEON_dataset.csv", nrows=0)
print(df.columns)