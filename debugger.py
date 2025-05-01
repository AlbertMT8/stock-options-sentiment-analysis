import pandas as pd
df = pd.read_csv("data/Twitter_Data.csv")
print(df["category"].value_counts())