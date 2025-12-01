
import pandas as pd
df = pd.read_csv("./data/raw/USA Housing Dataset.csv")
print(df.head())
print(df.shape)
print(df.info()) 
print(df.columns)


df["date"] = pd.to_datetime(df["date"])

df["year_sold"] = df["date"].dt.year
df["month_sold"] = df["date"].dt.month
df["day_sold"] = df["date"].dt.day

#df.drop(columns=["date"], inplace=True)
