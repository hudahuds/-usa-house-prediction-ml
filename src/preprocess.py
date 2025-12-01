import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_csv("./data/raw/USA Housing Dataset.csv")
print(df.head())
print(df.shape)
print(df.info()) 
print(df.columns)
print(df.iloc[1000])

df.drop(columns=["date"], inplace=True) #on a supprimè la colonne date elle etait toutes 00000
print(df.columns)


df["yr_built"] = pd.to_datetime(df["yr_built"], format="%Y")
df["yr_renovated"] = pd.to_datetime(df["yr_renovated"], format="%Y")

# Vérification
print(df[["yr_built", "yr_renovated"]].head())
print(df.dtypes)


# Remplacer les 0 par NaT
df["yr_renovated"] = df["yr_renovated"].replace(0, pd.NaT)

# Convertir en datetime
df["yr_built"] = pd.to_datetime(df["yr_built"], format="%Y")
df["yr_renovated"] = pd.to_datetime(df["yr_renovated"], format="%Y")

# Vérification
print(df[["yr_built", "yr_renovated"]].head())
print(df.dtypes)

print(df.isnull().sum())
duplicates = df[df.duplicated()]
print(duplicates)
print("Nombre de doublons :", len(duplicates))
print(df["country"].nunique())
print((df["country"] == "USA").all())
df.drop(columns=["country"], inplace=True)
print(df.describe())

# Age de la maison
df["age_of_house"] = 2025 - df["yr_built"].dt.year  # si yr_built est en datetime

# Maison rénovée ou pas
df["has_been_renovated"] = df["yr_renovated"].apply(lambda x: 1 if x > 0 else 0)
print(df.head())
df["yr_built"] = pd.to_datetime(df["yr_built"], errors='coerce')  
df["age_of_house"] = 2025 - df["yr_built"].dt.year