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
df["yr_renovated_filled"] = df["yr_renovated"].fillna(0)
df["has_been_renovated"] = df["yr_renovated"].notna().astype(int)
print(df[["yr_renovated", "has_been_renovated"]].head(10))


print(df.head())
df["yr_built"] = pd.to_datetime(df["yr_built"], errors='coerce')  
df["age_of_house"] = 2025 - df["yr_built"].dt.year
print(df[["yr_built", "age_of_house"]].head())
print(df[["has_been_renovated", "age_of_house"]].head())

print(df.describe())

# Conversion si nécessaire
df["price"] = pd.to_numeric(df["price"], errors="coerce")
df["age_of_house"] = pd.to_numeric(df["age_of_house"], errors="coerce")

plt.figure(figsize=(12,6))
sns.lineplot(
    data=df.groupby("age_of_house")["price"].mean().reset_index(),
    x="age_of_house",
    y="price"
)
plt.title("Prix moyen des maisons selon l'âge")
plt.xlabel("Âge de la maison (années)")
plt.ylabel("Prix moyen ($)")
plt.show()

plt.figure(figsize=(12,6))
sns.scatterplot(
    data=df,
    x="sqft_living",
    y="price",
    alpha=0.6
)
plt.title("Prix des maisons selon la surface habitable")
plt.xlabel("Surface habitable (sqft)")
plt.ylabel("Prix ($)")
plt.show()


# Filtrer les maisons à prix élevé
high_price = df[df["price"] > 1_000_000]

# Compter par ville
high_price_count = high_price["city"].value_counts()

print("Nombre de maisons à prix élevé par ville :")
print(high_price_count)

# Graphe
plt.figure(figsize=(10,5))
sns.barplot(x=high_price_count.index, y=high_price_count.values)
plt.title("Nombre de maisons à prix élevé par ville")
plt.xlabel("Ville")
plt.ylabel("Nombre de maisons > 1M $")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(12,6))

sns.scatterplot(
    data=df,
    x="sqft_living",
    y="price",
    hue="city",       # couleur selon ville
    size="age_of_house",  # taille selon âge
    alpha=0.6
)

plt.title("Prix des maisons vs Surface — Couleurs selon la ville")
plt.xlabel("Surface habitable (sqft)")
plt.ylabel("Prix ($)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
#Graphe : Prix vs Surface pour Seattle
import matplotlib.pyplot as plt

# Filtrer uniquement Seattle
df_seattle = df[df["city"].str.contains("seattle", case=False, na=False)].copy()

# Vérifier qu'il y a bien des valeurs
print(df_seattle[["age_of_house", "price"]].head())

# Graphe
plt.figure(figsize=(8,6))
plt.scatter(df_seattle["age_of_house"], df_seattle["price"], s=10)
plt.xlabel("Âge de la maison")
plt.ylabel("Prix")
plt.title("Seattle : Prix en fonction de l'âge de la maison")
plt.grid(True)
plt.show()


