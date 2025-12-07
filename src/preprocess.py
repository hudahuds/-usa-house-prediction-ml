import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os 

df = pd.read_csv("./data/raw/USA Housing Dataset.csv")
print(df.head(10))
print(df.shape)
print(df.info()) 
print(df.columns)
print(df.iloc[234])

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


final_columns = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "age_of_house",
    "has_been_renovated",
    "city",
    "statezip",
    "street"
]

# Sélectionner uniquement ces colonnes
df_processed = df[final_columns].copy()

# Sauvegarder dans data/processed/train.csv
df_processed.to_csv("./data/processed/train.csv", index=False)

print("Fichier train.csv créé dans data/processed !") # donc là on a crèer une nouvelle version CSV dans train.csv modele nettoyè et pret 



# Sélectionner les colonnes numériques
num_cols = [
    "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
    "floors", "waterfront", "view", "condition", "sqft_above",
    "sqft_basement", "yr_built", "yr_renovated"
]

# Calculer la corrélation
corr_matrix = df[num_cols].corr()

# Afficher la heatmap
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matrice de corrélation")
plt.show()



plt.figure(figsize=(15, 10))
sns.boxplot(data=df[['price', 'sqft_living', 'sqft_lot', 'bedrooms', 'bathrooms']])
plt.title("Visualisation des Outliers")
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(10, 6))
sns.histplot(df['price'], bins=50, kde=True)
plt.title("Distribution du prix")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['sqft_living'], y=df['price'])
plt.title("Prix vs Superficie (sqft_living)")
plt.show()

## comptage des outliers 

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1

# Limites pour définir les outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['price'] < lower_bound) | (df['price'] > upper_bound)]
print(f"Nombre d'outliers : {len(outliers)}")
## donc j'ai trouvè 215 outliers 

df_clean = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print(f"Données après suppression des outliers : {df_clean.shape}")

df_clean.to_csv("./data/processed/train_clean.csv", index=False)
print("Dataset nettoyé sauvegardé dans train_clean.csv")






# Charger le dataset nettoyé
df = pd.read_csv("./data/processed/train_clean.csv")  # ou train.csv nettoyé

# Fonction pour calculer les outliers selon l'IQR
def detect_outliers_IQR(col):
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = col[(col < lower_bound) | (col > upper_bound)]
    return outliers

# Exemple pour la colonne "price"
outliers_price = detect_outliers_IQR(df["price"])
print(f"Nombre d'outliers dans price : {len(outliers_price)}")

# Tu peux faire pareil pour d'autres colonnes numériques
numeric_cols = ["bedrooms","bathrooms","sqft_living","sqft_lot","floors","sqft_above","sqft_basement","age_of_house"]
for col in numeric_cols:
    outliers_col = detect_outliers_IQR(df[col])
    print(f"{col}: {len(outliers_col)} outliers")


# Supprimer les outliers de price
df_cleaned = df[~df["price"].isin(outliers_price)].copy()

# Vérification
print(f"Nombre de lignes après suppression des outliers price : {len(df_cleaned)}")

# Optionnel : sauvegarder le dataset propre
df_cleaned.to_csv("./data/processed/train_clean_no_outliers.csv", index=False)
print("Dataset nettoyé sauvegardé !")


# encodage 

df = pd.read_csv("./data/processed/train_clean_no_outliers.csv")
df.describe
print(list(df.columns))
df.head(50)
df = df.drop(columns=["yr_renovated_filled"])
df.to_csv("./data/processed/train_clean_no_outliers.csv", index=False)
df = df.drop(columns=["yr_built", "yr_renovated"])
df.to_csv("./data/processed/train_clean_no_outliers.csv", index=False)

print(df.columns)
df.info()

# Afficher les valeurs uniques
print("Waterfront:", df["waterfront"].unique())
print("View:", df["view"].unique())
print("Condition:", df["condition"].unique())

print(df["statezip"])
print(df["statezip"].unique())
