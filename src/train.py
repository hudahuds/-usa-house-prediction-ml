# src/train.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
from sklearn.metrics import mean_absolute_error

# --- Chemins ---
processed_csv = "./data/processed/train.csv"
model_dir = "./models"
model_path = os.path.join(model_dir, "linear_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

# --- 1) Vérifier que le CSV prétraité existe ---
if not os.path.exists(processed_csv):
    raise FileNotFoundError(f"Fichier manquant : {processed_csv}. Exécute preprocess.py d'abord.")

# --- 2) Charger le CSV prétraité ---
df = pd.read_csv(processed_csv)

# --- 3) Colonnes/features attendues ---
features = [
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",git
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "age_of_house",
     "has_been_renovated" ]

missing = [c for c in features + ["price"] if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans {processed_csv} : {missing}")

# --- 4) Gérer rapidement les NaN numériques (ne supprime pas les lignes du df original) ---
# Remplacement par la médiane est souvent une option neutre pour démarrer
num_cols = features + ["price"]
for c in num_cols:
    if df[c].isna().any():git
        med = df[c].median()
        df[c] = df[c].fillna(med)

# --- 5) Préparer X et y ---
X = df[features]
y = df["price"]

# --- 6) Train / Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0,2, random_state=42) #TEST_size c'est le pourcentage de donnèes utilisè 20%

# --- 7) Standardisation ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 8) Entraînement du modèle ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- 9) Prédiction et évaluation ---
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
mae = mean_absolute_error(y_test, y_pred)
print(f"MAE: {mae:.2f}")




# --- 10) Sauvegarder modèle + scaler (crée le dossier si nécessaire) ---
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Modèle sauvegardé : {model_path}")
print(f"Scaler sauvegardé : {scaler_path}")


#resumè du 1er modele  model = LinearRegression()  
