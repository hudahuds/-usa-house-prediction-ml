import os
import pandas as pd
import joblib

# --- Chemins ---
input_csv = "./data/processed/train.csv"              # <= tu prédis sur train.csv
output_csv = "./data/processed/train_with_predictions.csv"
model_path = "./models/linear_model.pkl"
scaler_path = "./models/scaler.pkl"

# --- 1) Vérifier que train.csv existe ---
if not os.path.exists(input_csv):
    raise FileNotFoundError(f"Fichier introuvable : {input_csv}. Exécute preprocess.py d’abord.")

# --- 2) Charger les données ---
df = pd.read_csv(input_csv)

# --- 3) Colonnes utilisées pour l'entraînement ---
features = [
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
    "age_of_house"
]

# Vérifier les colonnes
missing = [c for c in features if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans le CSV : {missing}")

# --- 4) Charger modèle + scaler ---
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# --- 5) Préparer X ---
X = df[features]
X_scaled = scaler.transform(X)

# --- 6) Prédictions ---
df["predicted_price"] = model.predict(X_scaled)

# --- 7) Sauvegarde ---
df.to_csv(output_csv, index=False)
print(f"Prédictions sauvegardées dans : {output_csv}")

# --- 8) Aperçu ---
print(df[["bedrooms", "bathrooms", "sqft_living", "predicted_price"]].head())
