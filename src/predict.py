import pandas as pd
import joblib
import os

# Charger modèle et scaler
model = joblib.load("./models/linear_model.pkl")
scaler = joblib.load("./models/scaler.pkl")

# Charger les données
df_new = pd.read_csv("./data/processed/test.csv")  

# Colonnes à utiliser
features = [
    "bedrooms","bathrooms","sqft_living","sqft_lot","floors",
    "waterfront","view","condition","sqft_above","sqft_basement","age_of_house"
]

X_new = df_new[features]

# Standardiser
X_new_scaled = scaler.transform(X_new)

# Prédictions
df_new["predicted_price"] = model.predict(X_new_scaled)

# Sauvegarder
output_csv = "../data/processed/test.csv"
df_new.to_csv(output_csv, index=False)
print(f"Prédictions sauvegardées dans {output_csv}")

# Vérification
df_result = pd.read_csv(output_csv)
print(df_result.head())
