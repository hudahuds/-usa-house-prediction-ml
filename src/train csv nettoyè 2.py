# src/train_clean.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# --- Chemins ---
processed_csv = "./data/processed/train_clean.csv"  # dataset sans outliers
model_dir = "./models"
model_path = os.path.join(model_dir, "linear_model_clean.pkl")
scaler_path = os.path.join(model_dir, "scaler_clean.pkl")
output_csv = "./data/processed/train_clean_with_predictions.csv"

# --- 1) Vérifier que le CSV prétraité existe ---
if not os.path.exists(processed_csv):
    raise FileNotFoundError(f"Fichier manquant : {processed_csv}. Exécute preprocess.py pour créer train_clean.csv")

# --- 2) Charger le CSV prétraité ---
df = pd.read_csv(processed_csv)

# --- 3) Colonnes/features attendues ---
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
    "age_of_house",
    "has_been_renovated"  # inclure si présente
]

missing = [c for c in features + ["price"] if c not in df.columns]
if missing:
    raise ValueError(f"Colonnes manquantes dans {processed_csv} : {missing}")

# --- 4) Gérer rapidement les NaN numériques ---
num_cols = features + ["price"]
for c in num_cols:
    if df[c].isna().any():
        med = df[c].median()
        df[c] = df[c].fillna(med)

# --- 5) Préparer X et y ---
X = df[features]
y = df["price"]

# --- 6) Train / Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 7) Standardisation ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 8) Entraînement du modèle ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- 9) Prédiction sur le test et tout le dataset ---
y_test_pred = model.predict(X_test_scaled)
df["predicted_price"] = model.predict(scaler.transform(X))

# --- 10) Évaluation ---
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

print(f"MSE (test) : {mse:.2f}")
print(f"R² (test) : {r2:.2f}")
print(f"MAE (test) : {mae:.2f}")

# --- 11) Sauvegarder modèle + scaler ---
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Modèle sauvegardé : {model_path}")
print(f"Scaler sauvegardé : {scaler_path}")

# --- 12) Sauvegarder dataset avec prédictions ---
df.to_csv(output_csv, index=False)
print(f"Dataset avec prédictions sauvegardé : {output_csv}")
