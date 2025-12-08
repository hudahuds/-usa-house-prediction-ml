# src/train_rf.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.ensemble import RandomForestRegressor

# --- Chemins ---
processed_csv = "./data/processed/train_clean_encoded.csv"  # dataset nettoyé
model_dir = "./models"
model_path = os.path.join(model_dir, "rf_model.pkl")
scaler_path = os.path.join(model_dir, "scaler_rf.pkl")

# --- Charger les données prétraitées ---
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
    "has_been_renovated",
    "city_encoded",    # ajout si tu as fait un target encoding
    "statezip_encoded" # ajout si tu as encodé statezip
]

X = df[features]
y = df["price"]

# --- Split train/test ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Standardisation ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Créer et entraîner Random Forest ---
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,         # ajustable selon ton dataset
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train)

# --- Évaluation ---
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# --- Sauvegarder modèle et scaler ---
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Modèle Random Forest sauvegardé : {model_path}")
print(f"Scaler sauvegardé : {scaler_path}")
