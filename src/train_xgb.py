# src/train_xgb.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBRegressor

# --- Chemins ---
processed_csv = "./data/processed/train_clean_encoded.csv"  # dataset nettoyé
model_dir = "./models"
model_path = os.path.join(model_dir, "xgb_model.pkl")
scaler_path = os.path.join(model_dir, "scaler_xgb.pkl")

# --- Charger les données prétraitées ---
df = pd.read_csv(processed_csv)

# --- Colonnes/features ---
features = [
    "bedrooms","bathrooms","sqft_living","sqft_lot","floors",
    "waterfront","view","condition","sqft_above","sqft_basement",
    "age_of_house","has_been_renovated"
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

# --- Créer et entraîner XGBoost ---
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=5,
    random_state=42
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
print(f"Modèle XGBoost sauvegardé : {model_path}")
print(f"Scaler sauvegardé : {scaler_path}")





#Best params: {'learning_rate': 0.01, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.7}