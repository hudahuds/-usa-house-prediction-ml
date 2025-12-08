# src/train_clean.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# --- Chemins ---
processed_csv = "./data/processed/train_clean_no_outliers.csv"
model_dir = "./models"
model_path = os.path.join(model_dir, "linear_model_clean.pkl")
scaler_path = os.path.join(model_dir, "scaler_clean.pkl")

# --- Charger le CSV nettoyé ---
df = pd.read_csv(processed_csv)

# --- Colonnes/features ---
features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "age_of_house", "has_been_renovated"
]

X = df[features]
y = df["price"]

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Standardisation ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Entraînement ---
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# --- Prédiction et évaluation ---
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")
print(f"MAE: {mae:.2f}")

# --- Sauvegarder modèle + scaler ---
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Modèle sauvegardé : {model_path}")
print(f"Scaler sauvegardé : {scaler_path}")

# --- Sauvegarder les prédictions sur tout le dataset ---
df["predicted_price"] = model.predict(scaler.transform(X))
df.to_csv("./data/processed/train_clean_predictions.csv", index=False)
print("Prédictions sauvegardées dans train_clean_predictions.csv")


# src/train_gb_xgb_eval.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor

try:
    import xgboost as xgb
except ImportError:
    xgb = None
    print("XGBoost n'est pas installé. Installe-le avec `pip install xgboost`.")

# --- Charger le dataset nettoyé ---
df = pd.read_csv("./data/processed/train_clean_no_outliers.csv")

# --- Colonnes/features ---
features = [
    "bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
    "waterfront", "view", "condition", "sqft_above", "sqft_basement",
    "age_of_house", "has_been_renovated"
]
X = df[features]
y = df["price"]

# --- Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Standardisation ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Gradient Boosting Regressor ---
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)

mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)
mae_gb = mean_absolute_error(y_test, y_pred_gb)

print("=== Gradient Boosting ===")
print(f"MSE: {mse_gb:.2f}, R²: {r2_gb:.2f}, MAE: {mae_gb:.2f}")

# --- Créer et entraîner XGBoost ---
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.01,
    max_depth=5,
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --- Prédiction ---
y_pred_xgb = model.predict(X_test_scaled)

# --- Évaluation ---
mse_xgb = mean_squared_error(y_test, y_pred_xgb)
r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print("=== XGBoost ===")
print(f"MSE: {mse_xgb:.2f}, R²: {r2_xgb:.2f}, MAE: {mae_xgb:.2f}")
