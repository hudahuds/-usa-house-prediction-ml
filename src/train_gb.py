# src/train_gb.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

# --- Chemins ---
processed_csv = "./data/processed/train_clean_encoded.csv"  # dataset nettoyé et encodé
model_dir = "./models"
model_path = os.path.join(model_dir, "gb_model.pkl")
scaler_path = os.path.join(model_dir, "gb_scaler.pkl")
output_csv = "./data/processed/train_clean_with_gb_predictions.csv"

# --- 1) Vérifier que le CSV prétraité existe ---
if not os.path.exists(processed_csv):
    raise FileNotFoundError(f"Fichier manquant : {processed_csv}. Exécute preprocess.py pour créer le dataset nettoyé et encodé.")

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
    "has_been_renovated",
    "city_encoded",    # ajout si tu as fait un target encoding
    "statezip_encoded" # ajout si tu as encodé statezip
]

# Vérification des colonnes
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

# --- 7) Standardisation (optionnelle pour Gradient Boosting) ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 8) Entraînement du modèle Gradient Boosting ---
model = GradientBoostingRegressor(
    n_estimators=300,      # nombre d'arbres
    learning_rate=0.01,    # taux d'apprentissage
    max_depth=5,           # profondeur max
    random_state=42
)
model.fit(X_train_scaled, y_train)

# --- 9) Prédiction et évaluation ---
y_test_pred = model.predict(X_test_scaled)
df["predicted_price"] = model.predict(scaler.transform(X))

mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)

print(f"MSE (test) : {mse:.2f}")
print(f"R² (test) : {r2:.2f}")
print(f"MAE (test) : {mae:.2f}")

# --- 10) Sauvegarder modèle + scaler ---
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
print(f"Modèle GB sauvegardé : {model_path}")
print(f"Scaler sauvegardé : {scaler_path}")

# --- 11) Sauvegarder dataset avec prédictions ---
df.to_csv(output_csv, index=False)
print(f"Dataset avec prédictions GB sauvegardé : {output_csv}")


import matplotlib.pyplot as plt

importances = model.feature_importances_
plt.barh(features, importances)
plt.title("Importance des features")
plt.show()



from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
import numpy as np

train_sizes, train_scores, test_scores = learning_curve(
    model, X_train_scaled, y_train, cv=5, scoring='r2', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Train R²')
plt.plot(train_sizes, test_mean, 'o-', color='green', label='Validation R²')
plt.xlabel("Nombre d'exemples d'entraînement")
plt.ylabel("R²")
plt.title("Courbe d'apprentissage")
plt.legend()
plt.show()


