import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # Pour sauvegarder le modèle

# 1. Charger les données brutes
df = pd.read_csv("./data/raw/USA Housing Dataset.csv")

df["yr_built"] = pd.to_datetime(df["yr_built"], errors="coerce").dt.year

# Créer l'âge de la maison
df["age_of_house"] = 2025 - df["yr_built"]  # ou l'année actuelle
# 2. Colonnes prédictives
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
X = df[features]
y = df["price"]

# 4️ Diviser en train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️ Standardiser les données
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6️ Créer et entraîner le modèle
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 7️ Faire des prédictions
y_pred = model.predict(X_test_scaled)

# 8️Évaluer le modèle
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R²: {r2:.2f}")

# 9️ Sauvegarder le modèle et le scaler
joblib.dump(model, "../models/linear_model.pkl")
joblib.dump(scaler, "../models/scaler.pkl")
print("Modèle et scaler sauvegardés !")