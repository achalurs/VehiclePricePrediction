# vehicle_price_train.py
# Train ML model for Vehicle Price Prediction (scikit-learn >= 1.4 compatible)

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Load dataset
df = pd.read_csv("dataset.csv")

print("✅ Dataset Loaded:", df.shape)
print("Columns:", df.columns.tolist())

# 2. Drop rows with missing price
df = df.dropna(subset=["price"]).reset_index(drop=True)

# 3. Derived feature
df["age"] = df["year"].max() - df["year"]

# 4. Select relevant features
features = [
    "make", "model", "year", "age", "mileage", "cylinders",
    "fuel", "transmission", "body", "doors", "drivetrain"
]
features = [c for c in features if c in df.columns]

X = df[features]
y = df["price"].astype(float)

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# 6. Preprocessing
numeric_features = [c for c in ["year", "age", "mileage", "cylinders", "doors"] if c in X_train.columns]
categorical_features = [c for c in features if c not in numeric_features]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# ✅ FIXED for scikit-learn >= 1.4
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# 7. Model pipeline
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1))
])

# 8. Train
print("🚀 Training model...")
model.fit(X_train, y_train)

# 9. Predict & Evaluate
y_pred = model.predict(X_test)

rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae = float(mean_absolute_error(y_test, y_pred))
r2 = float(r2_score(y_test, y_pred))

print("\n📊 Model Evaluation:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R²  : {r2:.3f}")

# 10. Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/vehicle_price_model.joblib")
print("\n💾 Model saved to: model/vehicle_price_model.joblib")