
import pandas as pd
import joblib

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sales.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "sales_model.joblib"

# 1. Load data
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "sales.csv not found. Put it inside the data folder."
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip().str.lower()

required = [
    "date", "product", "quantity",
    "unit_price", "region", "salesperson"
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(
        f"Missing columns: {missing}. "
        f"Found columns: {df.columns.tolist()}"
    )

# 2. Clean data
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
df["unit_price"] = pd.to_numeric(
    df["unit_price"], errors="coerce"
)

for col in ["product", "region", "salesperson"]:
    df[col] = df[col].astype("string").str.strip()

df = df.dropna(subset=required).copy()

df = df[
    (df["quantity"] > 0)
    & (df["unit_price"] >= 0)
    & (df["product"] != "")
    & (df["region"] != "")
    & (df["salesperson"] != "")
].copy()

if len(df) < 20:
    raise ValueError(
        "At least 20 valid sales records are required."
    )

# 3. Extract date features
df["month"] = df["date"].dt.month
df["weekday"] = df["date"].dt.dayofweek

# 4. Select features and target
categorical_features = ["product", "region", "salesperson"]
numeric_features = ["unit_price", "month", "weekday"]
features = categorical_features + numeric_features

X = df[features]
y = df["quantity"]

# 5. Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42
)

# 6. Prepare categorical data
preprocessor = ColumnTransformer([
    (
        "categories",
        OneHotEncoder(handle_unknown="ignore"),
        categorical_features
    ),
    ("numbers", "passthrough", numeric_features)
])

# 7. Create the model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=150,
            random_state=42,
            min_samples_leaf=2
        )
    )
])

# 8. Train
print("Training sales prediction model...")
model.fit(X_train, y_train)

# 9. Evaluate
predictions = model.predict(X_test)

print("\nMean Absolute Error (MAE):",
      round(mean_absolute_error(y_test, predictions), 2))

print("R2 score:", round(r2_score(y_test, predictions), 3))

# 10. Save model and metadata
MODEL_DIR.mkdir(parents=True, exist_ok=True)

joblib.dump({
    "model": model,
    "features": features,
    "products": sorted(df["product"].unique().tolist()),
    "regions": sorted(df["region"].unique().tolist()),
    "salespeople": sorted(
        df["salesperson"].unique().tolist()
    ),
    "price_min": float(df["unit_price"].min()),
    "price_max": float(df["unit_price"].max())
}, MODEL_PATH)

print("\nModel saved successfully!")
print("Location:", MODEL_PATH)
