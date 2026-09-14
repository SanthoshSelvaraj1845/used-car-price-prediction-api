import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


# -----------------------------------
# 1. File paths
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "used_cars.csv"

MODEL_DIR = BASE_DIR / "ml" / "saved_model"

MODEL_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = MODEL_DIR / "used_car_pipeline.pkl"


# -----------------------------------
# 2. Load dataset
# -----------------------------------

df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully")
print("Shape:", df.shape)


# -----------------------------------
# 3. Features and target
# -----------------------------------

features = [
    "name",
    "year",
    "km_driven",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]

target = "selling_price"

X = df[features]
y = df[target]


# -----------------------------------
# 4. Train/test split
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------------
# 5. Preprocessing
# -----------------------------------

numeric_features = [
    "year",
    "km_driven"
]

categorical_features = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            numeric_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        )
    ]
)


# -----------------------------------
# 6. Model
# -----------------------------------

model = RandomForestRegressor(
    random_state=42
)


# -----------------------------------
# 7. Create Pipeline
# -----------------------------------

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# -----------------------------------
# 8. Train Pipeline
# -----------------------------------

pipeline.fit(X_train, y_train)

print("Pipeline trained successfully")


# -----------------------------------
# 9. Test prediction
# -----------------------------------

predictions = pipeline.predict(X_test)

print("First 5 predictions:")
print(predictions[:5])


# -----------------------------------
# 10. Save Pipeline
# -----------------------------------

joblib.dump(
    pipeline,
    PIPELINE_PATH
)

print()
print("Pipeline saved successfully!")
print("Saved at:", PIPELINE_PATH)