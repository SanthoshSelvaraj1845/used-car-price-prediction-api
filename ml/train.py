import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load dataset
df = pd.read_csv("data/used_cars.csv")

print("Dataset shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# Select features and target
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


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Define numerical features
numeric_features = [
    "year",
    "km_driven"
]


# Define categorical features
categorical_features = [
    "name",
    "fuel",
    "seller_type",
    "transmission",
    "owner"
]


# Create preprocessing pipeline
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


# Create complete ML pipeline
model = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)


# Train model
print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed.")


# Make predictions
predictions = model.predict(X_test)


# 10. Evaluate model
mae = mean_absolute_error(
    y_test,
    predictions
)

mse = mean_squared_error(
    y_test,
    predictions
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# Print metrics
print("\nModel Evaluation")
print("----------------")

print("MAE:", mae)
print("RMSE:", rmse)
print("R2 Score:", r2)


#Save complete pipeline
model_path = "ml/saved_model/model.joblib"

joblib.dump(
    model,
    model_path
)

print("\nModel saved successfully:")
print(model_path)