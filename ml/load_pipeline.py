import joblib
import pandas as pd

from pathlib import Path


# -----------------------------------
# 1. Locate saved pipeline
# -----------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

PIPELINE_PATH = (
    BASE_DIR
    / "ml"
    / "saved_model"
    / "used_car_pipeline.pkl"
)


# -----------------------------------
# 2. Load pipeline
# -----------------------------------

pipeline = joblib.load(PIPELINE_PATH)

print("Pipeline loaded successfully")


# -----------------------------------
# 3. New car data
# -----------------------------------

new_car = pd.DataFrame([
    {
        "name": "Hyundai i20 Sportz",
        "year": 2019,
        "km_driven": 35000,
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "owner": "First Owner"
    }
])


# -----------------------------------
# 4. Predict
# -----------------------------------

prediction = pipeline.predict(new_car)


# -----------------------------------
# 5. Display result
# -----------------------------------

print("Predicted selling price:", prediction[0])