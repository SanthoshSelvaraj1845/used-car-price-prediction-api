import pandas as pd
import joblib


# Load saved model
model = joblib.load(
    "ml/saved_model/model.joblib"
)


# Create one new car
car = pd.DataFrame([
    {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }
])


# Make prediction
prediction = model.predict(car)


print("Predicted used car price:", prediction[0])