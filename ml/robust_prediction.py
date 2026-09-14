import joblib
import pandas as pd
from pathlib import Path


MODEL_PATH = Path(__file__).parent / "saved_model" / "model.joblib"


def predict_new_car(car_data):
    """
    Validate new car data and return predicted selling price.
    """

    required_fields = [
        "name",
        "year",
        "km_driven",
        "fuel",
        "seller_type",
        "transmission",
        "owner"
    ]

    # -----------------------------
    # 1. Validate required fields
    # -----------------------------
    for field in required_fields:
        if field not in car_data:
            raise ValueError(
                f"Missing required field: {field}"
            )

    # -----------------------------
    # 2. Validate data types
    # -----------------------------
    if not isinstance(car_data["name"], str):
        raise TypeError("name must be a string")

    if not isinstance(car_data["year"], int):
        raise TypeError("year must be an integer")

    if not isinstance(car_data["km_driven"], (int, float)):
        raise TypeError("km_driven must be a number")

    if not isinstance(car_data["fuel"], str):
        raise TypeError("fuel must be a string")

    if not isinstance(car_data["seller_type"], str):
        raise TypeError("seller_type must be a string")

    if not isinstance(car_data["transmission"], str):
        raise TypeError("transmission must be a string")

    if not isinstance(car_data["owner"], str):
        raise TypeError("owner must be a string")

    # -----------------------------
    # 3. Validate numeric ranges
    # -----------------------------
    if car_data["year"] <= 1900:
        raise ValueError(
            "year must be greater than 1900"
        )

    if car_data["km_driven"] < 0:
        raise ValueError(
            "km_driven cannot be negative"
        )

    # -----------------------------
    # 4. Validate categories
    # -----------------------------
    valid_fuels = [
        "Diesel",
        "Petrol",
        "CNG",
        "LPG",
        "Electric"
    ]

    valid_seller_types = [
        "Individual",
        "Dealer",
        "Trustmark Dealer"
    ]

    valid_transmissions = [
        "Manual",
        "Automatic"
    ]

    valid_owners = [
        "First Owner",
        "Second Owner",
        "Third Owner",
        "Fourth & Above Owner",
        "Test Drive Car"
    ]

    if car_data["fuel"] not in valid_fuels:
        raise ValueError(
            f"Invalid fuel: {car_data['fuel']}"
        )

    if car_data["seller_type"] not in valid_seller_types:
        raise ValueError(
            f"Invalid seller_type: {car_data['seller_type']}"
        )

    if car_data["transmission"] not in valid_transmissions:
        raise ValueError(
            f"Invalid transmission: {car_data['transmission']}"
        )

    if car_data["owner"] not in valid_owners:
        raise ValueError(
            f"Invalid owner: {car_data['owner']}"
        )

    # -----------------------------
    # 5. Convert to DataFrame
    # -----------------------------
    input_data = pd.DataFrame([car_data])

    # -----------------------------
    # 6. Load model and predict
    # -----------------------------
    try:
        model = joblib.load(MODEL_PATH)

        prediction = model.predict(input_data)

        return float(prediction[0])

    except Exception as error:
        raise RuntimeError(
            f"Prediction failed: {error}"
        ) from error