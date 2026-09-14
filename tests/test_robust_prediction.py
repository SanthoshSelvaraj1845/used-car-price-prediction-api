import pytest

from ml.robust_prediction import predict_new_car

def test_valid_car_prediction():

    car = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    prediction = predict_new_car(car)

    assert isinstance(prediction, float)

def test_missing_required_field():

    car = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual"
    }

    with pytest.raises(ValueError, match="Missing required field"):
        predict_new_car(car)

def test_negative_km_driven():

    car = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": -500,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    with pytest.raises(
        ValueError,
        match="km_driven cannot be negative"
    ):
        predict_new_car(car)

def test_invalid_year_type():

    car = {
        "name": "Maruti Swift VXI",
        "year": "2020",
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    with pytest.raises(
        TypeError,
        match="year must be an integer"
    ):
        predict_new_car(car)