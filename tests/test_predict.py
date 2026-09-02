def test_predict_valid_input(client):

    payload = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert "prediction" in data
    assert "confidence_score" in data
    assert "model_version" in data

    assert isinstance(data["prediction"], float)
    assert data["prediction"] > 0
    assert data["confidence_score"] is None


def test_predict_missing_required_field(client):

    payload = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_predict_negative_km_driven(client):

    payload = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": -500,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422


def test_predict_invalid_year(client):

    payload = {
        "name": "Maruti Swift VXI",
        "year": 1800,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    response = client.post(
        "/api/v1/predict",
        json=payload
    )

    assert response.status_code == 422