from app.config import settings


def test_predict_batch_valid(client):

    payload = {
        "cars": [
            {
                "name": "Maruti Swift VXI",
                "year": 2020,
                "km_driven": 45000,
                "fuel": "Diesel",
                "seller_type": "Dealer",
                "transmission": "Manual",
                "owner": "First Owner"
            },
            {
                "name": "Hyundai i20",
                "year": 2019,
                "km_driven": 50000,
                "fuel": "Petrol",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner"
            }
        ]
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2

    assert data["predictions"][0]["prediction"] > 0
    assert data["predictions"][1]["prediction"] > 0


def test_predict_batch_oversized(client):

    car = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }

    payload = {
        "cars": [
            car
            for _ in range(settings.MAX_BATCH_SIZE + 1)
        ]
    }

    response = client.post(
        "/api/v1/predict-batch",
        json=payload
    )

    assert response.status_code == 400

    data = response.json()

    assert "detail" in data
    assert str(settings.MAX_BATCH_SIZE) in data["detail"]