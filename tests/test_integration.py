import httpx


BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my-used-car-api-secret-2026"

HEADERS = {
    "x-api-key": API_KEY
}


def test_health_integration():
    response = httpx.get(
        f"{BASE_URL}/api/v1/health",
        headers=HEADERS,
        timeout=10,
    )

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_metrics_integration():
    response = httpx.get(
        f"{BASE_URL}/metrics",
        timeout=10,
    )

    assert response.status_code == 200
    assert "# HELP" in response.text


def test_predict_integration():
    payload = {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner",
    }

    response = httpx.post(
        f"{BASE_URL}/api/v1/predict",
        headers=HEADERS,
        json=payload,
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data
    assert "prediction" in data
    assert "model_version" in data


def test_batch_predict_integration():
    payload = {
        "cars": [
            {
                "name": "Maruti Swift VXI",
                "year": 2020,
                "km_driven": 45000,
                "fuel": "Diesel",
                "seller_type": "Dealer",
                "transmission": "Manual",
                "owner": "First Owner",
            },
            {
                "name": "Hyundai i20",
                "year": 2019,
                "km_driven": 30000,
                "fuel": "Petrol",
                "seller_type": "Individual",
                "transmission": "Manual",
                "owner": "First Owner",
            },
        ]
    }

    response = httpx.post(
        f"{BASE_URL}/api/v1/predict-batch",
        headers=HEADERS,
        json=payload,
        timeout=10,
    )

    assert response.status_code == 200

    data = response.json()

    assert "predictions" in data
    assert len(data["predictions"]) == 2