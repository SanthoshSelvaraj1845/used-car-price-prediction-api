from fastapi.testclient import TestClient 
from app.main import app


def valid_payload():
    return {
        "name": "Maruti Swift VXI",
        "year": 2020,
        "km_driven": 45000,
        "fuel": "Diesel",
        "seller_type": "Dealer",
        "transmission": "Manual",
        "owner": "First Owner"
    }


def test_missing_api_key(): 
    client = TestClient(app) 
    response = client.post( "/api/v1/predict", json=valid_payload() )

    assert response.status_code == 401 
    assert response.json()["detail"] == "Missing API key"


def test_invalid_api_key(client):
    response = client.post(
        "/api/v1/predict",
        json=valid_payload(),
        headers={
            "X-API-Key": "wrong-api-key"
        }
    )

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API key"


def test_unexpected_extra_field(client):
    payload = valid_payload()

    payload["unexpected_field"] = "not allowed"

    response = client.post(
        "/api/v1/predict",
        json=payload,
        headers={
            "X-API-Key": "my-used-car-api-secret-2026"
        }
    )

    assert response.status_code == 422