def test_model_info(client):

    response = client.get(
        "/api/v1/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert "model_type" in data
    assert "model_version" in data
    assert "training_date" in data
    assert "features" in data
    assert "target" in data

    assert data["model_type"] == "RandomForestRegressor"
    assert data["target"] == "selling_price"

    assert isinstance(data["features"], list)

    assert "year" in data["features"]
    assert "km_driven" in data["features"]