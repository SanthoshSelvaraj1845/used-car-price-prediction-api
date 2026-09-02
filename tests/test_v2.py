# ---------------------------------
# Test V2 Prediction
# ---------------------------------

def test_v2_predict_valid_input(client):

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

        "/api/v2/predict",

        json=payload
    )

    assert response.status_code == 200

    data = response.json()

    assert "request_id" in data

    assert "predicted_price" in data

    assert "model_version" in data

    assert isinstance(
        data["predicted_price"],
        float
    )

    assert data["predicted_price"] > 0


# ---------------------------------
# Test V2 Validation
# ---------------------------------

def test_v2_predict_invalid_km_driven(client):

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

        "/api/v2/predict",

        json=payload
    )

    assert response.status_code == 422


# ---------------------------------
# Test V1 vs V2 Response Shape
# ---------------------------------

def test_v1_and_v2_have_different_response_shapes(
    client
):

    payload = {

        "name": "Maruti Swift VXI",

        "year": 2020,

        "km_driven": 45000,

        "fuel": "Diesel",

        "seller_type": "Dealer",

        "transmission": "Manual",

        "owner": "First Owner"
    }


    # -----------------------------
    # Call V1
    # -----------------------------

    v1_response = client.post(

        "/api/v1/predict",

        json=payload
    )


    # -----------------------------
    # Call V2
    # -----------------------------

    v2_response = client.post(

        "/api/v2/predict",

        json=payload
    )


    # -----------------------------
    # Both should work
    # -----------------------------

    assert v1_response.status_code == 200

    assert v2_response.status_code == 200


    v1_data = v1_response.json()

    v2_data = v2_response.json()


    # -----------------------------
    # V1 response
    # -----------------------------

    assert "prediction" in v1_data

    assert "confidence_score" in v1_data


    # -----------------------------
    # V2 response
    # -----------------------------

    assert "predicted_price" in v2_data


    # -----------------------------
    # Prove they are different
    # -----------------------------

    assert "predicted_price" not in v1_data

    assert "prediction" not in v2_data