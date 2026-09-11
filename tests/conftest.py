
import pytest

from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():

    with TestClient(
        app,
        headers={
            "X-API-Key": "my-used-car-api-secret-2026"
        }
    ) as test_client:

        yield test_client
