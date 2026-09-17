import httpx

from app.config import settings

BASE_URL = "http://127.0.0.1:8000"
API_KEY = settings.API_KEY

HEADERS = {
    "x-api-key": API_KEY
}