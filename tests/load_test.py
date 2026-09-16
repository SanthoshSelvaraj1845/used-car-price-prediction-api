import asyncio
import time

import httpx


BASE_URL = "http://127.0.0.1:8000"
API_KEY = "my-secret-key"

HEADERS = {
    "X-API-Key": API_KEY
}

PAYLOAD = {
    "name": "Maruti Swift VXI",
    "year": 2020,
    "km_driven": 45000,
    "fuel": "Diesel",
    "seller_type": "Dealer",
    "transmission": "Manual",
    "owner": "First Owner",
}


async def send_request(client):
    start = time.perf_counter()

    try:
        response = await client.post(
            f"{BASE_URL}/api/v1/predict",
            headers=HEADERS,
            json=PAYLOAD,
            timeout=10,
        )

        elapsed = time.perf_counter() - start

        return response.status_code, elapsed

    except Exception as error:
        elapsed = time.perf_counter() - start

        return f"ERROR: {error}", elapsed


async def main():
    total_requests = 50

    async with httpx.AsyncClient() as client:
        start = time.perf_counter()

        results = await asyncio.gather(
            *[
                send_request(client)
                for _ in range(total_requests)
            ]
        )

        total_time = time.perf_counter() - start

    successful = [
        result for result in results
        if result[0] == 200
    ]

    failed = [
        result for result in results
        if result[0] != 200
    ]

    response_times = [
        result[1]
        for result in results
    ]

    print("\n===== LOAD TEST RESULTS =====")
    print(f"Total requests : {total_requests}")
    print(f"Successful     : {len(successful)}")
    print(f"Failed         : {len(failed)}")
    print(f"Total time     : {total_time:.2f} seconds")
    print(f"Average time   : {sum(response_times) / len(response_times):.4f} seconds")
    print(f"Fastest        : {min(response_times):.4f} seconds")
    print(f"Slowest        : {max(response_times):.4f} seconds")


if __name__ == "__main__":
    asyncio.run(main())