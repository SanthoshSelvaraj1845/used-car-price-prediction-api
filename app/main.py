from contextlib import asynccontextmanager
import time
import uuid

import joblib

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_config import setup_logger
from app.routers.v1 import router as v1_router


logger = setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):

    model = joblib.load(
        "ml/saved_model/model.joblib"
    )

    app.state.model = model

    logger.info("ML model loaded successfully")

    yield


app = FastAPI(
    title="Used Car Price Prediction API",
    description="API for predicting used car prices using Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)


app.include_router(v1_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):

    request_id = str(uuid.uuid4())

    request.state.request_id = request_id

    start_time = time.perf_counter()

    logger.info(
        f"Request started | "
        f"request_id={request_id} | "
        f"method={request.method} | "
        f"path={request.url.path}"
    )

    try:

        response = await call_next(request)

        duration = time.perf_counter() - start_time

        logger.info(
            f"Request completed | "
            f"request_id={request_id} | "
            f"method={request.method} | "
            f"path={request.url.path} | "
            f"status_code={response.status_code} | "
            f"duration={duration:.4f}s"
        )

        response.headers["X-Request-ID"] = request_id

        return response

    except Exception:

        duration = time.perf_counter() - start_time

        logger.exception(
            f"Request failed | "
            f"request_id={request_id} | "
            f"method={request.method} | "
            f"path={request.url.path} | "
            f"duration={duration:.4f}s"
        )

        raise


@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError
):

    logger.error(
        f"ValueError | "
        f"request_id={getattr(request.state, 'request_id', 'unknown')} | "
        f"error={exc}"
    )

    return JSONResponse(
        status_code=400,
        content={
            "error": "Invalid value",
            "message": str(exc)
        }
    )


@app.get("/")
def root():
    return {
        "message": "ML API is alive"
    }