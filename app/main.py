import time
import uuid

import joblib

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import settings
from app.logging_config import setup_logger
from app.routers.v1 import router as v1_router


# ---------------------------------
# Logger
# ---------------------------------

logger = setup_logger()


# ---------------------------------
# Lifespan
# ---------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Load ML model using configuration

    model = joblib.load(
        settings.MODEL_PATH
    )

    app.state.model = model

    logger.info(
        "ML model loaded successfully"
    )

    yield

    logger.info(
        "Application shutting down"
    )


# ---------------------------------
# FastAPI Application
# ---------------------------------

app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="ML API for predicting used car prices",
    lifespan=lifespan
)


# ---------------------------------
# Include Version 1 Router
# ---------------------------------

app.include_router(
    v1_router
)


# ---------------------------------
# Request Logging Middleware
# ---------------------------------

@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next
):

    request_id = str(
        uuid.uuid4()
    )

    request.state.request_id = request_id

    start_time = time.perf_counter()

    logger.info(
        f"Request started | "
        f"request_id={request_id} | "
        f"method={request.method} | "
        f"path={request.url.path}"
    )

    try:

        response = await call_next(
            request
        )

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.info(
            f"Request completed | "
            f"request_id={request_id} | "
            f"status_code={response.status_code} | "
            f"duration={duration:.4f}s"
        )

        response.headers[
            "X-Request-ID"
        ] = request_id

        return response

    except Exception as e:

        duration = (
            time.perf_counter()
            - start_time
        )

        logger.exception(
            f"Request failed | "
            f"request_id={request_id} | "
            f"duration={duration:.4f}s | "
            f"error={e}"
        )

        raise


# ---------------------------------
# ValueError Handler
# ---------------------------------

@app.exception_handler(ValueError)
async def value_error_handler(
    request: Request,
    exc: ValueError
):

    request_id = getattr(
        request.state,
        "request_id",
        "unknown"
    )

    logger.error(
        f"ValueError | "
        f"request_id={request_id} | "
        f"error={exc}"
    )

    return JSONResponse(

        status_code=400,

        content={
            "error": "Invalid value",
            "message": str(exc),
            "request_id": request_id
        }
    )


# ---------------------------------
# Root Endpoint
# ---------------------------------

@app.get("/")
def root():

    return {
        "message": "ML API is alive"
    }