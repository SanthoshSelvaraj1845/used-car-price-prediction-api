import logging

from logging.handlers import RotatingFileHandler

from app.config import settings


def setup_logger():

    logger = logging.getLogger(
        "used_car_api"
    )

    logger.setLevel(
        getattr(
            logging,
            settings.LOG_LEVEL.upper()
        )
    )

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    )

    # ---------------------------------
    # Console Handler
    # ---------------------------------

    console_handler = logging.StreamHandler()

    console_handler.setLevel(
        getattr(
            logging,
            settings.LOG_LEVEL.upper()
        )
    )

    console_handler.setFormatter(
        formatter
    )

    # ---------------------------------
    # File Handler
    # ---------------------------------

    file_handler = RotatingFileHandler(
        settings.LOG_FILE_PATH,
        maxBytes=5_000_000,
        backupCount=3
    )

    file_handler.setLevel(
        getattr(
            logging,
            settings.LOG_LEVEL.upper()
        )
    )

    file_handler.setFormatter(
        formatter
    )

    # ---------------------------------
    # Add Handlers
    # ---------------------------------

    logger.addHandler(
        console_handler
    )

    logger.addHandler(
        file_handler
    )

    return logger