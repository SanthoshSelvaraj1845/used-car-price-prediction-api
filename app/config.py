from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):

    MODEL_PATH: str = "ml/saved_model/model.joblib"

    LOG_LEVEL: str = "INFO"

    MAX_BATCH_SIZE: int = 100

    API_TITLE: str = "Used Car Price Prediction API"

    API_VERSION: str = "1.0.0"

    MODEL_INFO_PATH: str = "ml/saved_model/model_info.json"

    LOG_FILE_PATH: str = "logs/app.log"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()