#Ini kayaknya yang related sama configuration. Dibikin jadi 1 biar ga repeatable

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Ini kayaknya salah
PROJECT_ROOT = Path(__file__).resolve().parents[3]

class Settings(BaseSettings):
    APP_NAME:str =  "Sunsight"
    APP_VERSION:str = "1.0.0"
    API_V1_PREFIX:str = "/api/v1"
    #Ini kayaknya salah
    MODEL_PATH: Path = PROJECT_ROOT / "weights" / "best.pt"
    MODEL_DEVICE: str = "cpu"
    MODEL_CONFIDENCE: float = 0.25
    MODEL_IMAGE_SIZE: int = 640

    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024
    MAX_IMAGE_PIXELS: int = 20_000_000

    MAX_CONCCURENT_INFERENCES: int = 1

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings()
