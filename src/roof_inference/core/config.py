#Ini kayaknya yang related sama configuration. Dibikin jadi 1 biar ga repeatable

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# config.py -> core -> roof_inference (package root that holds weights/)
PACKAGE_ROOT = Path(__file__).resolve().parents[1]

class Settings(BaseSettings):
    APP_NAME:str =  "Sunsight"
    APP_VERSION:str = "1.0.0"
    API_V1_PREFIX:str = "/api/v1"
    # Bisa di-override lewat env MODEL_PATH
    MODEL_PATH: Path = PACKAGE_ROOT / "weights" / "best.pt"
    MODEL_DEVICE: str = "cpu"
    MODEL_CONFIDENCE: float = 0.25
    # Model dilatih pada imgsz=1024. Pengukuran pada citra satelit ubin zoom 19
    # di Bogor: 640 tidak menemukan satu bidang atap pun, 1024 menemukan 2-21
    # bidang, dan 1280 menemukan 8-13 bidang dengan confidence 0,66-0,79.
    MODEL_IMAGE_SIZE: int = 1280

    # Versi bobot yang benar-benar dimuat. Diisi dari env saat deploy supaya
    # setiap hasil deteksi dapat ditelusuri ke satu rilis bobot tertentu.
    MODEL_VERSION: str = "latest"

    MAX_IMAGE_SIZE: int = 10 * 1024 * 1024
    MAX_IMAGE_PIXELS: int = 20_000_000

    MAX_CONCURRENT_INFERENCES: int = 1

    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

settings = Settings()
