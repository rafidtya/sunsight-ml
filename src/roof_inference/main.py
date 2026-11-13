from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

import logging
import secrets
from roof_inference.core.config import settings
from roof_inference.models.loader import model_loader


from roof_inference.api.v1.endpoints.predict import router as predict_router

logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Loading... Yolo")
    
    model_loader.load()
    
    logging.info("Yolo loaded")

    yield

    logging.info("Application shutting down")

app = FastAPI(
    title= settings.APP_NAME,
    version= settings.APP_VERSION,
    lifespan=lifespan,
)

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    """Menolak permintaan prediksi yang tidak membawa kunci layanan.

    Hanya rute di bawah API_V1_PREFIX yang diperiksa. ``/health`` dan ``/ready``
    sengaja dibiarkan terbuka karena healthcheck container serta polling deploy
    berjalan dari mesin yang sama dan tidak memegang rahasia.
    """
    if settings.ML_API_KEY and request.url.path.startswith(f"{settings.API_V1_PREFIX}/"):
        supplied = request.headers.get("x-api-key", "")
        # Dibandingkan sebagai bytes: compare_digest menolak str non-ASCII, dan
        # perbandingan waktu-tetap tidak boleh diganti dengan `==`.
        if not secrets.compare_digest(
            supplied.encode("utf-8"), settings.ML_API_KEY.encode("utf-8")
        ):
            return JSONResponse(
                status_code=401,
                content={"detail": "Kunci layanan tidak valid"},
            )

    return await call_next(request)

app.include_router(
    predict_router,
    prefix=f"{settings.API_V1_PREFIX}",
    tags=["Prediction"],
)

@app.get("/health")
def health():
    return{"status":"ok"}

@app.get("/ready")
def readiness():
    if not model_loader.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model is not ready"
        )
    return{
        "status":"ready",
        "model_loaded": True
    }

