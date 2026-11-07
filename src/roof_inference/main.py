from fastapi import FastAPI,HTTPException
from contextlib import asynccontextmanager

import logging

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

