import asyncio
from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool

from roof_inference.core.config import settings
from roof_inference.services.image_processor import process_image
from roof_inference.services.predictor import run_prediction
from roof_inference.services.postprocessor import process_results

from roof_inference.schemas.prediction import (
    PredictionResponse,
    PredictionData,
    PredictionMeta,
    ImageMetadata,
)

router = APIRouter()

inference_semaphore = asyncio.Semaphore(
    settings.MAX_CONCURRENT_INFERENCES
)


@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(file: UploadFile = File(...)):

    content = await file.read(settings.MAX_IMAGE_SIZE + 1)

    try:
        if len(content) > settings.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds maximum allowed size"
            )

        image = process_image(content)

        async with inference_semaphore:
            result, inference_time = await run_in_threadpool(
                run_prediction,
                image
            )

        detections = process_results(result)

        return PredictionResponse(
            success=True,
            data=PredictionData(
                image=ImageMetadata(
                    width=result.orig_shape[1],
                    height=result.orig_shape[0],
                ),
                total_detections=len(detections),
                detections=detections,
            ),
            meta=PredictionMeta(
                model="yolov8-seg",
                inference_time_ms=inference_time,
            ),
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Internal inference error",
        ) from exc

    finally:
        await file.close()