import asyncio
import uuid
from typing import Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Query
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
async def predict(
    file: UploadFile = File(...),
    roi_x1: Optional[int] = Query(0, ge=0),
    roi_y1: Optional[int] = Query(0, ge=0),
    roi_x2: Optional[int] = None,
    roi_y2: Optional[int] = None,
    model_version: Optional[str] = Query("latest"),
    min_confidence: Optional[float] = Query(0.25, ge=0.0, le=1.0),
):
    request_id = f"req_{uuid.uuid4().hex[:12]}"
    content = await file.read(settings.MAX_IMAGE_SIZE + 1)

    try:
        if len(content) > settings.MAX_IMAGE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="Image exceeds maximum allowed size"
            )

        image = process_image(content)
        orig_height, orig_width = image.shape[:2]

        roi_x2 = roi_x2 or orig_width
        roi_y2 = roi_y2 or orig_height

        if roi_x1 >= roi_x2 or roi_y1 >= roi_y2:
            raise HTTPException(
                status_code=400,
                detail="Invalid ROI coordinates"
            )

        async with inference_semaphore:
            result, inference_time = await run_in_threadpool(
                run_prediction,
                image
            )

        detections = process_results(result, min_confidence=min_confidence)

        if roi_x1 > 0 or roi_y1 > 0 or roi_x2 < orig_width or roi_y2 < orig_height:
            detections = [
                d for d in detections
                if _is_inside_roi(d["bbox"], roi_x1, roi_y1, roi_x2, roi_y2)
            ]

        return PredictionResponse(
            success=True,
            request_id=request_id,
            data=PredictionData(
                image=ImageMetadata(
                    width=orig_width,
                    height=orig_height,
                ),
                total_detections=len(detections),
                detections=detections,
            ),
            meta=PredictionMeta(
                model="yolov8-seg",
                model_version=model_version,
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
        return PredictionResponse(
            success=False,
            request_id=request_id,
            error="Internal inference error",
        )

    finally:
        await file.close()


def _is_inside_roi(bbox, roi_x1, roi_y1, roi_x2, roi_y2):
    """Check if bbox center is inside ROI."""
    cx = (bbox["x1"] + bbox["x2"]) / 2
    cy = (bbox["y1"] + bbox["y2"]) / 2
    return roi_x1 <= cx <= roi_x2 and roi_y1 <= cy <= roi_y2