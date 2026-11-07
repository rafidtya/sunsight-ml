from pydantic import BaseModel
from typing import Optional

class ImageMetadata(BaseModel):
    width: int
    height: int

class BoundingBox(BaseModel):
    x1:float
    y1:float
    x2:float
    y2:float

class Detection(BaseModel):
    id: str
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    polygon: list[list[float]]
    area_px: float

class PredictionData(BaseModel):
    image: ImageMetadata
    total_detections: int
    detections: list[Detection]

class PredictionMeta(BaseModel):
    model: str
    model_version: Optional[str] = None
    inference_time_ms: float

class PredictionResponse(BaseModel):
    success: bool
    request_id: Optional[str] = None
    data: Optional[PredictionData] = None
    error: Optional[str] = None
    meta: Optional[PredictionMeta] = None