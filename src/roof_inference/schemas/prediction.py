from pydantic import BaseModel

class ImageMetadata(BaseModel):
    width: int
    height: int

class BoundingBox(BaseModel):
    x1:float
    y1:float
    x2:float
    y2:float

class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    polygon: list[list[float]]

class PredictionData(BaseModel):
    image: ImageMetadata
    total_detections: int
    detections: list[Detection]

class PredictionMeta(BaseModel):
    model: str
    inference_time_ms: float

class PredictionResponse(BaseModel):
    success: bool
    data: PredictionData
    meta: PredictionMeta