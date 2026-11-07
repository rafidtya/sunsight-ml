import time

from PIL import Image

from roof_inference.models.loader import model_loader
from roof_inference.core.config import settings

def run_prediction(image: Image.Image):
    model = model_loader.get_model()

    start_time = time.perf_counter()

    result = model.predict(
        source=image,
        conf=settings.MODEL_CONFIDENCE,
        imgsz=settings.MODEL_IMAGE_SIZE,
        device=settings.MODEL_DEVICE,
        verbose=False,
        save=False
    )


    inference_time = (time.perf_counter() - start_time) * 1000

    return result[0], round(inference_time,2)