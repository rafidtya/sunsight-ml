from ultralytics import YOLO

from roof_inference.core.config import settings

class ModelLoader:
    def __init__(self):
        self._model: YOLO | None = None
    
    def load(self):
        # Ini kalo modelnya belum ke load
        if self._model is None:
            # Kalo weightsnya ga nemu
            if not settings.MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Model Not Found: {settings.MODEL_PATH}"
                )
            # Load ke YOLO
            self._model = YOLO(str(settings.MODEL_PATH))
            # cek model segment apa bukan
            if self._model.task != "segment":
                raise ValueError(
                    f"Model is not segmented: {self._model.task}"
                )        
        return self._model

    def get_model(self):
        if self._model is None:
            raise RuntimeError("Model has not been loaded")
        return self._model
    
    def is_loaded(self) -> bool:
        if self._model is None:
            return False
        return True
    
model_loader = ModelLoader()
