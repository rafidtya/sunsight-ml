import warnings
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from roof_inference.core.config import settings


ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


def process_image(content: bytes) -> Image.Image:
    if not content:
        raise ValueError("Empty image")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter(
                "error",
                Image.DecompressionBombWarning
            )

            with Image.open(BytesIO(content)) as image:
                if image.format not in ALLOWED_FORMATS:
                    raise ValueError("Unsupported image format")

                if image.width * image.height > settings.MAX_IMAGE_PIXELS:
                    raise ValueError("Image resolution too large")

                image.verify()

            with Image.open(BytesIO(content)) as image:
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")
                image.load()

                return image.copy()

    except (
        UnidentifiedImageError,
        OSError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ValueError("Invalid or unsafe image") from exc