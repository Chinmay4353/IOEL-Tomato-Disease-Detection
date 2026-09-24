from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def load_image(image_path: str | Path) -> Image.Image:
    """
    Load an image and convert it to RGB.

    EXIF orientation is applied before conversion.
    """
    image = Image.open(image_path)
    return ImageOps.exif_transpose(image).convert("RGB")


def resize_image(
    image: Image.Image,
    target_size: tuple[int, int] = (224, 224),
) -> Image.Image:
    """
    Resize an image to the model input size.
    """
    return image.resize(target_size)


def preprocess_image(
    image: Image.Image,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Prepare an image for model inference.

    Important:
    The trained MobileNetV2 model already contains its own
    Rescaling layer that converts 0-255 pixels to approximately
    [-1, 1].

    Therefore, this function intentionally does NOT divide
    pixel values by 255.
    """
    rgb_image = ImageOps.exif_transpose(image).convert("RGB")
    resized = rgb_image.resize(target_size)

    array = np.asarray(
        resized,
        dtype=np.float32,
    )

    array = np.expand_dims(
        array,
        axis=0,
    )

    return array


def preprocess_path(
    image_path: str | Path,
    target_size: tuple[int, int] = (224, 224),
) -> np.ndarray:
    """
    Load an image from disk and prepare it for inference.
    """
    image = load_image(image_path)

    return preprocess_image(
        image,
        target_size,
    )