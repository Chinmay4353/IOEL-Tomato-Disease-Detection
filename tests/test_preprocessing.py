from pathlib import Path

import numpy as np
from PIL import Image

from src.data.preprocessing import preprocess_image
from src.data.validate_dataset import validate_dataset


def test_preprocess_image_returns_expected_shape():
    image = Image.fromarray(np.random.randint(0, 255, (120, 120, 3), dtype=np.uint8))
    array = preprocess_image(image, target_size=(224, 224))

    assert array.shape == (1, 224, 224, 3)
    assert array.dtype == np.float32


def test_validate_dataset_reports_counts(tmp_path):
    tomato_dir = tmp_path / "tomato"
    non_tomato_dir = tmp_path / "not_tomato"
    tomato_dir.mkdir()
    non_tomato_dir.mkdir()

    tomato_image = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))
    not_tomato_image = Image.fromarray(np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8))

    tomato_image.save(tomato_dir / "tomato_1.jpg")
    not_tomato_image.save(non_tomato_dir / "not_tomato_1.jpg")

    result = validate_dataset(tmp_path)

    assert result["total_images"] == 2
    assert result["tomato_count"] == 1
    assert result["not_tomato_count"] == 1
    assert result["invalid_images"] == 0
    assert result["duplicate_images"] == 0
