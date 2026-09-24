"""Keras inference for tomato disease classification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from src.data.preprocessing import preprocess_path
from src.utils.config import (
    CONFIDENCE_THRESHOLD,
    KERAS_MODEL_PATH,
)


def load_class_names(model_path: Path) -> list[str]:
    """
    Load class names stored next to the model.
    """
    class_names_path = model_path.with_name(
        "class_names.json"
    )

    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_names_path}"
        )

    class_names = json.loads(
        class_names_path.read_text(
            encoding="utf-8"
        )
    )

    if not isinstance(class_names, list):
        raise ValueError(
            "class_names.json must contain a JSON list."
        )

    return class_names


def predict_image(
    model_path: str | Path | None = None,
    image_path: str | Path = "",
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """
    Predict the tomato disease class for one image.
    """

    selected_model_path = (
        Path(model_path)
        if model_path is not None
        else KERAS_MODEL_PATH
    )

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not selected_model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {selected_model_path}"
        )

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(
            f"Invalid image: {image_path}"
        ) from exc

    model = tf.keras.models.load_model(
        selected_model_path
    )

    class_names = load_class_names(
        selected_model_path
    )

    image_array = preprocess_path(
        image_path
    )

    probabilities = model.predict(
        image_array,
        verbose=0,
    )[0]

    probabilities = np.asarray(
        probabilities
    ).reshape(-1)

    if probabilities.size != len(class_names):
        raise ValueError(
            f"Model returned {probabilities.size} outputs, "
            f"but class_names.json contains "
            f"{len(class_names)} classes."
        )

    predicted_index = int(
        np.argmax(probabilities)
    )

    confidence = float(
        probabilities[predicted_index]
    )

    prediction = class_names[
        predicted_index
    ]

    display_label = (
        "Healthy"
        if prediction.lower() == "healthy"
        else prediction.replace("_", " ")
    )

    confidence_status = (
        "HIGH_CONFIDENCE"
        if confidence >= threshold
        else "LOW_CONFIDENCE"
    )

    return {
        "prediction": prediction,
        "confidence": confidence,
        "threshold": float(threshold),
        "confidence_status": confidence_status,
        "label": display_label,
        "disease": (
            None
            if prediction.lower() == "healthy"
            else display_label
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Predict tomato disease using "
            "the trained Keras model."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the image file.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(KERAS_MODEL_PATH),
        help="Path to the Keras model.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0 and 1.",
    )

    args = parser.parse_args()

    result = predict_image(
        model_path=args.model,
        image_path=args.image,
        threshold=args.threshold,
    )

    print("\n===============================")
    print("Tomato Disease Detector")
    print("===============================")
    print(f"Prediction: {result['prediction']}")
    print(f"Label:      {result['label']}")
    print(
        f"Confidence: "
        f"{result['confidence'] * 100:.2f}%"
    )
    print(
        f"Status:     "
        f"{result['confidence_status']}"
    )
    print("===============================\n")


if __name__ == "__main__":
    main()