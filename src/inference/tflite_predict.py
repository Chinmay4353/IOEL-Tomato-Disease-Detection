"""TensorFlow Lite inference for tomato disease classification."""

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
    MODEL_PATH,
)


_INTERPRETER_CACHE: dict[
    str,
    tuple[
        tf.lite.Interpreter,
        list,
        list,
    ],
] = {}


def load_class_names(
    model_path: Path,
    output_size: int,
) -> list[str]:
    """
    Load the class names associated with the model.
    """

    metadata_path = model_path.with_name(
        "class_names.json"
    )

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: "
            f"{metadata_path}"
        )

    class_names = json.loads(
        metadata_path.read_text(
            encoding="utf-8"
        )
    )

    if len(class_names) != output_size:
        raise ValueError(
            f"Model has {output_size} outputs, "
            f"but class_names.json contains "
            f"{len(class_names)} classes."
        )

    return class_names


def _get_interpreter(
    model_path: Path,
):
    """
    Load and cache the TFLite interpreter.
    """

    cache_key = str(
        model_path.resolve()
    )

    if cache_key not in _INTERPRETER_CACHE:
        interpreter = tf.lite.Interpreter(
            model_path=cache_key
        )

        interpreter.allocate_tensors()

        _INTERPRETER_CACHE[cache_key] = (
            interpreter,
            interpreter.get_input_details(),
            interpreter.get_output_details(),
        )

    return _INTERPRETER_CACHE[cache_key]


def predict_tflite_image(
    model_path: str | Path = MODEL_PATH,
    image_path: str | Path = "",
    threshold: float = CONFIDENCE_THRESHOLD,
) -> dict[str, Any]:
    """
    Run 11-class tomato disease inference
    using TensorFlow Lite.
    """

    model_path = Path(model_path)
    image_path = Path(image_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"TFLite model not found: {model_path}"
        )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    try:
        with Image.open(image_path) as image:
            image.verify()
    except Exception as exc:
        raise ValueError(
            f"Invalid image: {image_path}"
        ) from exc

    (
        interpreter,
        input_details_list,
        output_details_list,
    ) = _get_interpreter(model_path)

    input_details = input_details_list[0]
    output_details = output_details_list[0]

    image_array = preprocess_path(
        image_path
    )

    input_dtype = input_details["dtype"]

    input_data = image_array.astype(
        input_dtype
    )

    interpreter.set_tensor(
        input_details["index"],
        input_data,
    )

    interpreter.invoke()

    output = np.asarray(
        interpreter.get_tensor(
            output_details["index"]
        )[0]
    ).reshape(-1)

    if output.size <= 1:
        raise ValueError(
            "The TFLite model returned a single "
            "output. Expected an 11-class model."
        )

    class_names = load_class_names(
        model_path,
        output.size,
    )

    predicted_index = int(
        np.argmax(output)
    )

    confidence = float(
        output[predicted_index]
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
            "TensorFlow Lite."
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
        default=str(MODEL_PATH),
        help="Path to the TFLite model.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold between 0 and 1.",
    )

    args = parser.parse_args()

    result = predict_tflite_image(
        model_path=args.model,
        image_path=args.image,
        threshold=args.threshold,
    )

    print("\n===============================")
    print("Tomato Disease Detector (TFLite)")
    print("===============================")
    print(
        f"Prediction: {result['prediction']}"
    )
    print(
        f"Label:      {result['label']}"
    )
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