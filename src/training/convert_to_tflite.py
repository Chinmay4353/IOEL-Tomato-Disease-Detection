"""TensorFlow Lite conversion utilities for the tomato detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from src.utils.config import KERAS_MODEL_PATH, MODEL_PATH


def convert_to_tflite(model_path: str | Path = KERAS_MODEL_PATH, tflite_path: str | Path = MODEL_PATH, quantization: str = "float16") -> str:
    """Convert a trained Keras model to a TensorFlow Lite model with optional quantization."""
    model_path = Path(model_path)
    tflite_path = Path(tflite_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Keras model not found: {model_path}")

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == "dynamic_range":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "default":
        converter.optimizations = []
    else:
        raise ValueError(f"Unsupported quantization mode: {quantization}")

    tflite_path.parent.mkdir(parents=True, exist_ok=True)
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)
    return str(tflite_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the trained Keras model into TensorFlow Lite.")
    parser.add_argument("--model", type=str, default=str(KERAS_MODEL_PATH), help="Path to the trained Keras model.")
    parser.add_argument("--output", type=str, default=str(MODEL_PATH), help="Path for the converted .tflite file.")
    parser.add_argument("--quantization", type=str, default="float16", choices=["float16", "dynamic_range", "default"], help="Quantization mode.")
    args = parser.parse_args()

    output = convert_to_tflite(model_path=args.model, tflite_path=args.output, quantization=args.quantization)
    print(f"Model converted successfully: {output}")


if __name__ == "__main__":
    main()
