from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from src.inference.tflite_predict import predict_tflite_image


def test_predict_tflite_image_runs(tmp_path):
    model_path = tmp_path / "model.tflite"
    image_path = tmp_path / "sample.jpg"

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    model_bytes = converter.convert()
    model_path.write_bytes(model_bytes)

    image = Image.new("RGB", (224, 224), color=(0, 255, 0))
    image.save(image_path)

    result = predict_tflite_image(model_path=model_path, image_path=image_path, threshold=0.70)

    assert "prediction" in result
    assert "confidence" in result
    assert "label" in result
