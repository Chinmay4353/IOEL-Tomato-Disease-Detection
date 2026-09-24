from pathlib import Path

import tensorflow as tf
from PIL import Image

from src.training.convert_to_tflite import convert_to_tflite


def test_convert_to_tflite_creates_model(tmp_path):
    model_path = tmp_path / "model.keras"
    tflite_path = tmp_path / "model.tflite"

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.save(model_path)

    output = convert_to_tflite(model_path=model_path, tflite_path=tflite_path, quantization="float16")

    assert output == str(tflite_path)
    assert tflite_path.exists()
