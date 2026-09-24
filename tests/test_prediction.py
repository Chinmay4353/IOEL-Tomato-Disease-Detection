from pathlib import Path

import tensorflow as tf
from PIL import Image

from src.inference.predict import predict_image, threshold_prediction


def test_threshold_prediction_works():
    assert threshold_prediction(0.95, 0.70) == "TOMATO_PLANT"
    assert threshold_prediction(0.40, 0.70) == "UNCERTAIN"


def test_predict_image_for_rgb_image(tmp_path):
    image_path = tmp_path / "sample.jpg"
    model_path = tmp_path / "sample_model.keras"
    image = Image.new("RGB", (224, 224), color=(0, 255, 0))
    image.save(image_path)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(224, 224, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.save(model_path)

    result = predict_image(model_path=model_path, image_path=str(image_path), threshold=0.70)

    assert "prediction" in result
    assert "confidence" in result
    assert "label" in result
