from pathlib import Path

from src.inference.camera_app import CameraApp


def test_camera_app_initialization():
    app = CameraApp(camera_index=0, model_path="model/model.tflite")
    assert app.camera_index == 0
    assert app.model_path.endswith("model.tflite")
