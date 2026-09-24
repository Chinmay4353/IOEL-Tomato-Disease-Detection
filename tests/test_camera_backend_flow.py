from pathlib import Path

from src.inference.camera_app import CameraApp


def test_capture_and_predict_reports_result_to_backend(monkeypatch, tmp_path):
    app = CameraApp(backend_url="http://localhost:8000/api/predictions", device_id="pi-test")
    app.cap = object()

    monkeypatch.setattr(app, "capture_frame", lambda: object())
    saved_path = tmp_path / "capture.jpg"
    monkeypatch.setattr(app, "save_image", lambda frame: saved_path)
    monkeypatch.setattr(
        app,
        "run_tflite_inference",
        lambda image_path: {"prediction": "TOMATO_PLANT", "confidence": 0.91, "threshold": 0.70},
    )

    captured = {}

    def fake_send(**kwargs):
        captured.update(kwargs)
        return {"id": 1, "prediction": "TOMATO_PLANT"}

    monkeypatch.setattr("src.inference.camera_app.send_prediction_to_backend", fake_send)

    returned_path, result = app.capture_and_predict()

    assert returned_path == Path(saved_path)
    assert result["prediction"] == "TOMATO_PLANT"
    assert captured["api_url"] == "http://localhost:8000/api/predictions"
    assert captured["device_id"] == "pi-test"
    assert captured["image_path"] == str(saved_path)
    assert app.last_backend_response["id"] == 1
