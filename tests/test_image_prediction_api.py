from io import BytesIO

from PIL import Image

from backend.app import app


def test_predict_image_endpoint_runs_inference_and_stores_result(monkeypatch):
    client = app.test_client()
    monkeypatch.setattr(
        "backend.app.predict_tflite_image",
        lambda model_path, image_path: {
            "prediction": "TOMATO_PLANT",
            "confidence": 0.94,
            "threshold": 0.70,
        },
    )

    image_buffer = BytesIO()
    Image.new("RGB", (8, 8), "red").save(image_buffer, format="JPEG")
    image_buffer.seek(0)

    response = client.post(
        "/api/predict-image",
        data={"image": (image_buffer, "dashboard-test.jpg"), "device_id": "dashboard-test"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 201
    assert response.get_json()["prediction"] == "TOMATO_PLANT"
    assert response.get_json()["image_url"].startswith("/captured-images/")