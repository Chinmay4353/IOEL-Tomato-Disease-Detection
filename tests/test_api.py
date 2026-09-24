from pathlib import Path

from backend.app import app


def test_backend_health_endpoint():
    client = app.test_client()
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_backend_prediction_post_and_list():
    client = app.test_client()
    payload = {
        "prediction": "TOMATO_PLANT",
        "confidence": 0.96,
        "device_id": "device-01",
        "timestamp": "2026-09-06T12:00:00",
        "image_path": "/tmp/example.jpg",
    }

    post_response = client.post("/api/predictions", json=payload)
    assert post_response.status_code == 201

    list_response = client.get("/api/predictions")
    assert list_response.status_code == 200
    data = list_response.get_json()
    assert any(item["prediction"] == "TOMATO_PLANT" for item in data)
