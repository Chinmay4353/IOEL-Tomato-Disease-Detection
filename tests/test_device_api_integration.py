from src.inference.device_client import build_prediction_payload, send_prediction_to_backend


def test_build_prediction_payload_contains_expected_fields():
    payload = build_prediction_payload(
        prediction="TOMATO_PLANT",
        confidence=0.96,
        device_id="raspberry-pi-01",
        image_path="/tmp/capture.jpg",
    )

    assert payload["prediction"] == "TOMATO_PLANT"
    assert payload["confidence"] == 0.96
    assert payload["device_id"] == "raspberry-pi-01"
    assert payload["image_path"] == "/tmp/capture.jpg"
    assert "timestamp" in payload


def test_send_prediction_to_backend_uses_post(monkeypatch):
    captured = {}

    class FakeResponse:
        status_code = 201
        def json(self):
            return {"status": "ok"}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("src.inference.device_client.requests.post", fake_post)

    result = send_prediction_to_backend(
        prediction="NOT_TOMATO_PLANT",
        confidence=0.52,
        device_id="pi-2",
        api_url="http://localhost:8000/api/predictions",
        image_path="/tmp/test.jpg",
    )

    assert captured["url"] == "http://localhost:8000/api/predictions"
    assert captured["json"]["prediction"] == "NOT_TOMATO_PLANT"
    assert result["status"] == "ok"
