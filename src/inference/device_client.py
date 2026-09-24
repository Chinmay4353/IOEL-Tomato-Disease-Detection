"""Device-side client for sending predictions to the backend API."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests


def build_prediction_payload(
    prediction: str,
    confidence: float,
    device_id: str,
    image_path: str | None = None,
) -> dict[str, Any]:
    """
    Build a prediction payload compatible with
    the Flask backend.
    """

    payload: dict[str, Any] = {
        "prediction": prediction,
        "confidence": float(confidence),
        "device_id": device_id,
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if image_path is not None:
        payload["image_path"] = image_path

    return payload


def send_prediction_to_backend(
    prediction: str,
    confidence: float,
    device_id: str,
    api_url: str,
    image_path: str | None = None,
    timeout: int = 10,
) -> dict[str, Any]:
    """
    Send a prediction to the Flask backend.
    """

    payload = build_prediction_payload(
        prediction=prediction,
        confidence=confidence,
        device_id=device_id,
        image_path=image_path,
    )

    endpoint = api_url.rstrip("/")

    try:
        response = requests.post(
            endpoint,
            json=payload,
            timeout=timeout,
        )

        response.raise_for_status()

    except requests.RequestException as exc:
        return {
            "status": "error",
            "message": str(exc),
        }

    try:
        return response.json()

    except ValueError:
        return {
            "status": "error",
            "message": response.text,
        }