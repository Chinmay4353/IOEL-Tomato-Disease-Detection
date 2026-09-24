# Tomato Plant Detector

This project is being built as a college-level IoT mini-project for tomato plant detection using a camera and edge AI.

## Current Phase

Phase 14: End-to-end integration and demonstration.

The project now supports camera capture, MobileNetV2/TensorFlow Lite inference, device-to-backend reporting, SQLite storage, and a live dashboard.

## Planned Architecture

- Camera input
- Image preprocessing
- MobileNetV2 transfer-learning model
- Binary classification: TOMATO_PLANT / NOT_TOMATO_PLANT
- TensorFlow Lite export for edge deployment
- Raspberry Pi inference app
- Backend API and live dashboard

## Project Structure

- dataset/
- notebooks/
- src/
- model/
- results/
- captured_images/
- backend/
- dashboard/
- tests/
- logs/

## Run the Complete Demo

Start the backend from the project root:

```bash
python backend/app.py
```

Open the dashboard at `http://127.0.0.1:8000/dashboard`.

On a device with a camera, run the camera application and enable reporting:

```bash
python -m src.inference.camera_app --model model/model.tflite --backend-url http://SERVER_IP:8000/api/predictions --device-id raspberry-pi-01
```

Press `c` to capture and classify an image. The result is saved locally, sent to the backend, stored in SQLite, and displayed on the dashboard. Press `q` to exit.

## Phase 14 Validation

Run all automated checks from the project root:

```bash
python -m pytest -q
```
