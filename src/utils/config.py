"""Central configuration for the tomato plant detector project."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.70
KERAS_MODEL_PATH = PROJECT_ROOT / "model" / "tomato_detector.keras"
MODEL_PATH = PROJECT_ROOT / "model" / "model.tflite"
CAMERA_INDEX = 0
INFERENCE_INTERVAL = 0.5
DEVICE_ID = "raspberry-pi-01"
API_URL = "http://SERVER_IP:8000"

DATASET_DIR = PROJECT_ROOT / "dataset"
TOMATO_DIR = DATASET_DIR / "tomato"
NOT_TOMATO_DIR = DATASET_DIR / "not_tomato"
LOGS_DIR = PROJECT_ROOT / "logs"
CAPTURED_IMAGES_DIR = PROJECT_ROOT / "captured_images"
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_DIR = PROJECT_ROOT / "model"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
BACKEND_DIR = PROJECT_ROOT / "backend"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
