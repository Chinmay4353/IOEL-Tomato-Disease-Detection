"""Camera-based inference application for tomato disease classification."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from src.data.preprocessing import preprocess_path
from src.inference.device_client import (
    send_prediction_to_backend,
)
from src.utils.config import (
    CAPTURED_IMAGES_DIR,
    CAMERA_INDEX,
    CONFIDENCE_THRESHOLD,
    DEVICE_ID,
    MODEL_PATH,
)

logger = logging.getLogger(__name__)


class CameraApp:
    """Camera application for tomato disease detection."""

    def __init__(
        self,
        camera_index: int = CAMERA_INDEX,
        model_path: str | Path = MODEL_PATH,
        threshold: float = CONFIDENCE_THRESHOLD,
        backend_url: str | None = None,
        device_id: str = DEVICE_ID,
    ):
        self.camera_index = camera_index
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.backend_url = backend_url
        self.device_id = device_id

        self.cap = None
        self.last_backend_response = None

    def open_camera(self):
        """Open the camera."""

        self.cap = cv2.VideoCapture(
            self.camera_index
        )

        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera could not be opened. "
                "Check whether the camera is connected."
            )

        return self.cap

    def capture_frame(self):
        """Capture one frame."""

        if self.cap is None:
            raise RuntimeError(
                "Camera is not open."
            )

        ret, frame = self.cap.read()

        if not ret or frame is None:
            raise RuntimeError(
                "Failed to read camera frame."
            )

        return frame

    def save_image(
        self,
        frame,
        output_dir: str | Path = CAPTURED_IMAGES_DIR,
    ) -> Path:
        """Save a captured frame."""

        output_dir = Path(output_dir)
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        filename = (
            output_dir
            / f"capture_{timestamp}.jpg"
        )

        success = cv2.imwrite(
            str(filename),
            frame,
        )

        if not success:
            raise RuntimeError(
                f"Failed to save image: {filename}"
            )

        return filename

    def load_class_names(
        self,
        output_size: int,
    ) -> list[str]:
        """Load class names from class_names.json."""

        class_names_path = (
            self.model_path.with_name(
                "class_names.json"
            )
        )

        if not class_names_path.exists():
            raise FileNotFoundError(
                f"Class names file not found: "
                f"{class_names_path}"
            )

        class_names = json.loads(
            class_names_path.read_text(
                encoding="utf-8"
            )
        )

        if len(class_names) != output_size:
            raise ValueError(
                f"Model has {output_size} outputs, "
                f"but class_names.json contains "
                f"{len(class_names)} classes."
            )

        return class_names

    def run_tflite_inference(
        self,
        image_path: str | Path,
    ) -> dict:
        """Run TFLite disease inference."""

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: "
                f"{self.model_path}"
            )

        interpreter = tf.lite.Interpreter(
            model_path=str(
                self.model_path.resolve()
            )
        )

        interpreter.allocate_tensors()

        input_details = (
            interpreter.get_input_details()[0]
        )

        output_details = (
            interpreter.get_output_details()[0]
        )

        image_array = preprocess_path(
            image_path
        )

        input_data = image_array.astype(
            input_details["dtype"]
        )

        interpreter.set_tensor(
            input_details["index"],
            input_data,
        )

        interpreter.invoke()

        output = np.asarray(
            interpreter.get_tensor(
                output_details["index"]
            )[0]
        ).reshape(-1)

        if output.size <= 1:
            raise ValueError(
                "Expected an 11-class model."
            )

        class_names = self.load_class_names(
            output.size
        )

        predicted_index = int(
            np.argmax(output)
        )

        confidence = float(
            output[predicted_index]
        )

        prediction = class_names[
            predicted_index
        ]

        display_label = (
            "Healthy"
            if prediction.lower() == "healthy"
            else prediction.replace("_", " ")
        )

        confidence_status = (
            "HIGH_CONFIDENCE"
            if confidence >= self.threshold
            else "LOW_CONFIDENCE"
        )

        return {
            "prediction": prediction,
            "label": display_label,
            "confidence": confidence,
            "threshold": self.threshold,
            "confidence_status": confidence_status,
            "disease": (
                None
                if prediction.lower() == "healthy"
                else display_label
            ),
        }

    def capture_and_predict(self):
        """Capture, save, predict, and optionally send to backend."""

        frame = self.capture_frame()

        saved_path = self.save_image(
            frame
        )

        result = self.run_tflite_inference(
            saved_path
        )

        if self.backend_url:
            self.last_backend_response = (
                send_prediction_to_backend(
                    prediction=result["prediction"],
                    confidence=result["confidence"],
                    device_id=self.device_id,
                    api_url=self.backend_url,
                    image_path=str(saved_path),
                )
            )

        return saved_path, result

    def close(self):
        """Release camera resources."""

        if self.cap is not None:
            self.cap.release()
            self.cap = None


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Capture images and predict "
            "tomato diseases."
        )
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=CAMERA_INDEX,
        help="Camera index.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help="Path to the TFLite model.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold.",
    )

    parser.add_argument(
        "--backend-url",
        type=str,
        default=None,
        help=(
            "Backend prediction endpoint, "
            "for example "
            "http://SERVER_IP:8000/api/predictions."
        ),
    )

    parser.add_argument(
        "--device-id",
        type=str,
        default=DEVICE_ID,
        help="Device identifier.",
    )

    args = parser.parse_args()

    app = CameraApp(
        camera_index=args.camera_index,
        model_path=args.model,
        threshold=args.threshold,
        backend_url=args.backend_url,
        device_id=args.device_id,
    )

    try:
        app.open_camera()

        print(
            "Camera opened successfully."
        )
        print(
            "Press C to capture and predict."
        )
        print(
            "Press Q to quit."
        )

        while True:
            frame = app.capture_frame()

            cv2.imshow(
                "Tomato Disease Detector",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("c"):
                saved_path, result = (
                    app.capture_and_predict()
                )

                print(
                    f"\nSaved image: "
                    f"{saved_path}"
                )

                print(
                    f"Prediction: "
                    f"{result['prediction']}"
                )

                print(
                    f"Label: "
                    f"{result['label']}"
                )

                print(
                    f"Confidence: "
                    f"{result['confidence'] * 100:.2f}%"
                )

                print(
                    f"Status: "
                    f"{result['confidence_status']}"
                )

                if app.backend_url:
                    print(
                        "Backend response: "
                        f"{app.last_backend_response}"
                    )

    except Exception as exc:
        logger.error(
            "Camera application error: %s",
            exc,
        )
        print(f"Error: {exc}")

    finally:
        cv2.destroyAllWindows()
        app.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO
    )

    main()