"""Flask backend for the tomato disease detector."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import (
    Flask,
    jsonify,
    request,
    send_from_directory,
)
from werkzeug.utils import secure_filename

try:
    from backend.database import (
        get_connection,
        init_db,
    )
except ModuleNotFoundError:
    from database import (
        get_connection,
        init_db,
    )

from src.inference.combined_inference import (
    TomatoDiseasePipeline,
)
from src.utils.config import (
    CAPTURED_IMAGES_DIR,
    MODEL_PATH,
)


DASHBOARD_DIR = PROJECT_ROOT / "dashboard"

app = Flask(
    __name__,
    static_folder=str(
        PROJECT_ROOT / "backend" / "static"
    ),
)


# Load both TFLite models once when the backend starts.
# This avoids reloading the models for every image request.
inference_pipeline = TomatoDiseasePipeline()


ALLOWED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


RECOMMENDATIONS = {
    "Bacterial_spot": (
        "Remove badly affected leaves, avoid overhead watering, "
        "and keep foliage dry with good airflow."
    ),
    "Early_blight": (
        "Remove affected leaves, improve airflow, avoid overhead watering, "
        "and monitor nearby leaves for spreading symptoms."
    ),
    "Late_blight": (
        "Remove affected plant material, improve airflow, avoid prolonged "
        "leaf wetness, and isolate heavily affected plants when practical."
    ),
    "Leaf_Mold": (
        "Improve ventilation and reduce leaf humidity, especially around "
        "dense foliage. Remove severely affected leaves."
    ),
    "powdery_mildew": (
        "Improve airflow and sunlight exposure, remove severely affected "
        "leaves, and avoid excessive humidity around the foliage."
    ),
    "Septoria_leaf_spot": (
        "Remove affected leaves, avoid splashing water onto foliage, "
        "and improve airflow around the plant."
    ),
    "Spider_mites Two-spotted_spider_mite": (
        "Inspect the undersides of leaves, wash foliage where appropriate, "
        "and monitor mite levels and plant stress."
    ),
    "Target_Spot": (
        "Remove severely affected leaves, reduce leaf wetness, "
        "and improve airflow around the plant."
    ),
    "Tomato_mosaic_virus": (
        "Remove and isolate visibly infected plants where practical, "
        "sanitize tools, and avoid handling plants when foliage is wet."
    ),
    "Tomato_Yellow_Leaf_Curl_Virus": (
        "Inspect for whitefly activity, remove severely affected plants "
        "where practical, and manage the insect vector."
    ),
    "healthy": (
        "No disease was detected with sufficient confidence. "
        "Continue regular monitoring and good plant-care practices."
    ),
}


ALLOWED_PREDICTIONS = {
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew",
}



def get_prediction_status(prediction, confidence):
    if prediction == "NOT_TOMATO":
        return "NOT_TOMATO"

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return "UNCERTAIN"

    if confidence_value < 0.80:
        return "UNCERTAIN"

    return "ACCEPTED"

def validate_prediction_payload(
    prediction,
    confidence,
    device_id,
):
    """Validate a device prediction request."""

    if (
        not prediction
        or confidence is None
        or not device_id
    ):
        return (
            "prediction, confidence, and "
            "device_id are required."
        )

    if prediction not in ALLOWED_PREDICTIONS:
        return (
            f"Invalid prediction label: "
            f"{prediction}"
        )

    try:
        confidence_value = float(
            confidence
        )
    except (
        TypeError,
        ValueError,
    ):
        return (
            "confidence must be a numeric value."
        )

    if not 0.0 <= confidence_value <= 1.0:
        return (
            "confidence must be between 0 and 1."
        )

    return None


@app.get("/")
@app.get("/dashboard")
def dashboard_index():
    """Serve the dashboard."""

    return send_from_directory(
        str(DASHBOARD_DIR),
        "index.html",
    )


@app.get("/dashboard/<path:filename>")
def dashboard_static(filename: str):
    """Serve dashboard static files."""

    return send_from_directory(
        str(DASHBOARD_DIR),
        filename,
    )


@app.get("/captured-images/<path:filename>")
def captured_image(filename: str):
    """Serve captured images."""

    return send_from_directory(
        str(CAPTURED_IMAGES_DIR),
        filename,
    )


@app.before_request
def ensure_db() -> None:
    """Ensure database tables exist."""

    init_db()


@app.get("/api/health")
def health():
    """Health-check endpoint."""

    return jsonify(
        {
            "status": "ok",
            "message": (
                "Tomato disease detector "
                "backend is running."
            ),
        }
    ), 200


@app.post("/api/predictions")
def create_prediction():
    """
    Receive a prediction generated by a device.
    """

    payload = (
        request.get_json(
            silent=True
        )
        or {}
    )

    prediction = payload.get(
        "prediction"
    )

    confidence = payload.get(
        "confidence"
    )

    device_id = payload.get(
        "device_id"
    )

    image_path = payload.get(
        "image_path"
    )

    timestamp = (
        payload.get("timestamp")
        or datetime.utcnow().isoformat()
    )

    validation_error = (
        validate_prediction_payload(
            prediction,
            confidence,
            device_id,
        )
    )

    if validation_error:
        return jsonify(
            {"error": validation_error}
        ), 400

    confidence_value = float(
        confidence
    )

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO predictions
            (
                prediction,
                confidence,
                device_id,
                timestamp,
                image_path
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction,
                confidence_value,
                device_id,
                timestamp,
                image_path,
            ),
        )

        connection.commit()

        prediction_id = cursor.lastrowid

    finally:
        connection.close()

    return jsonify(
        {
            "id": prediction_id,
            "prediction": prediction,
            "confidence": confidence_value,
            "device_id": device_id,
            "timestamp": timestamp,
            "image_path": image_path,
        }
    ), 201


@app.post("/api/predict-image")
def predict_uploaded_image():
    """
    Receive an image, run TFLite inference,
    and store the prediction.
    """

    image = request.files.get(
        "image"
    )

    if image is None or not image.filename:
        return jsonify(
            {
                "error": (
                    "An image file is required."
                )
            }
        ), 400

    filename = secure_filename(
        image.filename
    )

    if not filename:
        return jsonify(
            {
                "error": (
                    "The image filename "
                    "is invalid."
                )
            }
        ), 400

    extension = Path(
        filename
    ).suffix.lower()

    if extension not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify(
            {
                "error": (
                    "Please upload a JPG, PNG, "
                    "WebP, or BMP image."
                )
            }
        ), 400

    CAPTURED_IMAGES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    saved_path = (
        CAPTURED_IMAGES_DIR
        / (
            f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S_%f')}"
            f"_{filename}"
        )
    )

    image.save(saved_path)

    try:
        result = inference_pipeline.predict(
            saved_path
        )

    except (
        FileNotFoundError,
        ValueError,
        OSError,
    ) as exc:

        saved_path.unlink(
            missing_ok=True
        )

        return jsonify(
            {"error": str(exc)}
        ), 400

    device_id = request.form.get(
        "device_id",
        "dashboard-camera",
    )

    timestamp = (
        datetime.utcnow().isoformat()
    )

    status = result["status"]

    if status == "NOT_TOMATO":
        prediction = "NOT_TOMATO"
        confidence = result["tomato_probability"]
        label = "Not a tomato leaf/plant image"
        disease = None
        recommendation = (
            "Capture a clear image of a tomato leaf or tomato plant."
        )
        confidence_status = "NOT_TOMATO"

    elif status == "UNCERTAIN":
        prediction = result["prediction"]
        confidence = result["confidence"]
        label = "Uncertain"
        disease = None
        recommendation = (
            "Capture another clear image of the affected leaf "
            "with good lighting and the leaf filling most of the frame."
        )
        confidence_status = "UNCERTAIN"

    else:
        prediction = result["prediction"]
        confidence = result["confidence"]
        label = prediction
        disease = (
            None
            if prediction == "healthy"
            else prediction
        )
        recommendation = RECOMMENDATIONS.get(
            prediction,
            "Continue monitoring the plant and capture another clear image "
            "if symptoms change."
        )
        confidence_status = "ACCEPTED"

    connection = get_connection()

    try:
        cursor = connection.execute(
            """
            INSERT INTO predictions
            (
                prediction,
                confidence,
                device_id,
                timestamp,
                image_path
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                prediction,
                confidence,
                device_id,
                timestamp,
                saved_path.name,
            ),
        )

        connection.commit()

        prediction_id = cursor.lastrowid

    finally:
        connection.close()

    return jsonify(
        {
            "success": True,
            "id": prediction_id,
            "status": status,
            "is_tomato": result["is_tomato"],
            "tomato_probability": result[
                "tomato_probability"
            ],
            "prediction": prediction,
            "label": label,
            "disease": disease,
            "confidence": confidence,
            "confidence_percent": round(
                confidence * 100,
                2,
            ),
            "confidence_status": confidence_status,
            "threshold": (
                0.80
                if result["is_tomato"]
                else 0.50
            ),
            "recommendation": recommendation,
            "device_id": device_id,
            "timestamp": timestamp,
            "image_path": saved_path.name,
            "image_url": (
                f"/captured-images/"
                f"{saved_path.name}"
            ),
        }
    ), 201


@app.get("/api/predictions")
def list_predictions():
    """Return prediction history."""

    connection = get_connection()

    try:
        rows = connection.execute(
            """
            SELECT
                id,
                prediction,
                confidence,
                device_id,
                timestamp,
                image_path
            FROM predictions
            ORDER BY id DESC
            """
        ).fetchall()

    finally:
        connection.close()

    return jsonify(
        [
            {
                "id": row["id"],
                "prediction": row[
                    "prediction"
                ],
                "confidence": row[
                    "confidence"
                ],
                "status": get_prediction_status(
                    row["prediction"],
                    row["confidence"],
                ),
                "device_id": row[
                    "device_id"
                ],
                "timestamp": row[
                    "timestamp"
                ],
                "image_path": row[
                    "image_path"
                ],
            }
            for row in rows
        ]
    ), 200


@app.get("/api/latest")
def latest_prediction():
    """Return the most recent prediction."""

    connection = get_connection()

    try:
        row = connection.execute(
            """
            SELECT
                id,
                prediction,
                confidence,
                device_id,
                timestamp,
                image_path
            FROM predictions
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()

    finally:
        connection.close()

    if row is None:
        return jsonify(
            {
                "message": (
                    "No predictions "
                    "available yet."
                )
            }
        ), 404

    return jsonify(
        {
            "id": row["id"],
            "prediction": row[
                "prediction"
            ],
            "confidence": row[
                "confidence"
            ],
            "device_id": row[
                "device_id"
            ],
            "timestamp": row[
                "timestamp"
            ],
            "image_path": row[
                "image_path"
            ],
            "status": get_prediction_status(
                row["prediction"],
                row["confidence"],
            ),
        }
    ), 200


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8000,
        debug=True,
    )

