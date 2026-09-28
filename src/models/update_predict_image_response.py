from pathlib import Path

path = Path(r".\backend\app.py")
text = path.read_text(encoding="utf-8")

old = '''    connection = get_connection()

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
                result["prediction"],
                result["confidence"],
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
            "id": prediction_id,
            "prediction": result[
                "prediction"
            ],
            "label": result["label"],
            "disease": result["disease"],
            "confidence": result[
                "confidence"
            ],
            "confidence_status": result[
                "confidence_status"
            ],
            "threshold": result[
                "threshold"
            ],
            "device_id": device_id,
            "timestamp": timestamp,
            "image_path": saved_path.name,
            "image_url": (
                f"/captured-images/"
                f"{saved_path.name}"
            ),
        }
    ), 201
'''

new = '''    status = result["status"]

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
'''

if old not in text:
    raise SystemExit(
        "Expected old database/response block was not found. "
        "No changes made."
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("API response updated for the two-model pipeline.")
print("NOT_TOMATO, UNCERTAIN, and ACCEPTED statuses are supported.")
print("Recommendation is included in the API response.")
