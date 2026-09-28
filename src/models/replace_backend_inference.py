from pathlib import Path

path = Path(r".\backend\app.py")
text = path.read_text(encoding="utf-8")

old = '''    try:
        result = predict_tflite_image(
            model_path=MODEL_PATH,
            image_path=saved_path,
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
'''

new = '''    try:
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
'''

if old not in text:
    raise SystemExit(
        "Old inference block was not found. No changes made."
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("Old single-model inference call replaced.")
print("Backend now calls inference_pipeline.predict().")
