from pathlib import Path

path = Path(r".\backend\app.py")

text = path.read_text(encoding="utf-8")

old = '''from src.inference.tflite_predict import (
    predict_tflite_image,
)'''

new = '''from src.inference.combined_inference import (
    TomatoDiseasePipeline,
)'''

if old not in text:
    raise SystemExit(
        "Expected old inference import was not found. "
        "No changes made."
    )

text = text.replace(old, new)

marker = '''app = Flask(
    __name__,
    static_folder=str(
        PROJECT_ROOT / "backend" / "static"
    ),
)
'''

replacement = marker + '''

# Load both TFLite models once when the backend starts.
# This avoids reloading the models for every image request.
inference_pipeline = TomatoDiseasePipeline()
'''

if marker not in text:
    raise SystemExit(
        "Expected Flask app initialization block was not found. "
        "No changes made."
    )

text = text.replace(marker, replacement, 1)

path.write_text(text, encoding="utf-8")

print("Backend inference import updated.")
print("TomatoDiseasePipeline initialization added.")

