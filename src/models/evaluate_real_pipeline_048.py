from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

GATE_MODEL = "models/tomato_gate.tflite"
DISEASE_MODEL = "models/tomato_disease_finetuned.tflite"
CLASS_FILE = "models/tomato_disease_classes.json"

GATE_THRESHOLD = 0.48
DISEASE_THRESHOLD = 0.80
IMG_SIZE = (160, 160)

with open(CLASS_FILE, "r", encoding="utf-8") as f:
    classes = json.load(f)

gate = tf.lite.Interpreter(model_path=GATE_MODEL)
disease = tf.lite.Interpreter(model_path=DISEASE_MODEL)

gate.allocate_tensors()
disease.allocate_tensors()

gate_input = gate.get_input_details()[0]
gate_output = gate.get_output_details()[0]

disease_input = disease.get_input_details()[0]
disease_output = disease.get_output_details()[0]

def preprocess(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE)
    return np.expand_dims(
        np.asarray(image, dtype=np.float32),
        axis=0
    )

def predict(path):
    batch = preprocess(path)

    gate.set_tensor(gate_input["index"], batch)
    gate.invoke()
    tomato_probability = float(
        gate.get_tensor(gate_output["index"])[0][0]
    )

    if tomato_probability < GATE_THRESHOLD:
        return tomato_probability, "-", 0.0, "NOT_TOMATO"

    disease.set_tensor(disease_input["index"], batch)
    disease.invoke()

    probabilities = disease.get_tensor(
        disease_output["index"]
    )[0]

    index = int(np.argmax(probabilities))
    confidence = float(probabilities[index])
    prediction = classes[index]

    status = (
        "ACCEPTED"
        if confidence >= DISEASE_THRESHOLD
        else "UNCERTAIN"
    )

    return tomato_probability, prediction, confidence, status

images = sorted(
    [
        p for p in Path("captured_images").iterdir()
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True
)[:10]

print()
print("=" * 105)
print("REAL-WORLD PIPELINE — TEMPORARY GATE THRESHOLD 0.48")
print("=" * 105)
print(
    f"{'Image':55s} "
    f"{'Gate':>8s} "
    f"{'Prediction':35s} "
    f"{'Conf.':>8s} "
    f"{'Status':>12s}"
)
print("-" * 105)

for image_path in images:
    gate_probability, prediction, confidence, status = predict(image_path)

    print(
        f"{image_path.name[:55]:55s} "
        f"{gate_probability * 100:7.2f}% "
        f"{prediction[:35]:35s} "
        f"{confidence * 100:7.2f}% "
        f"{status:>12s}"
    )

print("=" * 105)
