from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

GATE_MODEL = "models/tomato_gate.tflite"
DISEASE_MODEL = "models/tomato_disease_finetuned.tflite"
CLASS_FILE = "models/tomato_disease_classes.json"

IMAGE_DIR = Path("captured_images")

IMG_SIZE = (160, 160)
GATE_THRESHOLD = 0.50
DISEASE_THRESHOLD = 0.80

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_interpreter(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize(IMG_SIZE)

    array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def predict_gate(interpreter, batch):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"],
        batch
    )
    interpreter.invoke()

    return float(
        interpreter.get_tensor(
            output_details[0]["index"]
        )[0][0]
    )


def predict_disease(interpreter, batch, classes):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(
        input_details[0]["index"],
        batch
    )
    interpreter.invoke()

    probabilities = interpreter.get_tensor(
        output_details[0]["index"]
    )[0]

    index = int(np.argmax(probabilities))

    return classes[index], float(probabilities[index])


print("Loading models...")

gate_interpreter = load_interpreter(GATE_MODEL)
disease_interpreter = load_interpreter(DISEASE_MODEL)

with open(CLASS_FILE, "r", encoding="utf-8") as file:
    classes = json.load(file)

images = sorted(
    [
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in VALID_EXTENSIONS
    ],
    key=lambda p: p.stat().st_mtime,
    reverse=True
)[:10]

if not images:
    raise SystemExit(
        f"No images found in {IMAGE_DIR}"
    )

print()
print("=" * 110)
print("TWO-MODEL TOMATO PIPELINE EVALUATION")
print("=" * 110)
print(
    f"{'Image':45s} "
    f"{'Gate':>9s} "
    f"{'Disease':40s} "
    f"{'Conf.':>8s} "
    f"{'Status':>12s}"
)
print("-" * 110)

for image_path in images:
    batch = preprocess(image_path)

    gate_probability = predict_gate(
        gate_interpreter,
        batch
    )

    if gate_probability < GATE_THRESHOLD:
        disease = "-"
        disease_confidence = 0.0
        status = "NOT_TOMATO"

    else:
        disease, disease_confidence = predict_disease(
            disease_interpreter,
            batch,
            classes
        )

        if disease_confidence < DISEASE_THRESHOLD:
            status = "UNCERTAIN"
        else:
            status = "ACCEPTED"

    print(
        f"{image_path.name[:45]:45s} "
        f"{gate_probability * 100:8.2f}% "
        f"{disease[:40]:40s} "
        f"{disease_confidence * 100:7.2f}% "
        f"{status:>12s}"
    )

print("=" * 110)
print(f"Gate threshold:    {GATE_THRESHOLD:.2f}")
print(f"Disease threshold: {DISEASE_THRESHOLD:.2f}")
print(f"Images tested:     {len(images)}")
print("=" * 110)
