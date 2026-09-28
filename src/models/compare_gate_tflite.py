from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

KERAS_PATH = "models/tomato_gate_best.keras"
TFLITE_PATH = "models/tomato_gate.tflite"
VAL_DIR = Path("dataset/gate_split/validation")

IMG_SIZE = (160, 160)
THRESHOLD = 0.5

print("Loading Keras model...")
keras_model = tf.keras.models.load_model(KERAS_PATH)

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

keras_correct = 0
tflite_correct = 0
same_prediction = 0
total = 0
max_difference = 0.0

for class_name, label in [("not_tomato", 0), ("tomato", 1)]:
    class_dir = VAL_DIR / class_name

    for image_path in sorted(class_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(image, dtype=np.float32)
        batch = np.expand_dims(array, axis=0)

        # Keras prediction
        keras_prob = float(keras_model(batch, training=False).numpy()[0][0])
        keras_pred = 1 if keras_prob >= THRESHOLD else 0

        # TFLite prediction
        interpreter.set_tensor(input_details[0]["index"], batch)
        interpreter.invoke()
        tflite_prob = float(
            interpreter.get_tensor(output_details[0]["index"])[0][0]
        )
        tflite_pred = 1 if tflite_prob >= THRESHOLD else 0

        if keras_pred == label:
            keras_correct += 1

        if tflite_pred == label:
            tflite_correct += 1

        if keras_pred == tflite_pred:
            same_prediction += 1

        difference = abs(keras_prob - tflite_prob)
        max_difference = max(max_difference, difference)

        total += 1

keras_accuracy = keras_correct / total
tflite_accuracy = tflite_correct / total
agreement = same_prediction / total

print()
print("========== KERAS vs TFLITE ==========")
print(f"Total images:        {total}")
print(f"Keras accuracy:      {keras_accuracy:.4f} ({keras_accuracy * 100:.2f}%)")
print(f"TFLite accuracy:     {tflite_accuracy:.4f} ({tflite_accuracy * 100:.2f}%)")
print(f"Prediction agreement:{agreement:.4f} ({agreement * 100:.2f}%)")
print(f"Max probability diff:{max_difference:.6f}")
print("======================================")
