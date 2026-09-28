import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

KERAS_PATH = Path("models/tomato_disease_finetuned_best.keras")
TFLITE_PATH = Path("models/tomato_disease_finetuned.tflite")
CLASS_PATH = Path("models/tomato_disease_classes.json")

DATASET = Path("dataset/tomato/valid")

IMG_SIZE = (160, 160)
SAMPLES_PER_CLASS = 3


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)
        return np.asarray(image, dtype=np.float32)


classes = json.loads(CLASS_PATH.read_text(encoding="utf-8"))

print("Loading Keras model...")
keras_model = tf.keras.models.load_model(KERAS_PATH)

print("Loading TFLite model...")
interpreter = tf.lite.Interpreter(model_path=str(TFLITE_PATH))
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nTFLite input:")
print(input_details[0]["shape"], input_details[0]["dtype"])

print("TFLite output:")
print(output_details[0]["shape"], output_details[0]["dtype"])

paths = []

for class_name in classes:
    folder = DATASET / class_name

    if not folder.exists():
        raise FileNotFoundError(f"Missing folder: {folder}")

    class_paths = sorted(
        p for p in folder.iterdir()
        if p.is_file()
    )[:SAMPLES_PER_CLASS]

    paths.extend(class_paths)

images = np.asarray(
    [load_image(path) for path in paths],
    dtype=np.float32,
)

keras_probs = keras_model.predict(
    images,
    batch_size=16,
    verbose=0,
)

tflite_probs = []

for image in images:
    input_data = np.expand_dims(image, axis=0).astype(np.float32)

    interpreter.set_tensor(
        input_details[0]["index"],
        input_data,
    )

    interpreter.invoke()

    output = interpreter.get_tensor(
        output_details[0]["index"]
    )

    tflite_probs.append(output[0])

tflite_probs = np.asarray(tflite_probs)

keras_pred = np.argmax(keras_probs, axis=1)
tflite_pred = np.argmax(tflite_probs, axis=1)

agreement = np.mean(keras_pred == tflite_pred)
max_difference = np.max(
    np.abs(keras_probs - tflite_probs)
)

print("\n" + "=" * 60)
print("KERAS vs TFLITE")
print("=" * 60)

print(f"Images tested:       {len(images)}")
print(f"Prediction agreement: {agreement * 100:.2f}%")
print(f"Max probability diff: {max_difference:.8f}")

print("\nSample predictions:")

for i, path in enumerate(paths[:15]):
    k = int(keras_pred[i])
    t = int(tflite_pred[i])

    print(
        f"{path.parent.name:<35} "
        f"Keras={classes[k]:<35} "
        f"TFLite={classes[t]}"
    )
