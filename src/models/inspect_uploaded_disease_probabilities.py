from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

MODEL_PATH = "models/tomato_disease_finetuned.tflite"
CLASS_PATH = "models/tomato_disease_classes.json"
IMAGE_PATH = r".\captured_images\20260925_085016_598359_tomato-test3.jpg"

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

image = Image.open(IMAGE_PATH)
image = ImageOps.exif_transpose(image).convert("RGB")
image = image.resize((160, 160))

array = np.asarray(image, dtype=np.float32)
batch = np.expand_dims(array, axis=0)

interpreter.set_tensor(input_details[0]["index"], batch)
interpreter.invoke()

probabilities = interpreter.get_tensor(
    output_details[0]["index"]
)[0]

ranking = np.argsort(probabilities)[::-1]

print("========== DISEASE MODEL PROBABILITIES ==========")

for index in ranking:
    print(
        f"{classes[index]:45s} "
        f"{probabilities[index] * 100:6.2f}%"
    )

print("==================================================")
