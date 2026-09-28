import json
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

IMAGE_PATH = r".\captured_images\20260925_085016_598359_tomato-test3.jpg"
KERAS_PATH = "models/tomato_disease_finetuned_best.keras"
TFLITE_PATH = "models/tomato_disease_finetuned.tflite"
CLASS_PATH = "models/tomato_disease_classes.json"

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    classes = json.load(f)

image = Image.open(IMAGE_PATH)
image = ImageOps.exif_transpose(image).convert("RGB")
image = image.resize((160, 160))
batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)

keras_model = tf.keras.models.load_model(KERAS_PATH)
keras_probs = keras_model(batch, training=False).numpy()[0]

interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], batch)
interpreter.invoke()
tflite_probs = interpreter.get_tensor(output_details[0]["index"])[0]

keras_top = int(np.argmax(keras_probs))
tflite_top = int(np.argmax(tflite_probs))

print("========== EXACT IMAGE COMPARISON ==========")
print(f"Keras:  {classes[keras_top]} - {keras_probs[keras_top] * 100:.2f}%")
print(f"TFLite: {classes[tflite_top]} - {tflite_probs[tflite_top] * 100:.2f}%")
print(f"Top prediction agreement: {keras_top == tflite_top}")
print(f"Maximum probability difference: {np.max(np.abs(keras_probs - tflite_probs)):.8f}")
print("============================================")
