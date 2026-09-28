from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path("models/tomato_gate_best.keras")
OUTPUT_PATH = Path("models/tomato_gate.tflite")

print("Loading Keras model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TensorFlow Lite...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

OUTPUT_PATH.write_bytes(tflite_model)

print()
print("TFLite conversion complete.")
print(f"Output: {OUTPUT_PATH}")
print(f"Size: {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")
