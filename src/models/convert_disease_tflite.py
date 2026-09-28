from pathlib import Path
import tensorflow as tf

MODEL_PATH = Path("models/tomato_disease_finetuned_best.keras")
OUTPUT_PATH = Path("models/tomato_disease_finetuned.tflite")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Keep float32 for the first deployment baseline.
tflite_model = converter.convert()

OUTPUT_PATH.write_bytes(tflite_model)

print()
print("Conversion complete.")
print(f"Output: {OUTPUT_PATH.resolve()}")
print(f"Size:   {OUTPUT_PATH.stat().st_size / (1024 * 1024):.2f} MB")
