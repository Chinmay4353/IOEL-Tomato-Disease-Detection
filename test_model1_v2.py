from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = Path("models/model1_v2_stage2_best.keras")
IMAGE_PATH = Path("captured_images/20260924_174332_851977_Tomato_Test.jpeg")
THRESHOLD = 0.20
IMG_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with Image.open(IMAGE_PATH) as image:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0

x = np.expand_dims(array, axis=0)

probability = float(model.predict(x, verbose=0)[0][0])
prediction = "TOMATO" if probability >= THRESHOLD else "NOT_TOMATO"

print("=" * 60)
print("MODEL 1 V2 LOCAL TEST")
print("=" * 60)
print(f"Model:       {MODEL_PATH}")
print(f"Image:       {IMAGE_PATH}")
print(f"Input shape: {x.shape}")
print(f"Pixel range: {x.min():.4f} - {x.max():.4f}")
print(f"Probability: {probability:.4f} ({probability * 100:.2f}%)")
print(f"Threshold:   {THRESHOLD}")
print(f"Prediction:  {prediction}")
print("=" * 60)
