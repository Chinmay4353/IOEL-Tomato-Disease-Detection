from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/model1_v2_stage2_best.keras"
IMAGE_DIR = Path("captured_images")
THRESHOLD = 0.20

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

images = [
    "20260924_174332_851977_Tomato_Test.jpeg",
    "20260924_175739_168589_Tomato_Test2.jpeg",
    "20260924_181215_514359_tomato-test3.jpg",
]

print("=" * 65)
print("MODEL 1 V2 — THREE IMAGE LOCAL TEST")
print("=" * 65)

for name in images:
    path = IMAGE_DIR / name

    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((160, 160))
        x = np.asarray(image, dtype=np.float32) / 255.0

    x = np.expand_dims(x, axis=0)

    probability = float(model.predict(x, verbose=0)[0][0])
    prediction = "TOMATO" if probability >= THRESHOLD else "NOT_TOMATO"

    print(f"{name}")
    print(f"  Probability: {probability:.4f} ({probability * 100:.2f}%)")
    print(f"  Threshold:   {THRESHOLD}")
    print(f"  Prediction:  {prediction}")
    print("-" * 65)
