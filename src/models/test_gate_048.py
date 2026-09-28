from pathlib import Path
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

MODEL_PATH = Path(r".\models\tomato_gate_best.keras")
IMAGE_PATH = Path(r".\captured_images\20260925_102128_641390_Tomato_Test4.jpg")

IMG_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH)

image = Image.open(IMAGE_PATH)
image = ImageOps.exif_transpose(image).convert("RGB")
image = image.resize(IMG_SIZE)

array = np.asarray(image, dtype=np.float32)
array = np.expand_dims(array, axis=0)

tomato_probability = float(model.predict(array, verbose=0)[0][0])

print(f"Image: {IMAGE_PATH.name}")
print(f"Tomato probability: {tomato_probability:.4f}")
print(f"At 0.50: {'TOMATO' if tomato_probability >= 0.50 else 'NOT_TOMATO'}")
print(f"At 0.48: {'TOMATO' if tomato_probability >= 0.48 else 'NOT_TOMATO'}")
