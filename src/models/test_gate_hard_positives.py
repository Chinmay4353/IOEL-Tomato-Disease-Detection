import tensorflow as tf
import numpy as np
from PIL import Image

MODEL = r".\models\tomato_gate_best.keras"
THRESHOLD = 0.48

images = [
    r".\dataset\gate\tomato\real_hard_positive_FN001.jpg",
    r".\dataset\gate\tomato\real_hard_positive_FN002.jpg",
]

model = tf.keras.models.load_model(MODEL)

for path in images:
    image = Image.open(path).convert("RGB").resize((160, 160))
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    probability = float(model.predict(array, verbose=0)[0][0])

    print("=" * 55)
    print("Image:", path)
    print(f"Tomato probability: {probability:.4f}")
    print(f"At threshold {THRESHOLD}: "
          f"{'TOMATO' if probability >= THRESHOLD else 'NOT_TOMATO'}")
