from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix, classification_report

VAL_DIR = Path(r".\dataset\gate_split\validation")
MODEL_PATH = "models/tomato_gate_best.keras"
IMG_SIZE = (160, 160)
THRESHOLD = 0.50

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
y_pred = []

for class_name, label in [("not_tomato", 0), ("tomato", 1)]:
    for image_path in sorted((VAL_DIR / class_name).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(image, dtype=np.float32)
        probability = float(model(np.expand_dims(array, axis=0), training=False).numpy()[0][0])
        prediction = 1 if probability >= THRESHOLD else 0

        y_true.append(label)
        y_pred.append(prediction)

print("========== GATE CONFUSION MATRIX ==========")
print(confusion_matrix(y_true, y_pred))
print()
print(classification_report(
    y_true,
    y_pred,
    target_names=["not_tomato", "tomato"],
    digits=4
))
