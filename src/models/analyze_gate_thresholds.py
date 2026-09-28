from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import confusion_matrix

VAL_DIR = Path(r".\dataset\gate_split\validation")
MODEL_PATH = "models/tomato_gate_best.keras"
IMG_SIZE = (160, 160)

model = tf.keras.models.load_model(MODEL_PATH)

y_true = []
probabilities = []

for class_name, label in [("not_tomato", 0), ("tomato", 1)]:
    for image_path in sorted((VAL_DIR / class_name).iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue

        image = Image.open(image_path)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMG_SIZE)

        array = np.asarray(image, dtype=np.float32)
        probability = float(
            model(np.expand_dims(array, axis=0), training=False).numpy()[0][0]
        )

        y_true.append(label)
        probabilities.append(probability)

y_true = np.array(y_true)
probabilities = np.array(probabilities)

print("========== GATE THRESHOLD ANALYSIS ==========")

for threshold in [0.30, 0.35, 0.40, 0.45, 0.50]:
    y_pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tomato_recall = tp / (tp + fn)
    not_tomato_rejection = tn / (tn + fp)
    accuracy = (tp + tn) / len(y_true)

    print(
        f"Threshold {threshold:.2f} | "
        f"Accuracy {accuracy:.4f} | "
        f"Tomato recall {tomato_recall:.4f} | "
        f"Not-tomato rejection {not_tomato_rejection:.4f} | "
        f"FN {fn} | FP {fp}"
    )

print("=============================================")
