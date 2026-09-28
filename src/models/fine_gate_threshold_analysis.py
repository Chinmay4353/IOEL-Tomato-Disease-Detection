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

        batch = np.expand_dims(
            np.asarray(image, dtype=np.float32),
            axis=0
        )

        probability = float(
            model(batch, training=False).numpy()[0][0]
        )

        y_true.append(label)
        probabilities.append(probability)

y_true = np.array(y_true)
probabilities = np.array(probabilities)

print("========== FINE GATE THRESHOLD ANALYSIS ==========")
print(
    f"{'Threshold':>9} "
    f"{'Accuracy':>10} "
    f"{'Tomato Recall':>14} "
    f"{'Not-Tomato Rej.':>16} "
    f"{'FP':>5} "
    f"{'FN':>5}"
)
print("-" * 72)

for threshold in np.arange(0.30, 0.501, 0.02):
    threshold = round(float(threshold), 2)

    y_pred = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    accuracy = (tp + tn) / len(y_true)
    tomato_recall = tp / (tp + fn)
    not_tomato_rejection = tn / (tn + fp)

    print(
        f"{threshold:9.2f} "
        f"{accuracy:10.4f} "
        f"{tomato_recall:14.4f} "
        f"{not_tomato_rejection:16.4f} "
        f"{fp:5d} "
        f"{fn:5d}"
    )

print("=" * 72)
