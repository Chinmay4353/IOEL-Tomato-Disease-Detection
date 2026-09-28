from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

MODEL_PATH = "models/tomato_disease_best.keras"
VALID_DIR = Path("dataset/tomato/valid")
CLASS_NAMES_PATH = Path("models/tomato_disease_classes.json")

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

print("Loading best disease model...")
model = tf.keras.models.load_model(MODEL_PATH)

class_names = json.loads(
    CLASS_NAMES_PATH.read_text(encoding="utf-8")
)

print("\nClass order:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print("\nLoading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

print("\nEvaluating model...")
results = model.evaluate(
    valid_ds,
    verbose=1,
    return_dict=True,
)

print("\nKeras evaluation:")
for name, value in results.items():
    print(f"{name}: {value:.6f}")

# ------------------------------------------------------------
# Collect predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in valid_ds:
    probabilities = model(images, training=False).numpy()
    predictions = np.argmax(probabilities, axis=1)

    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ------------------------------------------------------------
# Overall accuracy
# ------------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print("\n" + "=" * 90)
print("DISEASE MODEL VALIDATION REPORT")
print("=" * 90)

print(f"\nTotal validation images: {len(y_true)}")
print(f"Accuracy: {accuracy:.6f} ({accuracy * 100:.2f}%)")

# ------------------------------------------------------------
# Per-class metrics
# ------------------------------------------------------------

print("\nPer-class metrics:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)

# ------------------------------------------------------------
# Confusion matrix
# ------------------------------------------------------------

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(len(class_names)),
)

print("\nConfusion Matrix:")
print("Rows = Actual")
print("Columns = Predicted\n")

print(" " * 45 + " ".join(f"{i:>5}" for i in range(len(class_names))))

for i, row in enumerate(cm):
    print(
        f"{i:<3} {class_names[i]:<40}"
        + " ".join(f"{value:>5}" for value in row)
    )

# ------------------------------------------------------------
# Most common misclassifications
# ------------------------------------------------------------

print("\nMost common misclassifications:")

errors = []

for actual in range(len(class_names)):
    for predicted in range(len(class_names)):
        if actual != predicted and cm[actual, predicted] > 0:
            errors.append(
                (
                    cm[actual, predicted],
                    class_names[actual],
                    class_names[predicted],
                )
            )

errors.sort(reverse=True)

for count, actual, predicted in errors[:15]:
    print(
        f"{count:>4}  {actual} -> {predicted}"
    )

print("\n" + "=" * 90)
