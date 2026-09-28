from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

# ============================================================
# Evaluate Fine-Tuned Disease Model
# ============================================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

VALID_DIR = Path("dataset/tomato/valid")

MODEL_PATH = Path(
    "models/tomato_disease_finetuned_best.keras"
)

CLASS_NAMES_PATH = Path(
    "models/tomato_disease_classes.json"
)

# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

class_names = json.loads(
    CLASS_NAMES_PATH.read_text(encoding="utf-8")
)

# ------------------------------------------------------------
# Load validation dataset
# ------------------------------------------------------------

print("Loading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

valid_ds = valid_ds.prefetch(tf.data.AUTOTUNE)

# ------------------------------------------------------------
# Load fine-tuned model
# ------------------------------------------------------------

print("\nLoading fine-tuned model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ------------------------------------------------------------
# Keras evaluation
# ------------------------------------------------------------

print("\nKeras evaluation:")

results = model.evaluate(
    valid_ds,
    verbose=1,
    return_dict=True,
)

for metric_name, value in results.items():
    print(
        f"{metric_name}: {value:.6f}"
    )

# ------------------------------------------------------------
# Generate predictions
# ------------------------------------------------------------

print("\nGenerating predictions...")

y_true = []
y_pred = []

for images, labels in valid_ds:
    probabilities = model.predict(
        images,
        verbose=0,
    )

    predictions = np.argmax(
        probabilities,
        axis=1,
    )

    y_true.extend(labels.numpy())
    y_pred.extend(predictions)

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# ------------------------------------------------------------
# Classification report
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("FINE-TUNED DISEASE MODEL VALIDATION REPORT")
print("=" * 90)

print(
    f"\nTotal validation images: {len(y_true)}"
)

accuracy = np.mean(
    y_true == y_pred
)

print(
    f"Accuracy: {accuracy:.6f} "
    f"({accuracy * 100:.2f}%)"
)

print("\nPer-class metrics:")

print(
    classification_report(
        y_true,
        y_pred,
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
)

print("\nConfusion Matrix:")
print("Rows = Actual")
print("Columns = Predicted\n")

print(
    "Class order:"
)

for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print("\nMatrix:")

for index, row in enumerate(cm):
    print(
        f"{index:2d} "
        f"{class_names[index]:45s} "
        + " ".join(
            f"{value:5d}"
            for value in row
        )
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
                    actual,
                    predicted,
                )
            )

errors.sort(
    reverse=True
)

for count, actual, predicted in errors[:20]:
    print(
        f"  {count:3d}  "
        f"{class_names[actual]} -> "
        f"{class_names[predicted]}"
    )

print("\n" + "=" * 90)
