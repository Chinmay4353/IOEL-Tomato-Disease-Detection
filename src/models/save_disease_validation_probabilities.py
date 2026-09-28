from pathlib import Path
import json
import numpy as np
import tensorflow as tf

# ============================================================
# Generate validation probabilities for threshold analysis
# Fine-tuned Disease Model
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

OUTPUT_DIR = Path("results")
OUTPUT_PATH = (
    OUTPUT_DIR
    / "disease_finetuned_validation_probabilities.npz"
)

# ------------------------------------------------------------
# Create output directory
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

class_names = json.loads(
    CLASS_NAMES_PATH.read_text(
        encoding="utf-8"
    )
)

print("Classes:")

for index, name in enumerate(class_names):
    print(f"{index}: {name}")

# ------------------------------------------------------------
# Load validation dataset
# ------------------------------------------------------------

print("\nLoading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

valid_ds = valid_ds.prefetch(
    tf.data.AUTOTUNE
)

# ------------------------------------------------------------
# Load fine-tuned model
# ------------------------------------------------------------

print("\nLoading model:")
print(MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)

# ------------------------------------------------------------
# Generate probabilities
# ------------------------------------------------------------

print("\nGenerating validation probabilities...")

all_probabilities = []
all_labels = []

for batch_index, (images, labels) in enumerate(
    valid_ds,
    start=1,
):
    probabilities = model.predict(
        images,
        verbose=0,
    )

    all_probabilities.append(
        probabilities
    )

    all_labels.append(
        labels.numpy()
    )

    if batch_index % 25 == 0:
        print(
            f"Processed batches: {batch_index}"
        )

probabilities = np.concatenate(
    all_probabilities,
    axis=0,
)

y_true = np.concatenate(
    all_labels,
    axis=0,
)

# ------------------------------------------------------------
# Basic validation
# ------------------------------------------------------------

if probabilities.shape[0] != y_true.shape[0]:
    raise RuntimeError(
        "Probability and label counts do not match."
    )

if probabilities.shape[1] != len(class_names):
    raise RuntimeError(
        "Probability class count does not match "
        "class names."
    )

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

np.savez_compressed(
    OUTPUT_PATH,
    probabilities=probabilities.astype(
        np.float32
    ),
    y_true=y_true.astype(
        np.int64
    ),
    class_names=np.array(
        class_names
    ),
)

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("PROBABILITY GENERATION COMPLETE")
print("=" * 70)

print(
    f"Validation samples: {len(y_true)}"
)

print(
    f"Probability shape: {probabilities.shape}"
)

print(
    f"Probability range: "
    f"{probabilities.min():.6f} - "
    f"{probabilities.max():.6f}"
)

print(
    f"Output file: {OUTPUT_PATH}"
)

print(
    f"Output size: "
    f"{OUTPUT_PATH.stat().st_size:,} bytes"
)

print("\nSaved arrays:")

print(
    "  probabilities : "
    f"{probabilities.shape}"
)

print(
    "  y_true        : "
    f"{y_true.shape}"
)

print(
    "  class_names   : "
    f"{len(class_names)} classes"
)
