from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# Model 2 - Stage 2 Fine-Tuning
# ============================================================

SEED = 42
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 10

TRAIN_DIR = Path("dataset/tomato/train")
VALID_DIR = Path("dataset/tomato/valid")

BASE_MODEL_PATH = Path("models/tomato_disease_best.keras")

BEST_MODEL_PATH = Path(
    "models/tomato_disease_finetuned_best.keras"
)
FINAL_MODEL_PATH = Path(
    "models/tomato_disease_finetuned_final.keras"
)

CLASS_NAMES_PATH = Path(
    "models/tomato_disease_classes.json"
)

tf.keras.utils.set_random_seed(SEED)

# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

class_names = json.loads(
    CLASS_NAMES_PATH.read_text(encoding="utf-8")
)

num_classes = len(class_names)

if num_classes != 11:
    raise ValueError(
        f"Expected 11 classes, found {num_classes}"
    )

print("Class order:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

# ------------------------------------------------------------
# Load datasets
# ------------------------------------------------------------

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

print("\nLoading validation dataset...")

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)

# ------------------------------------------------------------
# Calculate class weights
# ------------------------------------------------------------

class_counts = np.zeros(
    num_classes,
    dtype=np.int64,
)

for _, labels in train_ds:
    labels = labels.numpy()

    values, counts = np.unique(
        labels,
        return_counts=True,
    )

    for value, count in zip(values, counts):
        class_counts[value] += count

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.repeat(
        np.arange(num_classes),
        class_counts,
    ),
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(
        class_weights_array
    )
}

print("\nClass weights:")

for index, name in enumerate(class_names):
    print(
        f"{index}: {name:<45} "
        f"{class_weights[index]:.4f}"
    )

# ------------------------------------------------------------
# Load existing BEST baseline model
# ------------------------------------------------------------

print("\nLoading existing best checkpoint:")
print(BASE_MODEL_PATH)

model = tf.keras.models.load_model(
    BASE_MODEL_PATH
)

print("\nLoaded model:")
print(model.name)

# ------------------------------------------------------------
# Find MobileNetV3Small backbone
# ------------------------------------------------------------

base_model = None

for layer in model.layers:
    if isinstance(
        layer,
        tf.keras.Model,
    ):
        if "mobilenet" in layer.name.lower():
            base_model = layer
            break

if base_model is None:
    raise RuntimeError(
        "MobileNetV3Small backbone was not found."
    )

print("\nBackbone:")
print(base_model.name)

print(
    f"Total backbone layers: "
    f"{len(base_model.layers)}"
)

# ------------------------------------------------------------
# Freeze everything first
# ------------------------------------------------------------

base_model.trainable = True

for layer in base_model.layers:
    layer.trainable = False

# ------------------------------------------------------------
# Unfreeze only upper backbone layers
#
# Keep BatchNormalization frozen because this is a
# small fine-tuning dataset relative to ImageNet.
# ------------------------------------------------------------

FINE_TUNE_LAST_N = 30

start_index = max(
    0,
    len(base_model.layers) - FINE_TUNE_LAST_N,
)

for layer in base_model.layers[start_index:]:
    if not isinstance(
        layer,
        tf.keras.layers.BatchNormalization,
    ):
        layer.trainable = True

print(
    f"\nUnfreezing last "
    f"{FINE_TUNE_LAST_N} backbone layers."
)

trainable_backbone = [
    layer.name
    for layer in base_model.layers
    if layer.trainable
]

print(
    f"Trainable backbone layers: "
    f"{len(trainable_backbone)}"
)

for name in trainable_backbone:
    print(f"  {name}")

# ------------------------------------------------------------
# Recompile with LOW learning rate
# ------------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="accuracy"
        ),
        tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=3,
            name="top3_accuracy",
        ),
    ],
)

print("\nTrainable parameters:")

trainable_parameters = sum(
    np.prod(variable.shape)
    for variable in model.trainable_variables
)

print(trainable_parameters)

# ------------------------------------------------------------
# Callbacks
# ------------------------------------------------------------

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(BEST_MODEL_PATH),
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=3,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-7,
    verbose=1,
)

# ------------------------------------------------------------
# Fine-tuning
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING STAGE 2 FINE-TUNING")
print("=" * 70)

print(
    "\nThe original best checkpoint will NOT be overwritten."
)

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    class_weight=class_weights,
    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
    ],
)

# ------------------------------------------------------------
# Save final fine-tuned model
# ------------------------------------------------------------

model.save(FINAL_MODEL_PATH)

print("\n" + "=" * 70)
print("FINE-TUNING COMPLETE")
print("=" * 70)

print(f"Fine-tuned best:  {BEST_MODEL_PATH}")
print(f"Fine-tuned final: {FINAL_MODEL_PATH}")

# ------------------------------------------------------------
# Final validation evaluation
# ------------------------------------------------------------

print("\nFinal fine-tuned validation metrics:")

results = model.evaluate(
    valid_ds,
    verbose=1,
    return_dict=True,
)

for metric_name, value in results.items():
    print(
        f"{metric_name}: {value:.6f}"
    )
