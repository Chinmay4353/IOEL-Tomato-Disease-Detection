from pathlib import Path
import json
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

# ============================================================
# Model 2 — Tomato Disease Classifier
# ============================================================

SEED = 42
IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_DIR = Path("dataset/tomato/train")
VALID_DIR = Path("dataset/tomato/valid")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "tomato_disease_best.keras"
FINAL_MODEL_PATH = MODEL_DIR / "tomato_disease_final.keras"
CLASS_NAMES_PATH = MODEL_DIR / "tomato_disease_classes.json"

tf.keras.utils.set_random_seed(SEED)

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

valid_ds = tf.keras.utils.image_dataset_from_directory(
    VALID_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)

print()
print("Class names:")
for index, name in enumerate(class_names):
    print(f"{index}: {name}")

print(f"\nNumber of classes: {num_classes}")

if num_classes != 11:
    raise ValueError(
        f"Expected 11 classes, but found {num_classes}: {class_names}"
    )

# Save class order so inference always uses the same mapping.
CLASS_NAMES_PATH.write_text(
    json.dumps(class_names, indent=2),
    encoding="utf-8",
)

# ------------------------------------------------------------
# Calculate class weights from the training dataset
# ------------------------------------------------------------

class_counts = np.zeros(num_classes, dtype=np.int64)

for _, labels in train_ds:
    labels = labels.numpy()
    values, counts = np.unique(labels, return_counts=True)

    for value, count in zip(values, counts):
        class_counts[value] += count

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.repeat(np.arange(num_classes), class_counts),
)

class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(class_weights_array)
}

print("\nTraining class distribution:")
for index, name in enumerate(class_names):
    print(
        f"{index}: {name:<45} "
        f"{class_counts[index]:>5} images "
        f"weight={class_weights[index]:.4f}"
    )

# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
valid_ds = valid_ds.prefetch(AUTOTUNE)

# ------------------------------------------------------------
# Data augmentation
# ------------------------------------------------------------

augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ],
    name="disease_augmentation",
)

# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

print("\nBuilding MobileNetV3Small...")

base_model = tf.keras.applications.MobileNetV3Small(
    input_shape=(*IMG_SIZE, 3),
    include_top=False,
    weights="imagenet",
    include_preprocessing=True,
)

# Freeze pretrained backbone for baseline training.
base_model.trainable = False

inputs = tf.keras.Input(
    shape=(*IMG_SIZE, 3),
    name="image",
)

x = augmentation(inputs)

x = base_model(
    x,
    training=False,
)

x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(x)

x = tf.keras.layers.Dropout(
    0.20,
    name="dropout",
)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="disease_output",
)(x)

model = tf.keras.Model(
    inputs,
    outputs,
    name="tomato_disease_mobilenetv3small",
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3,
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

print("\nModel summary:")
model.summary()

print("\nTrainable parameters:")
print(
    sum(
        np.prod(variable.shape)
        for variable in model.trainable_variables
    )
)

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
    patience=4,
    restore_best_weights=True,
    verbose=1,
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=2,
    min_lr=1e-6,
    verbose=1,
)

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

print("\nStarting training...")
print("Original dataset is read-only; no files are modified.")

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
# Save final model
# ------------------------------------------------------------

model.save(FINAL_MODEL_PATH)

print("\nTraining complete.")
print(f"Best model:  {BEST_MODEL_PATH}")
print(f"Final model: {FINAL_MODEL_PATH}")
print(f"Classes:     {CLASS_NAMES_PATH}")

# ------------------------------------------------------------
# Final validation evaluation
# ------------------------------------------------------------

print("\nFinal validation metrics:")
results = model.evaluate(
    valid_ds,
    verbose=1,
    return_dict=True,
)

for metric_name, value in results.items():
    print(f"{metric_name}: {value:.6f}")
