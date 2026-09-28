from pathlib import Path
import tensorflow as tf

MODEL_DIR = Path("models")
DATASET = Path("dataset/gate_split")

IMG_SIZE = (160, 160)
BATCH_SIZE = 32
EPOCHS = 8
SEED = 42
LEARNING_RATE = 1e-5

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET / "validation",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

model = tf.keras.models.load_model(
    MODEL_DIR / "tomato_gate_best.keras"
)

print("Loaded baseline Gate model.")

# Find MobileNetV3 backbone.
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise RuntimeError("Could not find MobileNetV3 backbone.")

print("Backbone:", base_model.name)
print("Total backbone layers:", len(base_model.layers))

# Freeze everything first.
base_model.trainable = True

# Fine-tune only the last 30 layers.
fine_tune_from = max(0, len(base_model.layers) - 30)

for index, layer in enumerate(base_model.layers):
    if index < fine_tune_from:
        layer.trainable = False
    elif isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

trainable_count = sum(
    1 for layer in base_model.layers
    if layer.trainable
)

print("Fine-tuning last 30 backbone layers.")
print("Trainable backbone layers:", trainable_count)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
    ],
)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_DIR / "tomato_gate_finetuned_best.keras",
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

model.save(
    MODEL_DIR / "tomato_gate_finetuned_final.keras"
)

print("\nGate fine-tuning complete.")
print(
    "Best model:",
    MODEL_DIR / "tomato_gate_finetuned_best.keras"
)
print(
    "Final model:",
    MODEL_DIR / "tomato_gate_finetuned_final.keras"
)
