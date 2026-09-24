"""Training pipeline for the tomato plant detector."""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model

from src.utils.config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_DIR, PROJECT_ROOT, RESULTS_DIR

logger = logging.getLogger(__name__)


def prepare_training_directories(dataset_root: str | Path) -> Tuple[Path, Path]:
    """Create train and validation folders for the dataset."""
    root = Path(dataset_root)
    train_dir = root / "train"
    val_dir = root / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    return train_dir, val_dir


def build_model(
    input_shape: tuple[int, int, int] = (IMAGE_SIZE, IMAGE_SIZE, 3),
    learning_rate: float = 0.001,
    num_classes: int = 1,
    lightweight: bool = False,
) -> Model:
    """Construct a MobileNetV2 classifier for one or more leaf classes."""
    inputs = tf.keras.Input(shape=input_shape)
    if lightweight:
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(16, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
        x = layers.GlobalAveragePooling2D()(x)
    else:
        base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights="imagenet")
        base_model.trainable = False
        x = base_model(inputs, training=False)
        x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        loss = "sparse_categorical_crossentropy"

    model = tf.keras.Model(inputs, outputs, name="tomato_detector")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


def create_callbacks(model_path: str | Path) -> list:
    """Create the training callbacks used during the transfer-learning phase."""
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        ModelCheckpoint(filepath=str(model_path), monitor="val_loss", save_best_only=True, mode="min"),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6),
    ]
    return callbacks


def load_training_data(dataset_root: str | Path, batch_size: int = 32, target_size: tuple[int, int] = (IMAGE_SIZE, IMAGE_SIZE)) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """Load train and validation images using the supported TensorFlow Keras dataset API."""
    root = Path(dataset_root)
    train_dir = root / "train"
    val_dir = root / "val"
    if not val_dir.exists():
        val_dir = root / "valid"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Training directories not found in {root}. Run data splitting first.")

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=target_size,
        batch_size=batch_size,
        shuffle=True,
    )
    val_dataset = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        labels="inferred",
        label_mode="int",
        image_size=target_size,
        batch_size=batch_size,
        shuffle=False,
    )

    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset


def save_training_plots(history: tf.keras.callbacks.History, output_dir: str | Path) -> None:
    """Save training and validation accuracy/loss plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    acc = history.history.get("accuracy", [])
    val_acc = history.history.get("val_accuracy", [])
    loss = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train_accuracy")
    plt.plot(val_acc, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "training_history.png")
    plt.close()


def train_model(
    dataset_root: str | Path,
    epochs: int = 10,
    batch_size: int = 32,
    steps_per_epoch: int | None = None,
    validation_steps: int | None = None,
    model_path: str | Path = MODEL_DIR / "tomato_detector.keras",
    output_dir: str | Path = RESULTS_DIR,
    verbose: int = 1,
    lightweight: bool = False,
) -> Dict[str, float]:
    """Train the transfer-learning model for binary plant classification."""
    root = Path(dataset_root)
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_generator, val_generator = load_training_data(root, batch_size=batch_size)
    train_dir = root / "train"
    class_names = sorted(path.name for path in train_dir.iterdir() if path.is_dir())
    num_classes = len(class_names)
    model = build_model(num_classes=num_classes if num_classes > 2 else 1, lightweight=lightweight)
    callbacks = create_callbacks(model_path)

    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=verbose,
    )

    model.save(model_path)
    class_names_path = model_path.parent / CLASS_NAMES_PATH.name
    class_names_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    save_training_plots(history, output_dir)

    final_metrics = {
        "train_accuracy": float(history.history["accuracy"][-1]),
        "val_accuracy": float(history.history["val_accuracy"][-1]),
        "train_loss": float(history.history["loss"][-1]),
        "val_loss": float(history.history["val_loss"][-1]),
    }

    metrics_path = output_dir / "metrics.txt"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        for key, value in final_metrics.items():
            handle.write(f"{key}: {value}\n")

    logger.info("Training finished successfully.")
    return final_metrics


def main() -> None:
    print("Phase 4: MobileNetV2 transfer learning training pipeline")
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Model output: {MODEL_DIR / 'tomato_detector.keras'}")
    print(f"Results directory: {RESULTS_DIR}")
    print("Ready to train when the dataset has been split into train/ and val/ folders.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
