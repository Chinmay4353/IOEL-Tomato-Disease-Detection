"""
Tomato Disease Classification - Training Pipeline

Project:
    IOEL

Model:
    MobileNetV2 Transfer Learning

Classes:
    11 tomato disease / healthy classes

Dataset structure:
    dataset/
    └── tomato/
        ├── train/
        │   ├── Bacterial_spot/
        │   ├── Early_blight/
        │   ├── Late_blight/
        │   ├── Leaf_Mold/
        │   ├── Septoria_leaf_spot/
        │   ├── Spider_mites Two-spotted_spider_mite/
        │   ├── Target_Spot/
        │   ├── Tomato_Yellow_Leaf_Curl_Virus/
        │   ├── Tomato_mosaic_virus/
        │   ├── healthy/
        │   └── powdery_mildew/
        │
        └── valid/
            └── same 11 classes

Outputs:
    model/tomato_detector.keras
    model/class_names.json

    results/
        metrics.txt
        metrics.json
        training_accuracy.png
        training_loss.png
        training_history.png
"""

from pathlib import Path
import argparse
import json
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.applications import MobileNetV2

from src.utils.config import (
    PROJECT_ROOT,
    IMAGE_SIZE,
    KERAS_MODEL_PATH,
    RESULTS_DIR,
)


# ============================================================
# Configuration
# ============================================================

SEED = 42

DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "tomato"
DEFAULT_MODEL_OUTPUT = KERAS_MODEL_PATH
DEFAULT_RESULTS = RESULTS_DIR

DEFAULT_EPOCHS = 5
DEFAULT_BATCH_SIZE = 16
DEFAULT_LEARNING_RATE = 0.001


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Dataset
# ============================================================

def load_datasets(dataset_root: Path, batch_size: int):
    """
    Load training and validation datasets.

    Returns:
        train_dataset
        validation_dataset
        class_names
    """

    train_dir = dataset_root / "train"
    valid_dir = dataset_root / "valid"

    if not train_dir.is_dir():
        raise FileNotFoundError(
            f"Training directory does not exist:\n{train_dir}"
        )

    if not valid_dir.is_dir():
        raise FileNotFoundError(
            f"Validation directory does not exist:\n{valid_dir}"
        )

    print("=" * 70)
    print("LOADING DATASET")
    print("=" * 70)

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=True,
        seed=SEED,
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        valid_dir,
        labels="inferred",
        label_mode="int",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=False,
    )

    train_classes = train_dataset.class_names
    validation_classes = validation_dataset.class_names

    if train_classes != validation_classes:
        raise ValueError(
            "Training and validation class names/order do not match.\n\n"
            f"Training classes:\n{train_classes}\n\n"
            f"Validation classes:\n{validation_classes}"
        )

    if len(train_classes) < 2:
        raise ValueError(
            f"Expected multiple classes, but found {len(train_classes)}."
        )

    print()
    print(f"Number of classes: {len(train_classes)}")
    print("Class mapping:")

    for index, class_name in enumerate(train_classes):
        print(f"  {index}: {class_name}")

    print()

    return train_dataset, validation_dataset, train_classes


# ============================================================
# Dataset optimization
# ============================================================

def optimize_datasets(train_dataset, validation_dataset):
    """
    Configure efficient TensorFlow dataset pipelines.
    """

    autotune = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(autotune)
    validation_dataset = validation_dataset.prefetch(autotune)

    return train_dataset, validation_dataset


# ============================================================
# Model
# ============================================================

def build_model(num_classes: int, learning_rate: float):
    """
    Build MobileNetV2 transfer-learning classifier.

    Important:
    MobileNetV2 expects RGB pixels transformed from:

        [0, 255] -> [-1, 1]

    This implementation uses Keras Rescaling instead of
    Lambda(preprocess_input), making the model safe to
    serialize and reload.
    """

    # --------------------------------------------------------
    # Data augmentation
    # --------------------------------------------------------

    augmentation = models.Sequential(
        [
            layers.RandomFlip(
                mode="horizontal",
                name="random_flip",
            ),
            layers.RandomRotation(
                factor=0.10,
                name="random_rotation",
            ),
            layers.RandomZoom(
                height_factor=0.10,
                width_factor=0.10,
                name="random_zoom",
            ),
            layers.RandomContrast(
                factor=0.10,
                name="random_contrast",
            ),
        ],
        name="data_augmentation",
    )

    # --------------------------------------------------------
    # MobileNetV2 base model
    # --------------------------------------------------------

    base_model = MobileNetV2(
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )

    # Freeze pretrained layers during initial training.
    base_model.trainable = False

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    inputs = layers.Input(
        shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        dtype=tf.float32,
        name="input_image",
    )

    # --------------------------------------------------------
    # Augmentation
    # --------------------------------------------------------

    x = augmentation(inputs)

    # --------------------------------------------------------
    # MobileNetV2 preprocessing
    #
    # Equivalent to:
    #
    # (pixel / 127.5) - 1
    #
    # No Lambda layer is used.
    # --------------------------------------------------------

    x = layers.Rescaling(
        scale=1.0 / 127.5,
        offset=-1.0,
        name="mobilenet_v2_preprocess",
    )(x)

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    x = base_model(
        x,
        training=False,
    )

    # --------------------------------------------------------
    # Classification head
    # --------------------------------------------------------

    x = layers.GlobalAveragePooling2D(
        name="global_average_pooling",
    )(x)

    x = layers.Dropout(
        rate=0.30,
        name="dropout",
    )(x)

    outputs = layers.Dense(
        units=num_classes,
        activation="softmax",
        name="disease_prediction",
    )(x)

    # --------------------------------------------------------
    # Complete model
    # --------------------------------------------------------

    model = models.Model(
        inputs=inputs,
        outputs=outputs,
        name="tomato_disease_mobilenetv2",
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate,
        ),
        loss="sparse_categorical_crossentropy",
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(
                name="accuracy"
            )
        ],
    )

    return model


# ============================================================
# Save class names
# ============================================================

def save_class_names(class_names, model_output: Path):
    """
    Save class mapping next to the trained model.
    """

    model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    class_names_path = (
        model_output.parent / "class_names.json"
    )

    with open(
        class_names_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            class_names,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return class_names_path


# ============================================================
# Training plots
# ============================================================

def save_training_plots(history, results_dir: Path):
    """
    Save accuracy and loss plots.
    """

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    history_data = history.history

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_data["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history_data["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title(
        "Tomato Disease Classification - Accuracy"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir / "training_accuracy.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    plt.plot(
        history_data["loss"],
        label="Training Loss",
    )

    plt.plot(
        history_data["val_loss"],
        label="Validation Loss",
    )

    plt.title(
        "Tomato Disease Classification - Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir / "training_loss.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # --------------------------------------------------------
    # Combined plot
    # --------------------------------------------------------

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)

    plt.plot(
        history_data["accuracy"],
        label="Training Accuracy",
    )

    plt.plot(
        history_data["val_accuracy"],
        label="Validation Accuracy",
    )

    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)

    plt.plot(
        history_data["loss"],
        label="Training Loss",
    )

    plt.plot(
        history_data["val_loss"],
        label="Validation Loss",
    )

    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir / "training_history.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


# ============================================================
# Metrics
# ============================================================

def save_metrics(
    history,
    model,
    validation_dataset,
    results_dir: Path,
):
    """
    Save training and validation metrics.
    """

    history_data = history.history

    train_accuracy = float(
        history_data["accuracy"][-1]
    )

    train_loss = float(
        history_data["loss"][-1]
    )

    best_validation_accuracy = float(
        max(history_data["val_accuracy"])
    )

    best_validation_loss = float(
        min(history_data["val_loss"])
    )

    print()
    print("=" * 70)
    print("FINAL VALIDATION EVALUATION")
    print("=" * 70)

    validation_loss, validation_accuracy = (
        model.evaluate(
            validation_dataset,
            verbose=1,
        )
    )

    metrics = {
        "train_accuracy_last_epoch": train_accuracy,
        "train_loss_last_epoch": train_loss,
        "validation_accuracy": float(
            validation_accuracy
        ),
        "validation_loss": float(
            validation_loss
        ),
        "best_validation_accuracy": (
            best_validation_accuracy
        ),
        "best_validation_loss": (
            best_validation_loss
        ),
    }

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    metrics_json = results_dir / "metrics.json"

    with open(
        metrics_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    # --------------------------------------------------------
    # TXT
    # --------------------------------------------------------

    metrics_txt = results_dir / "metrics.txt"

    with open(
        metrics_txt,
        "w",
        encoding="utf-8",
    ) as file:
        for key, value in metrics.items():
            file.write(
                f"{key}: {value}\n"
            )

    return metrics


# ============================================================
# Main training pipeline
# ============================================================

def train_model(
    dataset_root,
    epochs,
    batch_size,
    learning_rate,
    model_output,
    results_dir,
):
    """
    Complete training pipeline.
    """

    dataset_root = Path(dataset_root)
    model_output = Path(model_output)
    results_dir = Path(results_dir)

    print()
    print("=" * 70)
    print("TOMATO DISEASE CLASSIFICATION")
    print("MOBILENETV2 TRAINING PIPELINE")
    print("=" * 70)

    print(f"Dataset       : {dataset_root}")
    print(f"Epochs        : {epochs}")
    print(f"Batch size    : {batch_size}")
    print(f"Learning rate : {learning_rate}")
    print(f"Model output  : {model_output}")
    print(f"Results       : {results_dir}")
    print()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    (
        train_dataset,
        validation_dataset,
        class_names,
    ) = load_datasets(
        dataset_root,
        batch_size,
    )

    train_dataset, validation_dataset = (
        optimize_datasets(
            train_dataset,
            validation_dataset,
        )
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print("=" * 70)
    print("BUILDING MODEL")
    print("=" * 70)

    model = build_model(
        num_classes=len(class_names),
        learning_rate=learning_rate,
    )

    model.summary()

    # --------------------------------------------------------
    # Save class mapping
    # --------------------------------------------------------

    class_names_path = save_class_names(
        class_names,
        model_output,
    )

    print()
    print(
        f"Class mapping saved: {class_names_path}"
    )

    # --------------------------------------------------------
    # Callbacks
    # --------------------------------------------------------

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(model_output),
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

    reduce_learning_rate = (
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        )
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=[
            checkpoint,
            early_stopping,
            reduce_learning_rate,
        ],
        verbose=1,
    )

    # --------------------------------------------------------
    # Load best checkpoint
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST MODEL")
    print("=" * 70)

    best_model = tf.keras.models.load_model(
        model_output
    )

    # --------------------------------------------------------
    # Save plots
    # --------------------------------------------------------

    save_training_plots(
        history,
        results_dir,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics = save_metrics(
        history,
        best_model,
        validation_dataset,
        results_dir,
    )

    # --------------------------------------------------------
    # Final save
    # --------------------------------------------------------

    best_model.save(
        model_output
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print(
        f"Best validation accuracy : "
        f"{metrics['best_validation_accuracy']:.4f}"
    )

    print(
        f"Validation accuracy      : "
        f"{metrics['validation_accuracy']:.4f}"
    )

    print(
        f"Validation loss          : "
        f"{metrics['validation_loss']:.4f}"
    )

    print()
    print(f"Model       : {model_output}")
    print(f"Class names : {class_names_path}")
    print(f"Results     : {results_dir}")
    print()


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Train the IOEL tomato disease "
            "classification model."
        )
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=str(DEFAULT_DATASET),
        help=(
            "Dataset root containing "
            "train/ and valid/."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Initial learning rate.",
    )

    parser.add_argument(
        "--model-output",
        type=str,
        default=str(DEFAULT_MODEL_OUTPUT),
        help="Path for the trained Keras model.",
    )

    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS),
        help="Directory for training results.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        raise ValueError(
            "Epochs must be at least 1."
        )

    if args.batch_size < 1:
        raise ValueError(
            "Batch size must be at least 1."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "Learning rate must be greater than 0."
        )

    train_model(
        dataset_root=Path(args.dataset),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_output=Path(args.model_output),
        results_dir=Path(args.results),
    )


if __name__ == "__main__":
    main()