"""
Multiclass evaluation for the Tomato Disease Classification model.

Evaluates the trained Keras model on:
    dataset/tomato/valid/

Generates:
    results/evaluation_metrics.json
    results/evaluation_metrics.txt
    results/classification_report.txt
    results/confusion_matrix.png
"""

from pathlib import Path
import json
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.utils.config import PROJECT_ROOT, IMAGE_SIZE


# -------------------------------------------------------------------
# Defaults
# -------------------------------------------------------------------

DEFAULT_DATASET = PROJECT_ROOT / "dataset" / "tomato"
DEFAULT_MODEL = PROJECT_ROOT / "model" / "tomato_detector.keras"
DEFAULT_RESULTS = PROJECT_ROOT / "results"


# -------------------------------------------------------------------
# Dataset loading
# -------------------------------------------------------------------

def load_validation_dataset(dataset_path, batch_size=16):
    """
    Load the validation dataset using the same class ordering
    used during training.
    """

    validation_dir = Path(dataset_path) / "valid"

    if not validation_dir.exists():
        raise FileNotFoundError(
            f"Validation directory not found: {validation_dir}"
        )

    dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=batch_size,
        shuffle=False,
        label_mode="int",
    )

    return dataset


# -------------------------------------------------------------------
# Class names
# -------------------------------------------------------------------

def load_class_names():
    """Load class names saved during training."""

    class_names_path = PROJECT_ROOT / "model" / "class_names.json"

    if not class_names_path.exists():
        raise FileNotFoundError(
            f"Class names file not found: {class_names_path}"
        )

    with open(class_names_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    return class_names


# -------------------------------------------------------------------
# Evaluation
# -------------------------------------------------------------------

def evaluate_model(model_path, dataset_path, results_dir, batch_size=16):

    model_path = Path(model_path)
    results_dir = Path(results_dir)

    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Tomato Disease Model Evaluation")
    print("=" * 70)

    print(f"Model:     {model_path}")
    print(f"Dataset:   {Path(dataset_path) / 'valid'}")
    print(f"Results:   {results_dir}")
    print()

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    print("Loading trained model...")
    model = tf.keras.models.load_model(model_path)

    # ---------------------------------------------------------------
    # Load class names
    # ---------------------------------------------------------------

    class_names = load_class_names()

    print(f"Classes: {len(class_names)}")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")

    print()

    # ---------------------------------------------------------------
    # Load validation dataset
    # ---------------------------------------------------------------

    print("Loading validation dataset...")

    dataset = load_validation_dataset(
        dataset_path,
        batch_size=batch_size
    )

    print(f"Found {len(dataset.file_paths)} validation images.")
    print()

    # ---------------------------------------------------------------
    # Get predictions
    # ---------------------------------------------------------------

    print("Running predictions...")

    y_true = []
    y_pred = []

    for images, labels in dataset:

        predictions = model.predict(
            images,
            verbose=0
        )

        predicted_classes = np.argmax(
            predictions,
            axis=1
        )

        y_true.extend(labels.numpy())
        y_pred.extend(predicted_classes)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("Prediction completed.")
    print()

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    # ---------------------------------------------------------------
    # Classification report
    # ---------------------------------------------------------------

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=list(range(len(class_names)))
    )

    # ---------------------------------------------------------------
    # Print results
    # ---------------------------------------------------------------

    print("=" * 70)
    print("OVERALL RESULTS")
    print("=" * 70)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)

    # ---------------------------------------------------------------
    # Save metrics JSON
    # ---------------------------------------------------------------

    metrics = {
        "accuracy": float(accuracy),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1),
        "number_of_classes": len(class_names),
        "number_of_validation_images": int(len(y_true)),
        "class_names": class_names,
    }

    metrics_json_path = results_dir / "evaluation_metrics.json"

    with open(
        metrics_json_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4
        )

    # ---------------------------------------------------------------
    # Save text metrics
    # ---------------------------------------------------------------

    metrics_txt_path = results_dir / "evaluation_metrics.txt"

    with open(
        metrics_txt_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write("Tomato Disease Classification Evaluation\n")
        file.write("=" * 60 + "\n\n")

        file.write(f"Validation images: {len(y_true)}\n")
        file.write(f"Number of classes: {len(class_names)}\n\n")

        file.write(f"Accuracy:  {accuracy:.4f}\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall:    {recall:.4f}\n")
        file.write(f"F1 Score:  {f1:.4f}\n\n")

        file.write("=" * 60 + "\n")
        file.write("Classification Report\n")
        file.write("=" * 60 + "\n\n")

        file.write(report)

    # ---------------------------------------------------------------
    # Save classification report separately
    # ---------------------------------------------------------------

    report_path = results_dir / "classification_report.txt"

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:
        file.write(report)

    # ---------------------------------------------------------------
    # Plot confusion matrix
    # ---------------------------------------------------------------

    plt.figure(figsize=(14, 12))

    plt.imshow(cm, interpolation="nearest")

    plt.title("Tomato Disease Classification - Confusion Matrix")
    plt.colorbar()

    tick_marks = np.arange(len(class_names))

    plt.xticks(
        tick_marks,
        class_names,
        rotation=90
    )

    plt.yticks(
        tick_marks,
        class_names
    )

    # Write values inside matrix cells
    threshold = cm.max() / 2.0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    plt.ylabel("Actual Class")
    plt.xlabel("Predicted Class")

    plt.tight_layout()

    confusion_matrix_path = (
        results_dir / "confusion_matrix.png"
    )

    plt.savefig(
        confusion_matrix_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    # ---------------------------------------------------------------
    # Final output
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print("Evaluation completed successfully.")
    print("=" * 70)

    print(f"Metrics JSON:          {metrics_json_path}")
    print(f"Metrics TXT:           {metrics_txt_path}")
    print(f"Classification report: {report_path}")
    print(f"Confusion matrix:      {confusion_matrix_path}")


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Evaluate the tomato disease classification model."
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to trained Keras model."
    )

    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Dataset root containing train/ and valid/."
    )

    parser.add_argument(
        "--results",
        default=str(DEFAULT_RESULTS),
        help="Directory where evaluation results will be saved."
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Validation batch size."
    )

    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        dataset_path=args.dataset,
        results_dir=args.results,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()  