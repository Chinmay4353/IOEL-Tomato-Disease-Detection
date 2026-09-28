from pathlib import Path
import json

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================================
# Disease Model - Separate Test Set Evaluation
# Thresholds: 0.80 and 0.85
# ============================================================

IMG_SIZE = (160, 160)
BATCH_SIZE = 32

TEST_DIR = Path(
    "dataset/tomato/test"
)

MODEL_PATH = Path(
    "models/tomato_disease_finetuned_best.keras"
)

CLASS_NAMES_PATH = Path(
    "models/tomato_disease_classes.json"
)

OUTPUT_DIR = Path("results")

CSV_OUTPUT = (
    OUTPUT_DIR
    / "disease_test_threshold_comparison.csv"
)

THRESHOLDS = [0.80, 0.85]

HEALTHY_CLASS = "healthy"


# ------------------------------------------------------------
# Check required files/folders
# ------------------------------------------------------------

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"Test directory not found: {TEST_DIR}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"Class names file not found: "
        f"{CLASS_NAMES_PATH}"
    )


# ------------------------------------------------------------
# Load class names
# ------------------------------------------------------------

class_names = json.loads(
    CLASS_NAMES_PATH.read_text(
        encoding="utf-8"
    )
)

healthy_index = class_names.index(
    HEALTHY_CLASS
)

num_classes = len(class_names)


# ------------------------------------------------------------
# Load completely separate test set
# ------------------------------------------------------------

print("=" * 100)
print("SEPARATE TEST SET EVALUATION")
print("=" * 100)

print(
    f"\nTest directory: {TEST_DIR}"
)

print(
    f"Model: {MODEL_PATH}"
)

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ------------------------------------------------------------
# Verify class order
# ------------------------------------------------------------

dataset_class_names = test_ds.class_names

if dataset_class_names != class_names:
    raise ValueError(
        "\nTest dataset class order does not match "
        "models/tomato_disease_classes.json.\n\n"
        f"Dataset order:\n{dataset_class_names}\n\n"
        f"Expected order:\n{class_names}"
    )

print(
    f"\nTest images: "
    f"{sum(1 for _ in TEST_DIR.glob('*/*') if _.is_file())}"
)

print("\nClass order verified.")


# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

print("\nLoading fine-tuned model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ------------------------------------------------------------
# Generate test probabilities
# ------------------------------------------------------------

print("\nGenerating predictions...")

all_probabilities = []
all_labels = []

for batch_index, (images, labels) in enumerate(
    test_ds,
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

predicted_classes = np.argmax(
    probabilities,
    axis=1,
)

max_confidence = np.max(
    probabilities,
    axis=1,
)


# ------------------------------------------------------------
# Basic test-set metrics
# ------------------------------------------------------------

overall_accuracy = accuracy_score(
    y_true,
    predicted_classes,
)

print("\n" + "=" * 100)
print("RAW TEST SET RESULT")
print("=" * 100)

print(
    f"\nTotal test images : {len(y_true)}"
)

print(
    f"11-class accuracy : "
    f"{overall_accuracy * 100:.2f}%"
)


# ------------------------------------------------------------
# Disease / healthy definitions
# ------------------------------------------------------------

actual_disease = (
    y_true != healthy_index
)

actual_healthy = (
    y_true == healthy_index
)

predicted_disease = (
    predicted_classes != healthy_index
)

predicted_healthy = (
    predicted_classes == healthy_index
)

total_disease = int(
    np.sum(actual_disease)
)

total_healthy = int(
    np.sum(actual_healthy)
)


# ------------------------------------------------------------
# Threshold evaluation
# ------------------------------------------------------------

results = []

confusion_matrices = {}


for threshold in THRESHOLDS:

    print("\n" + "=" * 100)
    print(
        f"THRESHOLD {threshold:.2f}"
    )
    print("=" * 100)

    # --------------------------------------------------------
    # Accepted / abstained
    # --------------------------------------------------------

    accepted = (
        max_confidence >= threshold
    )

    abstained = ~accepted

    accepted_count = int(
        np.sum(accepted)
    )

    abstained_count = int(
        np.sum(abstained)
    )

    coverage = (
        accepted_count / len(y_true)
    )

    abstention_rate = (
        abstained_count / len(y_true)
    )


    # --------------------------------------------------------
    # Accepted accuracy
    # --------------------------------------------------------

    if accepted_count > 0:

        accepted_accuracy = float(
            np.mean(
                predicted_classes[accepted]
                == y_true[accepted]
            )
        )

    else:

        accepted_accuracy = 0.0


    # --------------------------------------------------------
    # Disease / healthy confusion
    # among accepted predictions
    # --------------------------------------------------------

    tp = int(
        np.sum(
            actual_disease
            & predicted_disease
            & accepted
        )
    )

    fn = int(
        np.sum(
            actual_disease
            & predicted_healthy
            & accepted
        )
    )

    fp = int(
        np.sum(
            actual_healthy
            & predicted_disease
            & accepted
        )
    )

    tn = int(
        np.sum(
            actual_healthy
            & predicted_healthy
            & accepted
        )
    )


    # --------------------------------------------------------
    # Disease false-negative rate
    # --------------------------------------------------------

    disease_fn_rate = (
        fn / total_disease
        if total_disease > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Healthy false-positive rate
    # --------------------------------------------------------

    healthy_fp_rate = (
        fp / total_healthy
        if total_healthy > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Disease precision / recall / F1
    # --------------------------------------------------------

    disease_precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    disease_recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    disease_f1 = (
        2
        * disease_precision
        * disease_recall
        / (disease_precision + disease_recall)
        if (
            disease_precision
            + disease_recall
        ) > 0
        else 0.0
    )


    # --------------------------------------------------------
    # 11-class confusion matrix
    #
    # Only accepted predictions are included.
    # Abstained images are intentionally excluded.
    # --------------------------------------------------------

    if accepted_count > 0:

        cm = confusion_matrix(
            y_true[accepted],
            predicted_classes[accepted],
            labels=np.arange(num_classes),
        )

    else:

        cm = np.zeros(
            (num_classes, num_classes),
            dtype=int,
        )

    confusion_matrices[
        threshold
    ] = cm


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    results.append(
        {
            "threshold": threshold,

            "total_test_images":
                len(y_true),

            "accepted_count":
                accepted_count,

            "abstained_count":
                abstained_count,

            "coverage":
                coverage,

            "abstention_rate":
                abstention_rate,

            "accepted_accuracy":
                accepted_accuracy,

            "disease_false_negative_rate":
                disease_fn_rate,

            "healthy_false_positive_rate":
                healthy_fp_rate,

            "disease_precision":
                disease_precision,

            "disease_recall":
                disease_recall,

            "disease_f1":
                disease_f1,

            "true_positive_disease":
                tp,

            "false_negative_disease":
                fn,

            "false_positive_healthy":
                fp,

            "true_negative_healthy":
                tn,
        }
    )


    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print(
        f"\nCoverage            : "
        f"{coverage * 100:.2f}%"
    )

    print(
        f"Abstention          : "
        f"{abstention_rate * 100:.2f}%"
    )

    print(
        f"Accepted accuracy   : "
        f"{accepted_accuracy * 100:.2f}%"
    )

    print(
        f"Disease FN rate     : "
        f"{disease_fn_rate * 100:.2f}%"
    )

    print(
        f"Healthy FP rate     : "
        f"{healthy_fp_rate * 100:.2f}%"
    )

    print(
        f"Disease precision   : "
        f"{disease_precision * 100:.2f}%"
    )

    print(
        f"Disease recall      : "
        f"{disease_recall * 100:.2f}%"
    )

    print(
        f"Disease F1          : "
        f"{disease_f1 * 100:.2f}%"
    )

    print(
        f"\nTP / FN / FP / TN   : "
        f"{tp} / {fn} / {fp} / {tn}"
    )

    # --------------------------------------------------------
    # Print confusion matrix
    # --------------------------------------------------------

    print(
        "\n11-class confusion matrix "
        "(accepted predictions only):"
    )

    print(
        "Rows = Actual"
    )

    print(
        "Columns = Predicted\n"
    )

    print(
        f"{'':45s}"
        + "".join(
            f"{i:>7d}"
            for i in range(num_classes)
        )
    )

    for index, row in enumerate(cm):

        print(
            f"{class_names[index]:45s}"
            + "".join(
                f"{int(value):>7d}"
                for value in row
            )
        )


# ------------------------------------------------------------
# Comparison table
# ------------------------------------------------------------

comparison = pd.DataFrame(
    results
)

comparison.to_csv(
    CSV_OUTPUT,
    index=False,
)


print("\n" + "=" * 100)
print("THRESHOLD COMPARISON")
print("=" * 100)

print(
    f"\n{'Metric':<32}"
    f"{'0.80':>15}"
    f"{'0.85':>15}"
)

print("-" * 65)

metric_rows = [
    (
        "Coverage",
        "coverage",
        True,
    ),
    (
        "Abstention rate",
        "abstention_rate",
        True,
    ),
    (
        "Accepted accuracy",
        "accepted_accuracy",
        True,
    ),
    (
        "Disease FN rate",
        "disease_false_negative_rate",
        True,
    ),
    (
        "Healthy FP rate",
        "healthy_false_positive_rate",
        True,
    ),
    (
        "Disease precision",
        "disease_precision",
        True,
    ),
    (
        "Disease recall",
        "disease_recall",
        True,
    ),
    (
        "Disease F1",
        "disease_f1",
        True,
    ),
]


for display_name, column, percentage in metric_rows:

    value_80 = comparison.loc[
        comparison["threshold"] == 0.80,
        column,
    ].iloc[0]

    value_85 = comparison.loc[
        comparison["threshold"] == 0.85,
        column,
    ].iloc[0]

    if percentage:

        print(
            f"{display_name:<32}"
            f"{value_80 * 100:>14.2f}%"
            f"{value_85 * 100:>14.2f}%"
        )

    else:

        print(
            f"{display_name:<32}"
            f"{value_80:>15.4f}"
            f"{value_85:>15.4f}"
        )


print("\nConfusion counts:")

print(
    f"{'TP disease':<32}"
    f"{int(comparison.loc[comparison['threshold'] == 0.80, 'true_positive_disease'].iloc[0]):>15d}"
    f"{int(comparison.loc[comparison['threshold'] == 0.85, 'true_positive_disease'].iloc[0]):>15d}"
)

print(
    f"{'FN disease':<32}"
    f"{int(comparison.loc[comparison['threshold'] == 0.80, 'false_negative_disease'].iloc[0]):>15d}"
    f"{int(comparison.loc[comparison['threshold'] == 0.85, 'false_negative_disease'].iloc[0]):>15d}"
)

print(
    f"{'FP healthy':<32}"
    f"{int(comparison.loc[comparison['threshold'] == 0.80, 'false_positive_healthy'].iloc[0]):>15d}"
    f"{int(comparison.loc[comparison['threshold'] == 0.85, 'false_positive_healthy'].iloc[0]):>15d}"
)

print(
    f"{'TN healthy':<32}"
    f"{int(comparison.loc[comparison['threshold'] == 0.80, 'true_negative_healthy'].iloc[0]):>15d}"
    f"{int(comparison.loc[comparison['threshold'] == 0.85, 'true_negative_healthy'].iloc[0]):>15d}"
)


# ------------------------------------------------------------
# Output location
# ------------------------------------------------------------

print("\n" + "=" * 100)

print(
    "\nComparison CSV saved to:"
)

print(CSV_OUTPUT)

print(
    "\nImportant:"
)

print(
    "The test set must be completely separate from "
    "training, validation, fine-tuning, and threshold selection."
)

print(
    "Abstained images are excluded from accepted accuracy "
    "and the 11-class confusion matrices."
)

print(
    "Disease FN and healthy FP rates use the complete "
    "actual disease/healthy populations as denominators."
)

print("=" * 100)
