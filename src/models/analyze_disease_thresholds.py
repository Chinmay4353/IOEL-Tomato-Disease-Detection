from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Disease Model - Confidence Threshold Analysis
# ============================================================

INPUT_PATH = Path(
    "results/disease_finetuned_validation_probabilities.npz"
)

OUTPUT_DIR = Path("results")

CSV_OUTPUT = (
    OUTPUT_DIR
    / "disease_threshold_analysis.csv"
)

# Thresholds to evaluate
THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05,
)

# Class name used by the dataset
HEALTHY_CLASS = "healthy"


# ------------------------------------------------------------
# Load saved probabilities
# ------------------------------------------------------------

if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Probability file not found: {INPUT_PATH}"
    )

data = np.load(
    INPUT_PATH,
    allow_pickle=True,
)

probabilities = data["probabilities"]
y_true = data["y_true"]
class_names = [
    str(name)
    for name in data["class_names"]
]

# ------------------------------------------------------------
# Validate input
# ------------------------------------------------------------

if probabilities.ndim != 2:
    raise ValueError(
        "Expected probabilities to be a 2D array."
    )

if probabilities.shape[0] != len(y_true):
    raise ValueError(
        "Number of probabilities does not match "
        "number of labels."
    )

if probabilities.shape[1] != len(class_names):
    raise ValueError(
        "Number of probability columns does not "
        "match number of classes."
    )

if HEALTHY_CLASS not in class_names:
    raise ValueError(
        f"'{HEALTHY_CLASS}' class was not found."
    )

healthy_index = class_names.index(
    HEALTHY_CLASS
)

# ------------------------------------------------------------
# Model predictions
# ------------------------------------------------------------

predicted_classes = np.argmax(
    probabilities,
    axis=1,
)

max_confidence = np.max(
    probabilities,
    axis=1,
)

correct_predictions = (
    predicted_classes == y_true
)

actual_disease = (
    y_true != healthy_index
)

predicted_disease = (
    predicted_classes != healthy_index
)

total_samples = len(y_true)

# ------------------------------------------------------------
# Threshold analysis
# ------------------------------------------------------------

rows = []

for threshold in THRESHOLDS:

    # Model abstains when maximum confidence
    # is below the threshold.
    covered = (
        max_confidence >= threshold
    )

    abstained = ~covered

    covered_count = int(
        np.sum(covered)
    )

    abstained_count = int(
        np.sum(abstained)
    )

    # --------------------------------------------------------
    # Overall coverage / abstention
    # --------------------------------------------------------

    coverage = (
        covered_count / total_samples
        if total_samples > 0
        else 0.0
    )

    abstention_rate = (
        abstained_count / total_samples
        if total_samples > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Accuracy on covered predictions
    # --------------------------------------------------------

    if covered_count > 0:
        covered_accuracy = float(
            np.mean(
                correct_predictions[covered]
            )
        )
    else:
        covered_accuracy = 0.0

    # --------------------------------------------------------
    # Disease detection metrics
    #
    # These are calculated only on covered predictions.
    # --------------------------------------------------------

    covered_actual_disease = (
        actual_disease[covered]
    )

    covered_predicted_disease = (
        predicted_disease[covered]
    )

    true_positive = int(
        np.sum(
            covered_actual_disease
            & covered_predicted_disease
        )
    )

    false_positive = int(
        np.sum(
            ~covered_actual_disease
            & covered_predicted_disease
        )
    )

    false_negative = int(
        np.sum(
            covered_actual_disease
            & ~covered_predicted_disease
        )
    )

    true_negative = int(
        np.sum(
            ~covered_actual_disease
            & ~covered_predicted_disease
        )
    )

    disease_precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )

    disease_recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    disease_f1 = (
        2
        * disease_precision
        * disease_recall
        / (disease_precision + disease_recall)
        if (disease_precision + disease_recall) > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Disease / healthy coverage
    # --------------------------------------------------------

    actual_disease_count = int(
        np.sum(actual_disease)
    )

    actual_healthy_count = int(
        np.sum(~actual_disease)
    )

    disease_covered = int(
        np.sum(
            actual_disease
            & covered
        )
    )

    healthy_covered = int(
        np.sum(
            ~actual_disease
            & covered
        )
    )

    disease_coverage = (
        disease_covered / actual_disease_count
        if actual_disease_count > 0
        else 0.0
    )

    healthy_coverage = (
        healthy_covered / actual_healthy_count
        if actual_healthy_count > 0
        else 0.0
    )

    rows.append(
        {
            "threshold": float(threshold),
            "total_samples": total_samples,
            "covered_count": covered_count,
            "abstained_count": abstained_count,
            "coverage": coverage,
            "abstention_rate": abstention_rate,
            "covered_accuracy": covered_accuracy,
            "disease_precision": disease_precision,
            "disease_recall": disease_recall,
            "disease_f1": disease_f1,
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "disease_coverage": disease_coverage,
            "healthy_coverage": healthy_coverage,
        }
    )


# ------------------------------------------------------------
# Create DataFrame
# ------------------------------------------------------------

results = pd.DataFrame(rows)

# Save percentages as decimals in CSV.
results.to_csv(
    CSV_OUTPUT,
    index=False,
)

# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("\n" + "=" * 115)
print("DISEASE MODEL CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 115)

print(
    f"\nValidation samples: {total_samples}"
)

print(
    f"Classes: {len(class_names)}"
)

print(
    f"Healthy class: {HEALTHY_CLASS}"
)

print(
    "\nDisease classes:"
)

for index, name in enumerate(class_names):
    if index != healthy_index:
        print(f"  - {name}")

print(
    "\nThreshold results:"
)

print(
    "\n"
    f"{'Threshold':>9} "
    f"{'Coverage':>10} "
    f"{'Abstain':>10} "
    f"{'Accuracy':>11} "
    f"{'Disease P':>11} "
    f"{'Disease R':>11} "
    f"{'Disease F1':>11} "
    f"{'Disease Cov':>12}"
)

print("-" * 115)

for _, row in results.iterrows():

    print(
        f"{row['threshold']:>9.2f} "
        f"{row['coverage'] * 100:>9.2f}% "
        f"{row['abstention_rate'] * 100:>9.2f}% "
        f"{row['covered_accuracy'] * 100:>10.2f}% "
        f"{row['disease_precision'] * 100:>10.2f}% "
        f"{row['disease_recall'] * 100:>10.2f}% "
        f"{row['disease_f1'] * 100:>10.2f}% "
        f"{row['disease_coverage'] * 100:>11.2f}%"
    )

# ------------------------------------------------------------
# Highest disease F1
# ------------------------------------------------------------

best_f1_index = results[
    "disease_f1"
].idxmax()

best_f1 = results.loc[
    best_f1_index
]

print(
    "\n" + "=" * 115
)

print(
    "Highest disease F1 in tested thresholds:"
)

print(
    f"Threshold: "
    f"{best_f1['threshold']:.2f}"
)

print(
    f"Disease F1: "
    f"{best_f1['disease_f1'] * 100:.2f}%"
)

print(
    f"Disease precision: "
    f"{best_f1['disease_precision'] * 100:.2f}%"
)

print(
    f"Disease recall: "
    f"{best_f1['disease_recall'] * 100:.2f}%"
)

print(
    f"Coverage: "
    f"{best_f1['coverage'] * 100:.2f}%"
)

print(
    f"Abstention: "
    f"{best_f1['abstention_rate'] * 100:.2f}%"
)

print(
    f"\nCSV saved to:"
)

print(CSV_OUTPUT)

print(
    "\nNote:"
)

print(
    "This analysis does not change the model or select "
    "a production threshold automatically."
)

print(
    "The threshold must be chosen according to the "
    "intended behavior of the IOEL system."
)

print(
    "=" * 115
)
