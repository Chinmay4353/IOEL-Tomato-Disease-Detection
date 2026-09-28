from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Fine-Grained Disease Confidence Threshold Analysis
# Thresholds: 0.70 to 0.95 in 0.01 steps
# ============================================================

INPUT_PATH = Path(
    "results/disease_finetuned_validation_probabilities.npz"
)

OUTPUT_DIR = Path("results")

CSV_OUTPUT = (
    OUTPUT_DIR
    / "disease_threshold_analysis_fine.csv"
)

HEALTHY_CLASS = "healthy"

THRESHOLDS = np.round(
    np.arange(0.70, 0.951, 0.01),
    2,
)


# ------------------------------------------------------------
# Load saved validation probabilities
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
        "Probability count does not match label count."
    )

if probabilities.shape[1] != len(class_names):
    raise ValueError(
        "Probability class count does not match "
        "class names."
    )

if HEALTHY_CLASS not in class_names:
    raise ValueError(
        f"Class '{HEALTHY_CLASS}' was not found."
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

total_samples = len(y_true)

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
# Threshold analysis
# ------------------------------------------------------------

rows = []

for threshold in THRESHOLDS:

    # --------------------------------------------------------
    # Coverage / abstention
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
        accepted_count / total_samples
    )

    abstention_rate = (
        abstained_count / total_samples
    )

    # --------------------------------------------------------
    # Accepted prediction accuracy
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
    # Disease / healthy confusion counts
    #
    # These are calculated only among ACCEPTED
    # predictions.
    # --------------------------------------------------------

    accepted_actual_disease = (
        actual_disease & accepted
    )

    accepted_actual_healthy = (
        actual_healthy & accepted
    )

    accepted_predicted_disease = (
        predicted_disease & accepted
    )

    accepted_predicted_healthy = (
        predicted_healthy & accepted
    )


    # Disease -> Disease
    true_positive = int(
        np.sum(
            accepted_actual_disease
            & accepted_predicted_disease
        )
    )

    # Healthy -> Disease
    false_positive = int(
        np.sum(
            accepted_actual_healthy
            & accepted_predicted_disease
        )
    )

    # Disease -> Healthy
    false_negative = int(
        np.sum(
            accepted_actual_disease
            & accepted_predicted_healthy
        )
    )

    # Healthy -> Healthy
    true_negative = int(
        np.sum(
            accepted_actual_healthy
            & accepted_predicted_healthy
        )
    )


    # --------------------------------------------------------
    # Disease false-negative rate
    #
    # Among actual disease images, how many accepted
    # predictions were incorrectly classified as healthy?
    # --------------------------------------------------------

    false_negative_disease_rate = (
        false_negative / total_disease
        if total_disease > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Healthy false-positive rate
    #
    # Among actual healthy images, how many accepted
    # predictions were incorrectly classified as disease?
    # --------------------------------------------------------

    false_positive_healthy_rate = (
        false_positive / total_healthy
        if total_healthy > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Disease detection recall
    # --------------------------------------------------------

    disease_recall = (
        true_positive / total_disease
        if total_disease > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Disease precision
    # --------------------------------------------------------

    disease_precision = (
        true_positive
        / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )


    # --------------------------------------------------------
    # Accepted disease / healthy counts
    # --------------------------------------------------------

    accepted_disease_count = int(
        np.sum(
            actual_disease & accepted
        )
    )

    accepted_healthy_count = int(
        np.sum(
            actual_healthy & accepted
        )
    )


    rows.append(
        {
            "threshold": float(threshold),

            "total_samples": total_samples,

            "accepted_count": accepted_count,
            "abstained_count": abstained_count,

            "coverage": coverage,
            "abstention_rate": abstention_rate,

            "accepted_accuracy": accepted_accuracy,

            "total_disease": total_disease,
            "total_healthy": total_healthy,

            "accepted_disease_count":
                accepted_disease_count,

            "accepted_healthy_count":
                accepted_healthy_count,

            "true_positive_disease":
                true_positive,

            "false_positive_healthy":
                false_positive,

            "false_negative_disease":
                false_negative,

            "true_negative_healthy":
                true_negative,

            "false_negative_disease_rate":
                false_negative_disease_rate,

            "false_positive_healthy_rate":
                false_positive_healthy_rate,

            "disease_recall":
                disease_recall,

            "disease_precision":
                disease_precision,
        }
    )


# ------------------------------------------------------------
# DataFrame
# ------------------------------------------------------------

results = pd.DataFrame(rows)


# ------------------------------------------------------------
# Save CSV
# ------------------------------------------------------------

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

results.to_csv(
    CSV_OUTPUT,
    index=False,
)


# ------------------------------------------------------------
# Console report
# ------------------------------------------------------------

print("\n" + "=" * 155)
print(
    "FINE-GRAINED DISEASE CONFIDENCE THRESHOLD ANALYSIS"
)
print("=" * 155)

print(
    f"\nValidation samples : {total_samples}"
)

print(
    f"Actual disease     : {total_disease}"
)

print(
    f"Actual healthy     : {total_healthy}"
)

print(
    "\nThresholds: 0.70 -> 0.95 "
    "(step = 0.01)"
)

print("\n")

header = (
    f"{'Thr':>5} "
    f"{'Coverage':>9} "
    f"{'Abstain':>9} "
    f"{'Acc':>9} "
    f"{'FN Disease':>11} "
    f"{'FP Healthy':>11} "
    f"{'TP':>7} "
    f"{'FN':>7} "
    f"{'FP':>7} "
    f"{'TN':>7}"
)

print(header)
print("-" * 155)


for _, row in results.iterrows():

    print(
        f"{row['threshold']:>5.2f} "
        f"{row['coverage'] * 100:>8.2f}% "
        f"{row['abstention_rate'] * 100:>8.2f}% "
        f"{row['accepted_accuracy'] * 100:>8.2f}% "
        f"{row['false_negative_disease_rate'] * 100:>10.2f}% "
        f"{row['false_positive_healthy_rate'] * 100:>10.2f}% "
        f"{int(row['true_positive_disease']):>7d} "
        f"{int(row['false_negative_disease']):>7d} "
        f"{int(row['false_positive_healthy']):>7d} "
        f"{int(row['true_negative_healthy']):>7d}"
    )


# ------------------------------------------------------------
# Useful operating points
# ------------------------------------------------------------

print("\n" + "=" * 155)
print("SELECTED OPERATING POINTS")
print("=" * 155)

for target in [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:

    row = results[
        np.isclose(
            results["threshold"],
            target,
        )
    ].iloc[0]

    print(
        f"\nThreshold {target:.2f}"
    )

    print(
        f"  Coverage              : "
        f"{row['coverage'] * 100:.2f}%"
    )

    print(
        f"  Abstention            : "
        f"{row['abstention_rate'] * 100:.2f}%"
    )

    print(
        f"  Accepted accuracy     : "
        f"{row['accepted_accuracy'] * 100:.2f}%"
    )

    print(
        f"  Disease FN rate       : "
        f"{row['false_negative_disease_rate'] * 100:.2f}%"
    )

    print(
        f"  Healthy FP rate       : "
        f"{row['false_positive_healthy_rate'] * 100:.2f}%"
    )

    print(
        f"  TP / FN / FP / TN     : "
        f"{row['true_positive_disease']} / "
        f"{row['false_negative_disease']} / "
        f"{row['false_positive_healthy']} / "
        f"{row['true_negative_healthy']}"
    )


# ------------------------------------------------------------
# Save location
# ------------------------------------------------------------

print("\n" + "=" * 155)

print(
    "\nDetailed CSV saved to:"
)

print(CSV_OUTPUT)

print(
    "\nImportant:"
)

print(
    "False-negative disease rate and false-positive healthy "
    "rate use the full actual disease/healthy populations "
    "as their denominators."
)

print(
    "Abstained images are not counted as correct or incorrect "
    "accepted predictions."
)

print("=" * 155)
