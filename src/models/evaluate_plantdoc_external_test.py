from pathlib import Path
import json
import csv

import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

MODEL_PATH = Path("models/tomato_disease_finetuned_best.keras")
CLASS_PATH = Path("models/tomato_disease_classes.json")
DATASET_PATH = Path("results/plantdoc_external_test")
RESULTS_PATH = Path("results/plantdoc_external_test_results.csv")

IMG_SIZE = (160, 160)

FOLDER_TO_CLASS = {
    "Bacterial_spot": "Bacterial_spot",
    "Early_blight": "Early_blight",
    "Late_blight": "Late_blight",
    "Septoria_leaf_spot": "Septoria_leaf_spot",
    "Tomato_mosaic_virus": "Tomato_mosaic_virus",
    "Tomato_Yellow_Leaf_Curl_Virus": "Tomato_Yellow_Leaf_Curl_Virus",
}

THRESHOLDS = [0.80, 0.85]


def load_image(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize(IMG_SIZE)
        array = np.asarray(image, dtype=np.float32)

    return array


def main():
    print("Loading class mapping...")
    class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))

    print("Classes:")
    for index, name in enumerate(class_names):
        print(f"  {index}: {name}")

    print("\nLoading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    images = []
    true_names = []
    paths = []

    for folder_name, class_name in FOLDER_TO_CLASS.items():
        folder = DATASET_PATH / folder_name

        if not folder.exists():
            raise FileNotFoundError(f"Missing folder: {folder}")

        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue

            images.append(load_image(path))
            true_names.append(class_name)
            paths.append(path)

    x = np.asarray(images, dtype=np.float32)

    true_indices = np.array(
        [class_names.index(name) for name in true_names],
        dtype=np.int64,
    )

    print(f"\nExternal test images: {len(x)}")

    probabilities = model.predict(x, batch_size=16, verbose=1)

    predicted_indices = np.argmax(probabilities, axis=1)
    max_confidences = np.max(probabilities, axis=1)

    raw_accuracy = accuracy_score(true_indices, predicted_indices)

    print("\n" + "=" * 70)
    print("RAW EXTERNAL TEST RESULT")
    print("=" * 70)
    print(f"Images:   {len(x)}")
    print(f"Accuracy: {raw_accuracy:.4f} ({raw_accuracy * 100:.2f}%)")

    print("\nClassification report:")
    print(
        classification_report(
            true_indices,
            predicted_indices,
            labels=sorted(set(true_indices.tolist())),
            target_names=[
                class_names[i]
                for i in sorted(set(true_indices.tolist()))
            ],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    used_indices = sorted(set(true_indices.tolist()))
    cm = confusion_matrix(
        true_indices,
        predicted_indices,
        labels=used_indices,
    )

    print("Rows = actual, columns = predicted")
    print("Labels:")
    for i in used_indices:
        print(f"  {i}: {class_names[i]}")

    print(cm)

    print("\n" + "=" * 70)
    print("THRESHOLD RESULTS")
    print("=" * 70)

    rows = []

    for threshold in THRESHOLDS:
        accepted = max_confidences >= threshold

        coverage = float(np.mean(accepted))
        abstention = 1.0 - coverage

        accepted_count = int(np.sum(accepted))

        if accepted_count > 0:
            accepted_accuracy = accuracy_score(
                true_indices[accepted],
                predicted_indices[accepted],
            )
        else:
            accepted_accuracy = 0.0

        print(f"\nThreshold: {threshold:.2f}")
        print(f"Accepted:       {accepted_count}/{len(x)}")
        print(f"Coverage:       {coverage * 100:.2f}%")
        print(f"Abstention:     {abstention * 100:.2f}%")
        print(f"Accepted acc.:  {accepted_accuracy * 100:.2f}%")

        rows.append({
            "threshold": threshold,
            "total_images": len(x),
            "accepted_images": accepted_count,
            "coverage": coverage,
            "abstention": abstention,
            "accepted_accuracy": accepted_accuracy,
        })

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with RESULTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved threshold results to: {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
