from pathlib import Path
from datetime import datetime, timezone
import csv
import hashlib
import sys

import numpy as np
import tensorflow as tf
from PIL import Image


MODEL_PATH = Path(r".\models\tomato_gate_finetuned_best.keras")
CSV_PATH = Path(r".\results\gate_realworld_holdout.csv")

THRESHOLD = 0.48
IMG_SIZE = (160, 160)

IMAGES = [
    Path(r".\captured_images\20260925_082705_560063_Tomato_Test.jpeg"),
    Path(r".\captured_images\20260925_084006_016700_Tomato_Test2.jpeg"),
    Path(r".\captured_images\20260925_085016_598359_tomato-test3.jpg"),
]


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def predict(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMG_SIZE)

    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)

    probability = float(model.predict(array, verbose=0)[0][0])

    return probability


def main():
    run_label = sys.argv[1] if len(sys.argv) > 1 else "before"

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    missing = [str(path) for path in IMAGES if not path.exists()]

    if missing:
        raise FileNotFoundError(
            "Missing holdout image(s):\n" + "\n".join(missing)
        )

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("REAL-WORLD TOMATO GATE HOLDOUT TEST")
    print("=" * 70)
    print(f"Run label : {run_label}")
    print(f"Model     : {MODEL_PATH}")
    print(f"Threshold : {THRESHOLD}")
    print()

    model = tf.keras.models.load_model(MODEL_PATH)

    timestamp = datetime.now(timezone.utc).isoformat()

    rows = []

    for image_path in IMAGES:
        probability = predict(model, image_path)

        observed = (
            "TOMATO"
            if probability >= THRESHOLD
            else "NOT_TOMATO"
        )

        expected = "TOMATO"

        result = (
            "PASS"
            if observed == expected
            else "FAIL"
        )

        rows.append({
            "run_label": run_label,
            "timestamp_utc": timestamp,
            "image_id": image_path.name,
            "sha256": sha256(image_path),
            "model_path": str(MODEL_PATH),
            "threshold": THRESHOLD,
            "tomato_probability": round(probability, 6),
            "tomato_probability_percent": round(
                probability * 100, 2
            ),
            "observed_output": observed,
            "expected_output": expected,
            "result": result,
        })

        print(
            f"{image_path.name}\n"
            f"  Probability : {probability:.4f} "
            f"({probability * 100:.2f}%)\n"
            f"  Observed    : {observed}\n"
            f"  Expected    : {expected}\n"
            f"  Result      : {result}\n"
        )

    fieldnames = [
        "run_label",
        "timestamp_utc",
        "image_id",
        "sha256",
        "model_path",
        "threshold",
        "tomato_probability",
        "tomato_probability_percent",
        "observed_output",
        "expected_output",
        "result",
    ]

    file_exists = CSV_PATH.exists()

    with CSV_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

    print("=" * 70)
    print(f"Saved results to: {CSV_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
