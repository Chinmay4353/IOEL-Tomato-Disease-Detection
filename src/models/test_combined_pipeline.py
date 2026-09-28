from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.combined_inference import TomatoDiseasePipeline


DATASET = Path("dataset/tomato/valid")

MAX_IMAGES_PER_CLASS = 3


def main():
    pipeline = TomatoDiseasePipeline()

    total = 0
    gate_correct = 0
    disease_correct = 0
    accepted = 0
    uncertain = 0

    print("\n" + "=" * 70)
    print("COMBINED PIPELINE TEST")
    print("=" * 70)

    for class_dir in sorted(DATASET.iterdir()):

        if not class_dir.is_dir():
            continue

        images = sorted(
            p for p in class_dir.iterdir()
            if p.is_file()
        )[:MAX_IMAGES_PER_CLASS]

        for image_path in images:

            total += 1

            result = pipeline.predict(image_path)

            gate_ok = result.get("is_tomato", False)

            if gate_ok:
                gate_correct += 1

            predicted = result.get("prediction")

            disease_ok = predicted == class_dir.name

            if disease_ok:
                disease_correct += 1

            if result.get("status") == "ACCEPTED":
                accepted += 1
            elif result.get("status") == "UNCERTAIN":
                uncertain += 1

            print(
                f"{total:02d}. "
                f"ACTUAL={class_dir.name:<35} "
                f"PRED={str(predicted):<35} "
                f"CONF={result.get('confidence', 0):.3f} "
                f"STATUS={result.get('status'):<10} "
                f"{'OK' if disease_ok else 'WRONG'}"
            )

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"Images tested:       {total}")
    print(f"Gate accepted:       {gate_correct}/{total}")
    print(f"Disease correct:     {disease_correct}/{total}")
    print(f"Disease accuracy:    {disease_correct / total * 100:.2f}%")
    print(f"Accepted:            {accepted}/{total}")
    print(f"Uncertain:           {uncertain}/{total}")

    print("=" * 70)


if __name__ == "__main__":
    main()
