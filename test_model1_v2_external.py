from pathlib import Path
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = Path("models/model1_v2_stage2_best.keras")
DOWNLOADS = Path.home() / "Downloads"
THRESHOLD = 0.20

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

tomato_images = sorted(DOWNLOADS.glob("tomato_plant_*.jpg"))
non_tomato_images = sorted(DOWNLOADS.glob("non_tomato*.jpg"))

correct = 0
total = 0

print("=" * 75)
print("MODEL 1 V2 - TOMATO / NON-TOMATO EXTERNAL TEST")
print("=" * 75)
print(f"Model:     {MODEL_PATH}")
print(f"Threshold: {THRESHOLD}")
print("Preprocess: RGB -> 160x160 -> float32 / 255.0")
print("=" * 75)

for expected, images in [
    ("TOMATO", tomato_images),
    ("NOT_TOMATO", non_tomato_images),
]:
    print(f"\nEXPECTED: {expected}")
    print("-" * 75)

    for path in images:
        with Image.open(path) as image:
            original_size = image.size
            image = image.convert("RGB")
            image = image.resize((160, 160))
            array = np.asarray(image, dtype=np.float32) / 255.0

        x = np.expand_dims(array, axis=0)

        probability = float(model.predict(x, verbose=0)[0][0])
        prediction = (
            "TOMATO"
            if probability >= THRESHOLD
            else "NOT_TOMATO"
        )

        passed = prediction == expected

        if passed:
            correct += 1

        total += 1

        print(f"{path.name}")
        print(f"  Original size : {original_size}")
        print(f"  Probability   : {probability:.4f} ({probability * 100:.2f}%)")
        print(f"  Prediction    : {prediction}")
        print(f"  Expected      : {expected}")
        print(f"  Result        : {'PASS' if passed else 'FAIL'}")
        print()

print("=" * 75)
print("SUMMARY")
print("=" * 75)
print(f"Tomato images tested     : {len(tomato_images)}")
print(f"Non-tomato images tested : {len(non_tomato_images)}")
print(f"Total tested              : {total}")
print(f"Correct                   : {correct}")
print(
    f"External accuracy        : "
    f"{(correct / total * 100):.2f}%"
    if total else
    "External accuracy        : N/A"
)
print("=" * 75)
