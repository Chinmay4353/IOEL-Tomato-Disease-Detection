from pathlib import Path

from src.inference.tflite_predict import predict_tflite_image


classes = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites Two-spotted_spider_mite",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "healthy",
    "powdery_mildew",
]

root = Path("dataset/tomato/valid")

correct = 0
total = 0

print("\nMULTI-CLASS TFLITE TEST")
print("=" * 80)

for class_name in classes:
    class_dir = root / class_name

    image = next(
        p
        for p in class_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    result = predict_tflite_image(image_path=image)

    is_correct = result["prediction"] == class_name

    if is_correct:
        correct += 1

    total += 1

    status = "CORRECT" if is_correct else "WRONG"

    print(
        f"{class_name:40} -> "
        f"{result['prediction']:40} "
        f"{result['confidence'] * 100:6.2f}% "
        f"[{status}]"
    )

print("=" * 80)
print(f"Correct: {correct}/{total}")
print(f"Accuracy on sampled images: {correct / total * 100:.2f}%")