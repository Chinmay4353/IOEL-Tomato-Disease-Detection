from pathlib import Path
import random
import shutil

SOURCE = Path("dataset/gate")
DEST = Path("dataset/gate_split")

CLASSES = ["tomato", "not_tomato"]

TRAIN_RATIO = 0.80
SEED = 42

random.seed(SEED)

for class_name in CLASSES:
    source_dir = SOURCE / class_name

    images = [
        p for p in source_dir.iterdir()
        if p.is_file()
    ]

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    train_dir = DEST / "train" / class_name
    val_dir = DEST / "validation" / class_name

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    for image in train_images:
        shutil.copy2(image, train_dir / image.name)

    for image in val_images:
        shutil.copy2(image, val_dir / image.name)

    print(f"{class_name}:")
    print(f"  Train: {len(train_images)}")
    print(f"  Validation: {len(val_images)}")

print("\nSplit complete.")
