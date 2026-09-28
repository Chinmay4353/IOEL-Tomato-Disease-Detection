from pathlib import Path
import random
import shutil

SOURCE = Path("dataset/tomato/train")
DEST = Path("dataset/gate/tomato")

SAMPLES_PER_CLASS = 78
SEED = 42

random.seed(SEED)
DEST.mkdir(parents=True, exist_ok=True)

classes = sorted(
    path for path in SOURCE.iterdir()
    if path.is_dir()
)

total = 0

for class_dir in classes:
    images = sorted(
        path for path in class_dir.iterdir()
        if path.is_file()
    )

    if len(images) < SAMPLES_PER_CLASS:
        raise ValueError(
            f"{class_dir.name} has only {len(images)} images."
        )

    selected = random.sample(images, SAMPLES_PER_CLASS)

    for index, source_path in enumerate(selected, start=1):
        destination_name = (
            f"{class_dir.name}__{index:03d}{source_path.suffix.lower()}"
        )
        destination = DEST / destination_name
        shutil.copy2(source_path, destination)

    print(f"{class_dir.name}: {len(selected)} images copied")
    total += len(selected)

print(f"\nTOTAL: {total} tomato images copied")
