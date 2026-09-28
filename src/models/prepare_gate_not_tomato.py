from pathlib import Path
import json
import random
import subprocess

REPO = Path("plantdoc_source")
DEST = Path("dataset/gate/not_tomato")

ANNOTATION = "annotations/plant_diseases_instances_train.json"
SAMPLES = 858
SEED = 42

SELECTED_PREFIXES = (
    "apple_",
    "bell_pepper_",
    "corn_",
    "grape_",
    "potato_",
    "strawberry_",
)

DEST.mkdir(parents=True, exist_ok=True)

print("Reading PlantDoc annotations...")

raw = subprocess.check_output(
    ["git", "-C", str(REPO), "show", f"HEAD:{ANNOTATION}"],
    encoding="utf-8",
)

data = json.loads(raw)

selected_categories = {
    category["id"]: category["name"]
    for category in data["categories"]
    if category["name"].startswith(SELECTED_PREFIXES)
}

selected_ids = set(selected_categories)

annotations_by_image = {}

for annotation in data["annotations"]:
    if annotation["category_id"] in selected_ids:
        image_id = annotation["image_id"]
        annotations_by_image.setdefault(image_id, set()).add(
            annotation["category_id"]
        )

images_by_id = {
    image["id"]: image
    for image in data["images"]
}

candidate_images = []

for image_id, category_ids in annotations_by_image.items():
    image = images_by_id.get(image_id)

    if image:
        candidate_images.append({
            "id": image_id,
            "file_name": image["file_name"],
            "categories": [
                selected_categories[c]
                for c in sorted(category_ids)
            ],
        })

print(f"Candidate non-tomato images: {len(candidate_images)}")

if len(candidate_images) < SAMPLES:
    raise RuntimeError(
        f"Only {len(candidate_images)} candidates found; "
        f"{SAMPLES} required."
    )

random.seed(SEED)
selected = random.sample(candidate_images, SAMPLES)

print(f"Selected images: {len(selected)}")
print("Downloading selected images...")

for index, item in enumerate(selected, start=1):

    source_path = item["file_name"]
    suffix = Path(source_path).suffix.lower()

    if suffix not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        suffix = ".jpg"

    destination = DEST / f"plantdoc_{index:04d}{suffix}"

    result = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "cat-file",
            "blob",
            f"HEAD:{source_path}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )

    if result.returncode != 0:
        print(
            f"WARNING: failed to retrieve image {index}: "
            f"{source_path}"
        )
        print(result.stderr.decode("utf-8", errors="replace"))
        continue

    destination.write_bytes(result.stdout)

    if index % 50 == 0 or index == SAMPLES:
        print(f"Downloaded {index}/{SAMPLES}")

print("\nExtraction complete.")

actual = len(list(DEST.glob("*")))

print(f"Files in destination: {actual}")
print(f"Destination: {DEST}")
