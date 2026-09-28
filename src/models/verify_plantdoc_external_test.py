from pathlib import Path
from PIL import Image

ROOT = Path("results/plantdoc_external_test")

total = 0
valid = 0
invalid = []

for path in sorted(ROOT.rglob("*")):
    if not path.is_file():
        continue

    total += 1

    try:
        with Image.open(path) as image:
            image.verify()
        valid += 1
    except Exception as exc:
        invalid.append((str(path), str(exc)))

print(f"Total files: {total}")
print(f"Valid images: {valid}")
print(f"Invalid images: {len(invalid)}")

for path, error in invalid:
    print(f"INVALID: {path} -> {error}")
