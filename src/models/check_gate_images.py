from pathlib import Path
from PIL import Image

roots = [
    Path("dataset/gate/tomato"),
    Path("dataset/gate/not_tomato"),
]

total = 0
valid = 0
bad = []

for root in roots:
    for path in root.rglob("*"):
        if not path.is_file():
            continue

        total += 1

        try:
            with Image.open(path) as image:
                image.verify()
            valid += 1
        except Exception as e:
            bad.append((str(path), str(e)))

print(f"Total checked: {total}")
print(f"Valid images: {valid}")
print(f"Invalid images: {len(bad)}")

if bad:
    print("\nInvalid files:")
    for path, error in bad[:20]:
        print(path, "->", error)
