from pathlib import Path
from collections import Counter
from PIL import Image

ROOT = Path("dataset/tomato")

sizes = Counter()
formats = Counter()
extensions = Counter()

total = 0
valid = 0
invalid = 0

for path in ROOT.rglob("*"):
    if not path.is_file():
        continue

    total += 1
    extensions[path.suffix.lower()] += 1

    try:
        with Image.open(path) as image:
            image.verify()

        with Image.open(path) as image:
            sizes[image.size] += 1
            formats[image.format] += 1

        valid += 1

    except Exception:
        invalid += 1

print("=" * 70)
print("DISEASE DATASET IMAGE INSPECTION")
print("=" * 70)

print(f"Total files:       {total}")
print(f"Valid images:      {valid}")
print(f"Invalid images:    {invalid}")

print("\nFile extensions:")
for ext, count in extensions.most_common():
    print(f"{ext:<10} {count}")

print("\nImage formats:")
for fmt, count in formats.most_common():
    print(f"{fmt:<10} {count}")

print("\nTop image dimensions:")
for (width, height), count in sizes.most_common(20):
    print(f"{width:>5} x {height:<5}  {count}")

print("=" * 70)
