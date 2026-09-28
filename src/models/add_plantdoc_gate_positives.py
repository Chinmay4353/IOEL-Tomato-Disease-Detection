from pathlib import Path
import shutil

source = Path(r".\dataset\gate\plantdoc_tomato")
target = Path(r".\dataset\gate\tomato")

files = sorted(
    [
        f for f in source.iterdir()
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
)

copied = 0

for source_file in files:
    destination = target / f"plantdoc_{source_file.name}"
    if not destination.exists():
        shutil.copy2(source_file, destination)
        copied += 1

print(f"PlantDoc images found: {len(files)}")
print(f"New images copied: {copied}")
print(f"Gate tomato images now: {len(list(target.iterdir()))}")
print(f"Gate not_tomato images: {len(list((Path(r'.\dataset\gate\not_tomato')).iterdir()))}")
