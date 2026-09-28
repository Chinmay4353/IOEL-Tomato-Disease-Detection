from pathlib import Path
import hashlib
import csv

CAPTURED = Path("captured_images")
OUTPUT = Path("results/captured_image_manifest.csv")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)

HOLDOUT_HASHES = {
    "D8865898F99035EC24C5A20569AF31E3E4264FD20F2C8517575139794F042D59",
    "FB07E6D9B19BF275773C4CD9D59F074CC40DEFB9B7D78D88AD572BE3687ADCC3",
    "32303769CFCCD0A73EC56F2C1C30D9029154D0ACB0979D0D187346991F703388",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

rows = []

for path in sorted(CAPTURED.rglob("*")):
    if not path.is_file() or path.suffix not in IMAGE_EXTENSIONS:
        continue

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest().upper()

    rows.append({
        "filename": path.name,
        "relative_path": str(path.relative_to(CAPTURED)),
        "sha256": sha256,
        "is_holdout": sha256 in HOLDOUT_HASHES,
    })

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=["filename", "relative_path", "sha256", "is_holdout"]
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Captured images: {len(rows)}")
print(f"Protected holdout images: {sum(r['is_holdout'] for r in rows)}")
print(f"Manifest: {OUTPUT}")
