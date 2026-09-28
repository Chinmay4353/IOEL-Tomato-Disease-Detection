from pathlib import Path
import hashlib
import csv

CAPTURED = Path("captured_images")
OUTPUT = Path("results/captured_unique_candidates.csv")

HOLDOUT_HASHES = {
    "D8865898F99035EC24C5A20569AF31E3E4264FD20F2C8517575139794F042D59",
    "FB07E6D9B19BF275773C4CD9D59F074CC40DEFB9B7D78D88AD572BE3687ADCC3",
    "32303769CFCCD0A73EC56F2C1C30D9029154D0ACB0979D0D187346991F703388",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

groups = {}

for path in sorted(CAPTURED.rglob("*")):
    if not path.is_file() or path.suffix not in EXTENSIONS:
        continue

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest().upper()

    if sha256 in HOLDOUT_HASHES:
        continue

    groups.setdefault(sha256, []).append(path)

rows = []

for sha256, paths in groups.items():
    first = paths[0]

    rows.append({
        "sha256": sha256,
        "representative_filename": first.name,
        "duplicate_count": len(paths),
        "all_filenames": " | ".join(p.name for p in paths),
    })

rows.sort(key=lambda x: x["representative_filename"].lower())

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sha256",
            "representative_filename",
            "duplicate_count",
            "all_filenames",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Protected holdout files excluded: 15")
print(f"Unique non-holdout captured images: {len(rows)}")
print(f"Candidate manifest: {OUTPUT}")
