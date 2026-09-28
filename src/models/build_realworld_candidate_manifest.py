from pathlib import Path
import hashlib
import csv

CAPTURED = Path("captured_images")
GATE = Path("dataset/gate")
OUTPUT = Path("results/realworld_training_candidates.csv")

HOLDOUT_HASHES = {
    "D8865898F99035EC24C5A20569AF31E3E4264FD20F2C8517575139794F042D59",
    "FB07E6D9B19BF275773C4CD9D59F074CC40DEFB9B7D78D88AD572BE3687ADCC3",
    "32303769CFCCD0A73EC56F2C1C30D9029154D0ACB0979D0D187346991F703388",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Hash existing gate images
gate_hashes = {}

for path in GATE.rglob("*"):
    if path.is_file() and path.suffix in EXTENSIONS:
        h = hashlib.sha256(path.read_bytes()).hexdigest().upper()
        gate_hashes[h] = str(path)

# Deduplicate captured images
captured = {}

for path in sorted(CAPTURED.rglob("*")):
    if not path.is_file() or path.suffix not in EXTENSIONS:
        continue

    h = hashlib.sha256(path.read_bytes()).hexdigest().upper()

    # Never consider protected holdout images as training candidates
    if h in HOLDOUT_HASHES:
        continue

    captured.setdefault(h, path)

rows = []

for h, path in captured.items():

    if h in gate_hashes:
        status = "ALREADY_IN_GATE"
        gate_path = gate_hashes[h]
    else:
        status = "NEW_REALWORLD"
        gate_path = ""

    rows.append({
        "filename": path.name,
        "path": str(path),
        "sha256": h,
        "status": status,
        "gate_existing_path": gate_path,
        "label": "",
        "notes": "",
    })

rows.sort(key=lambda x: x["filename"].lower())

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "path",
            "sha256",
            "status",
            "gate_existing_path",
            "label",
            "notes",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Total unique captured candidates: {len(rows)}")
print(
    "Already in gate:",
    sum(r["status"] == "ALREADY_IN_GATE" for r in rows)
)
print(
    "New real-world images:",
    sum(r["status"] == "NEW_REALWORLD" for r in rows)
)
print(f"Manifest: {OUTPUT}")
