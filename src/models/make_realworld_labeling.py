from pathlib import Path
import csv

INPUT = Path("results/realworld_training_candidates.csv")
OUTPUT = Path("results/realworld_labeling.csv")

with INPUT.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

new_rows = [
    r for r in rows
    if r["status"] == "NEW_REALWORLD"
]

output_rows = []

for i, row in enumerate(new_rows, start=1):
    output_rows.append({
        "index": i,
        "filename": row["filename"],
        "sha256": row["sha256"],
        "label": "",
        "notes": "",
    })

fields = [
    "index",
    "filename",
    "sha256",
    "label",
    "notes",
]

with OUTPUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output_rows)

print(f"New real-world candidates: {len(output_rows)}")
print(f"Labeling file: {OUTPUT}")
