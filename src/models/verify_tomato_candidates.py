from pathlib import Path
import csv
from PIL import Image, ImageDraw, ImageFont

MANIFEST = Path("results/realworld_labeling.csv")
OUTPUT = Path("results/proposed_tomato_verification.jpg")
CAPTURED = Path("captured_images")

with MANIFEST.open("r", encoding="utf-8") as f:
    rows = [
        r for r in csv.DictReader(f)
        if r["label"] == "TOMATO"
    ]

thumb_w = 500
thumb_h = 390
cols = 2
rows_count = (len(rows) + cols - 1) // cols

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows_count * thumb_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("arial.ttf", 18)
except:
    font = ImageFont.load_default()

for i, row in enumerate(rows):

    filename = row["filename"]
    matches = list(CAPTURED.rglob(filename))

    if not matches:
        print("Missing:", filename)
        continue

    path = matches[0]

    img = Image.open(path).convert("RGB")
    img.thumbnail((470, 315))

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    img_x = x + (thumb_w - img.width) // 2
    img_y = y + 5

    sheet.paste(img, (img_x, img_y))

    label = f"{row['index']}. {filename}"

    if len(label) > 55:
        label = label[:52] + "..."

    draw.text(
        (x + 12, y + 330),
        label,
        fill="black",
        font=font
    )

sheet.save(OUTPUT, quality=95)

print(f"Proposed tomato images: {len(rows)}")
print(f"Verification sheet: {OUTPUT}")
