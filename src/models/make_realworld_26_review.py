from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw, ImageFont

MANIFEST = Path("results/realworld_labeling.csv")
OUTPUT = Path("results/realworld_26_review_sheet.jpg")
CAPTURED = Path("captured_images")

with MANIFEST.open("r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

thumb_w = 360
thumb_h = 260
cols = 3
rows_count = math.ceil(len(rows) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows_count * thumb_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("arial.ttf", 15)
except:
    font = ImageFont.load_default()

for i, row in enumerate(rows):

    filename = row["filename"]

    matches = list(CAPTURED.rglob(filename))

    if not matches:
        print(f"Could not find: {filename}")
        continue

    path = matches[0]

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((340, 205))

        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h

        img_x = x + (thumb_w - img.width) // 2
        img_y = y + 5

        sheet.paste(img, (img_x, img_y))

        label = f"{row['index']}. {filename}"

        # Wrap long filenames
        if len(label) > 48:
            label = label[:45] + "..."

        draw.text(
            (x + 10, y + 220),
            label,
            fill="black",
            font=font
        )

    except Exception as e:
        print(f"Error opening {filename}: {e}")

sheet.save(OUTPUT, quality=95)

print(f"Review sheet created: {OUTPUT}")
print(f"Images shown: {len(rows)}")
