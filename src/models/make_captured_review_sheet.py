from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw, ImageFont

MANIFEST = Path("results/captured_unique_candidates.csv")
OUTPUT = Path("results/captured_unique_review_sheet.jpg")

rows = list(csv.DictReader(MANIFEST.open("r", encoding="utf-8")))

thumb_w = 260
thumb_h = 220
cols = 4
rows_count = math.ceil(len(rows) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows_count * thumb_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()

captured_root = Path("captured_images")

for i, row in enumerate(rows):
    filename = row["representative_filename"]
    path = captured_root / filename

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((240, 170))

        x = (i % cols) * thumb_w
        y = (i // cols) * thumb_h

        img_x = x + (thumb_w - img.width) // 2
        img_y = y + 5

        sheet.paste(img, (img_x, img_y))

        label = f"{i + 1}. {filename}"

        # Wrap long filename
        if len(label) > 38:
            label = label[:35] + "..."

        draw.text(
            (x + 8, y + 180),
            label,
            fill="black",
            font=font
        )

    except Exception as e:
        print(f"Could not open {filename}: {e}")

sheet.save(OUTPUT, quality=95)

print(f"Created review sheet: {OUTPUT}")
print(f"Images shown: {len(rows)}")
