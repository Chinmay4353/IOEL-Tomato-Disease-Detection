from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path("results/plantdoc_external_test")
OUTPUT = Path("results/plantdoc_external_contact_sheet.jpg")

classes = sorted([p for p in ROOT.iterdir() if p.is_dir()])

thumb_w = 220
thumb_h = 220
label_h = 45
cols = 3

rows = sum((min(4, len(list(c.iterdir()))) + cols - 1) // cols for c in classes)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * (thumb_h + label_h)),
    "white"
)

draw = ImageDraw.Draw(sheet)

y = 0

for class_dir in classes:
    images = sorted([p for p in class_dir.iterdir() if p.is_file()])[:4]

    for i, path in enumerate(images):
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((thumb_w - 10, thumb_h - 10))

            x = (i % cols) * thumb_w
            cell_y = y

            px = x + (thumb_w - image.width) // 2
            py = cell_y + (thumb_h - image.height) // 2

            sheet.paste(image, (px, py))

            draw.text(
                (x + 5, cell_y + thumb_h),
                f"{class_dir.name} #{i + 1}",
                fill="black"
            )

    y += ((len(images) + cols - 1) // cols) * (thumb_h + label_h)

sheet.save(OUTPUT, quality=90)

print(f"Created: {OUTPUT.resolve()}")
