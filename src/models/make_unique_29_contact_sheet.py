from pathlib import Path
import hashlib
import math
from PIL import Image, ImageDraw, ImageFont

CAPTURED = Path("captured_images")
OUTPUT = Path("results/unique_29_captured_review_sheet.jpg")

HOLDOUT_HASHES = {
    "D8865898F99035EC24C5A20569AF31E3E4264FD20F2C8517575139794F042D59",
    "FB07E6D9B19BF275773C4CD9D59F074CC40DEFB9B7D78D88AD572BE3687ADCC3",
    "32303769CFCCD0A73EC56F2C1C30D9029154D0ACB0979D0D187346991F703388",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

# Deduplicate and exclude all copies of the 3 locked holdouts
unique = {}

for path in sorted(CAPTURED.rglob("*")):
    if not path.is_file() or path.suffix not in EXTENSIONS:
        continue

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest().upper()

    if sha256 in HOLDOUT_HASHES:
        continue

    unique.setdefault(sha256, path)

images = sorted(
    unique.values(),
    key=lambda p: p.name.lower()
)

print(f"Unique images after holdout exclusion: {len(images)}")

# Contact sheet settings
thumb_w = 420
thumb_h = 300
cols = 3
rows = math.ceil(len(images) / cols)

sheet = Image.new(
    "RGB",
    (cols * thumb_w, rows * thumb_h),
    "white"
)

draw = ImageDraw.Draw(sheet)

try:
    font = ImageFont.truetype("arial.ttf", 16)
    small_font = ImageFont.truetype("arial.ttf", 14)
except:
    font = ImageFont.load_default()
    small_font = font

for i, path in enumerate(images):

    x = (i % cols) * thumb_w
    y = (i // cols) * thumb_h

    try:
        img = Image.open(path).convert("RGB")
        img.thumbnail((395, 235))

        img_x = x + (thumb_w - img.width) // 2
        img_y = y + 5

        sheet.paste(img, (img_x, img_y))

        # Number
        draw.text(
            (x + 10, y + 245),
            f"{i + 1}.",
            fill="black",
            font=font
        )

        # Filename
        filename = path.name

        # Wrap long filenames
        max_chars = 48
        if len(filename) > max_chars:
            filename = filename[:max_chars - 3] + "..."

        draw.text(
            (x + 35, y + 245),
            filename,
            fill="black",
            font=small_font
        )

    except Exception as e:
        draw.text(
            (x + 10, y + 10),
            f"ERROR: {path.name}",
            fill="red",
            font=font
        )
        print(f"Could not open {path}: {e}")

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
sheet.save(OUTPUT, quality=95)

print(f"Contact sheet: {OUTPUT}")
print("No captured_images files were modified.")
