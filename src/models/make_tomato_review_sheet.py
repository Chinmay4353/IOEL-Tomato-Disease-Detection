from pathlib import Path
from PIL import Image, ImageOps, ImageDraw

files = [
    Path(r".\captured_images\20260925_082705_560063_Tomato_Test.jpeg"),
    Path(r".\captured_images\20260925_084006_016700_Tomato_Test2.jpeg"),
    Path(r".\captured_images\20260925_085016_598359_tomato-test3.jpg"),
    Path(r".\captured_images\20260925_102128_641390_Tomato_Test4.jpg"),
    Path(r".\captured_images\20260925_130948_845524_tomato-test5.jpg"),
]

thumb_w, thumb_h = 420, 360
label_h = 45

sheet = Image.new("RGB", (thumb_w * 2, (thumb_h + label_h) * 3), "white")
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(files):
    image = Image.open(path).convert("RGB")
    image = ImageOps.contain(image, (thumb_w, thumb_h))

    x = (i % 2) * thumb_w
    y = (i // 2) * (thumb_h + label_h)

    sheet.paste(image, (x, y + label_h))
    draw.text((x + 5, y + 5), path.name, fill="black")

output = Path(r".\captured_images\unique_tomato_review.jpg")
sheet.save(output)

print(f"Created: {output}")
