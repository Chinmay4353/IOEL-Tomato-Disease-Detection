from pathlib import Path
import subprocess
import re

repo = Path(r".\plantdoc_source")
output = Path(r".\dataset\gate\plantdoc_tomato")
output.mkdir(parents=True, exist_ok=True)

paths = subprocess.check_output(
    ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"],
    text=True,
    encoding="utf-8"
).splitlines()

image_paths = [
    p for p in paths
    if re.search(r"\.(jpg|jpeg|png)$", p, re.IGNORECASE)
    and re.search(r"(^|/)tomato_", p, re.IGNORECASE)
]

for index, source_path in enumerate(image_paths, start=1):
    data = subprocess.check_output(
        ["git", "-C", str(repo), "cat-file", "blob", f"HEAD:{source_path}"]
    )

    suffix = Path(source_path).suffix.lower()
    destination = output / f"plantdoc_tomato_{index:03d}{suffix}"
    destination.write_bytes(data)

print(f"Extracted: {len(image_paths)} PlantDoc tomato images")
print(f"Output: {output}")
