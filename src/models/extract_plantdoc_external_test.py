from pathlib import Path
import re
import subprocess
import urllib.parse
import xml.etree.ElementTree as ET

REPO = Path("plantdoc_source")
OUTPUT = Path("results/plantdoc_external_test")

LABEL_MAP = {
    "Tomato Early blight leaf": "Early_blight",
    "Tomato leaf bacterial spot": "Bacterial_spot",
    "Tomato leaf late blight": "Late_blight",
    "Tomato leaf mosaic virus": "Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato Septoria leaf spot": "Septoria_leaf_spot",
}

def git_show(path):
    result = subprocess.run(
        ["git", "-C", str(REPO), "show", f"HEAD:{path}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=True,
    )
    return result.stdout

def get_label(xml_bytes):
    root = ET.fromstring(xml_bytes)

    names = [
        node.text.strip()
        for node in root.findall(".//object/name")
        if node.text
    ]

    return names[0] if names else None

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    tree = subprocess.run(
        [
            "git",
            "-C",
            str(REPO),
            "-c",
            "core.quotepath=false",
            "ls-tree",
            "-r",
            "--name-only",
            "HEAD",
        ],
        stdout=subprocess.PIPE,
        text=True,
        check=True,
    )

    xml_paths = [
        line.strip()
        for line in tree.stdout.splitlines()
        if re.match(
            r"data/origin/TEST/.*\.xml$",
            line.strip(),
            re.IGNORECASE,
        )
    ]

    counts = {name: 0 for name in LABEL_MAP.values()}
    skipped = 0

    for xml_path in xml_paths:
        xml_bytes = git_show(xml_path)
        label = get_label(xml_bytes)

        if label not in LABEL_MAP:
            continue

        output_class = LABEL_MAP[label]

        image_path = re.sub(
            r"\.xml$",
            ".jpg",
            xml_path,
            flags=re.IGNORECASE,
        )

        # Some PlantDoc filenames contain URL-style encoding.
        # Git's tree path is the authoritative path, so first try
        # the exact sibling path.
        try:
            image_bytes = git_show(image_path)
        except subprocess.CalledProcessError:
            # Handle encoded extension/name variants.
            decoded_path = urllib.parse.unquote(image_path)

            try:
                image_bytes = git_show(decoded_path)
            except subprocess.CalledProcessError:
                print(f"SKIP: image not found for {xml_path}")
                skipped += 1
                continue

        destination_dir = OUTPUT / output_class
        destination_dir.mkdir(parents=True, exist_ok=True)

        original_name = Path(image_path).name
        safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", original_name)

        destination = destination_dir / safe_name

        if destination.exists():
            destination = destination_dir / f"{counts[output_class]:03d}_{safe_name}"

        destination.write_bytes(image_bytes)
        counts[output_class] += 1

    print("\nExtraction complete.")
    print("-" * 50)

    total = 0

    for class_name, count in counts.items():
        print(f"{class_name:35s} {count:3d}")
        total += count

    print("-" * 50)
    print(f"{'Total extracted':35s} {total:3d}")
    print(f"{'Skipped images':35s} {skipped:3d}")
    print(f"\nOutput: {OUTPUT.resolve()}")

if __name__ == "__main__":
    main()
