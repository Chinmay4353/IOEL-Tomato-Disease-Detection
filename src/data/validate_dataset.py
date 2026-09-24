"""Utilities for validating and summarizing the tomato dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List

from PIL import Image


def _sha256_of_file(file_path: Path) -> str:
    """Generate a hash for duplicate checking."""
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_root: str | Path) -> Dict[str, int]:
    """Validate dataset and report counts for each class and invalid/duplicate entries."""
    dataset_path = Path(dataset_root)
    tomato_dir = dataset_path / "tomato"
    not_tomato_dir = dataset_path / "not_tomato"

    summary = {
        "total_images": 0,
        "tomato_count": 0,
        "not_tomato_count": 0,
        "invalid_images": 0,
        "duplicate_images": 0,
    }

    seen_hashes: Dict[str, List[str]] = {}

    for class_dir in [tomato_dir, not_tomato_dir]:
        if not class_dir.exists():
            continue

        for file_path in sorted(class_dir.iterdir()):
            if not file_path.is_file():
                continue

            try:
                with Image.open(file_path) as image:
                    image.verify()
            except Exception:
                summary["invalid_images"] += 1
                continue

            try:
                with Image.open(file_path) as image:
                    image.load()
            except Exception:
                summary["invalid_images"] += 1
                continue

            digest = _sha256_of_file(file_path)
            if digest in seen_hashes:
                summary["duplicate_images"] += 1
                continue
            seen_hashes[digest] = [str(file_path)]

            if class_dir.name == "tomato":
                summary["tomato_count"] += 1
            elif class_dir.name == "not_tomato":
                summary["not_tomato_count"] += 1

            summary["total_images"] += 1

    summary["class_imbalance"] = summary["tomato_count"] - summary["not_tomato_count"]
    return summary


def main() -> None:
    dataset_root = Path(__file__).resolve().parents[2] / "dataset"
    result = validate_dataset(dataset_root)
    print("Dataset validation summary")
    print("-" * 30)
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
