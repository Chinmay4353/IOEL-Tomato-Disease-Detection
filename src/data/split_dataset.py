"""Split the dataset into training, validation, and test sets."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

random.seed(42)


def split_dataset(dataset_root: str | Path, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Dict[str, List[str]]:
    """Split all class images into train/val/test using a simple deterministic approach."""
    dataset_path = Path(dataset_root)
    output = {"train": [], "val": [], "test": []}

    for class_name in ["tomato", "not_tomato"]:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            continue

        files = sorted(str(path) for path in class_dir.iterdir() if path.is_file())
        random.shuffle(files)

        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        output["train"].extend([f"{class_name}:{path}" for path in files[:train_end]])
        output["val"].extend([f"{class_name}:{path}" for path in files[train_end:val_end]])
        output["test"].extend([f"{class_name}:{path}" for path in files[val_end:]])

    return output


def main() -> None:
    dataset_root = Path(__file__).resolve().parents[2] / "dataset"
    split = split_dataset(dataset_root)
    for key, value in split.items():
        print(f"{key}: {len(value)}")


if __name__ == "__main__":
    main()
