from pathlib import Path
import hashlib
import random
import shutil

SOURCE = Path("dataset/gate")
OUTPUT = Path("dataset/gate_split_clean")

SEED = 42
VAL_RATIO = 0.20


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_unique_images(folder):
    groups = {}

    for path in folder.rglob("*"):
        if not path.is_file():
            continue

        digest = sha256(path)
        groups.setdefault(digest, []).append(path)

    return groups


def split_hash_groups(groups, val_ratio):
    items = list(groups.items())

    rng = random.Random(SEED)
    rng.shuffle(items)

    val_count = round(len(items) * val_ratio)

    val_groups = items[:val_count]
    train_groups = items[val_count:]

    return train_groups, val_groups


def copy_groups(groups, destination):
    destination.mkdir(parents=True, exist_ok=True)

    copied = 0

    for digest, paths in groups:
        # Keep exactly one physical copy of each unique image.
        source = paths[0]

        # Preserve a stable unique filename.
        filename = f"{digest[:16]}_{source.name}"
        target = destination / filename

        shutil.copy2(source, target)
        copied += 1

    return copied


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    if OUTPUT.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT}\n"
            "Delete/rename it manually only after reviewing the result."
        )

    print("===== HASH-BASED GATE SPLIT REBUILD =====")
    print(f"Source: {SOURCE}")
    print(f"Output: {OUTPUT}")
    print()

    for class_name in ["tomato", "not_tomato"]:
        source_class = SOURCE / class_name

        if not source_class.exists():
            raise FileNotFoundError(
                f"Missing class folder: {source_class}"
            )

        groups = collect_unique_images(source_class)

        print(f"{class_name}:")
        print(f"  Unique SHA-256 images: {len(groups)}")
        print(f"  Physical files: {sum(len(v) for v in groups.values())}")

        duplicate_files = sum(len(v) - 1 for v in groups.values())

        print(f"  Duplicate physical files: {duplicate_files}")

        train_groups, val_groups = split_hash_groups(
            groups,
            VAL_RATIO
        )

        train_dir = OUTPUT / "train" / class_name
        val_dir = OUTPUT / "validation" / class_name

        train_count = copy_groups(train_groups, train_dir)
        val_count = copy_groups(val_groups, val_dir)

        print(f"  Train unique images: {train_count}")
        print(f"  Validation unique images: {val_count}")
        print()

    print("===== REBUILD COMPLETE =====")
    print(f"New dataset created at: {OUTPUT}")
    print()
    print("IMPORTANT:")
    print("- Original dataset/gate was NOT modified.")
    print("- Existing dataset/gate_split was NOT modified.")
    print("- The new split uses SHA-256 groups.")
    print("- Exact duplicate images cannot cross train/validation.")


if __name__ == "__main__":
    main()