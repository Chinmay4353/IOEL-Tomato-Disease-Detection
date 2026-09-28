from pathlib import Path

DATASET_ROOT = Path("dataset/tomato")

splits = ["train", "valid"]

print("=" * 75)
print("TOMATO DISEASE DATASET DISTRIBUTION")
print("=" * 75)

grand_total = 0

for split in splits:
    split_dir = DATASET_ROOT / split

    print(f"\n{split.upper()}")
    print("-" * 75)

    total = 0
    class_counts = []

    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue

        count = sum(
            1
            for p in class_dir.iterdir()
            if p.is_file()
        )

        class_counts.append((class_dir.name, count))
        total += count

    for class_name, count in class_counts:
        percentage = (count / total * 100) if total else 0
        print(f"{class_name:<45} {count:>6}  ({percentage:>6.2f}%)")

    print("-" * 75)
    print(f"{'TOTAL':<45} {total:>6}")

    grand_total += total

print("\n" + "=" * 75)
print(f"GRAND TOTAL: {grand_total}")
print("=" * 75)
