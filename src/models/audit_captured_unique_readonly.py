from pathlib import Path
import hashlib

CAPTURED = Path("captured_images")

HOLDOUT_HASHES = {
    "D8865898F99035EC24C5A20569AF31E3E4264FD20F2C8517575139794F042D59",
    "FB07E6D9B19BF275773C4CD9D59F074CC40DEFB9B7D78D88AD572BE3687ADCC3",
    "32303769CFCCD0A73EC56F2C1C30D9029154D0ACB0979D0D187346991F703388",
}

EXTENSIONS = {".jpg", ".jpeg", ".png"}

unique = {}

total = 0
holdout_files = 0

for path in sorted(CAPTURED.rglob("*")):
    if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
        continue

    total += 1

    sha256 = hashlib.sha256(path.read_bytes()).hexdigest().upper()

    if sha256 in HOLDOUT_HASHES:
        holdout_files += 1
        continue

    # Keep one representative file for each unique image
    unique.setdefault(sha256, path)

print()
print("===== CAPTURED IMAGE AUDIT =====")
print(f"Total image files scanned: {total}")
print(f"Protected holdout files excluded: {holdout_files}")
print(f"Unique images after holdout exclusion: {len(unique)}")
print()

for i, (sha256, path) in enumerate(
    sorted(unique.items(), key=lambda x: x[1].name.lower()),
    start=1
):
    print(f"{i}. {path.name}")
    print(f"   Path: {path}")
    print(f"   SHA256: {sha256}")
    print()

print("===== SAFETY CHECK =====")
print("Read-only audit completed.")
print("No files were copied, moved, renamed, deleted, or modified.")
