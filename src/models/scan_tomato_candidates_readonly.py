from pathlib import Path
import hashlib

CAPTURED = Path("captured_images")

HOLDOUT_HASHES = {
    "D8865898F99035EC24C5A20569AF31E3E4264FD20F2C8517575139794F042D59",
    "FB07E6D9B19BF275773C4CD9D59F074CC40DEFB9B7D78D88AD572BE3687ADCC3",
    "32303769CFCCD0A73EC56F2C1C30D9029154D0ACB0979D0D187346991F703388",
}

# Confirmed tomato images already established in this project
CONFIRMED_TOMATO_NAMES = {
    "20260925_102128_641390_Tomato_Test4.jpg",
    "20260925_130932_343311_tomato-test5.jpg",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}

unique = {}

for p in sorted(CAPTURED.rglob("*")):
    if not p.is_file() or p.suffix not in EXTENSIONS:
        continue

    h = hashlib.sha256(p.read_bytes()).hexdigest().upper()

    # Exclude every duplicate copy of the three locked holdouts
    if h in HOLDOUT_HASHES:
        continue

    # Keep only one copy of each unique image
    unique.setdefault(h, p)

candidates = []

for h, p in unique.items():

    name_lower = p.name.lower()

    # Previously confirmed tomato images
    # OR newly captured files whose filename explicitly indicates tomato
    if p.name in CONFIRMED_TOMATO_NAMES or "tomato" in name_lower:
        candidates.append((p.name, h))

candidates.sort(key=lambda x: x[0].lower())

print()
print("===== UNIQUE TOMATO CANDIDATES =====")
print(f"Unique captured images after holdout exclusion: {len(unique)}")
print(f"Tomato candidates: {len(candidates)}")
print()

for i, (name, h) in enumerate(candidates, 1):
    print(f"{i}. {name}")
    print(f"   SHA256: {h}")

print()
print("===== NOTE =====")
print("This is filename/confirmed-label based; it does not claim that ambiguous leaf images are tomato.")
print("No files were copied, moved, renamed, deleted, or modified.")
