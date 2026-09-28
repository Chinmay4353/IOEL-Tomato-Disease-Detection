from pathlib import Path
import hashlib

CAPTURED = Path("captured_images")
GATE = Path("dataset/gate")

HOLDOUT_HASHES = {
    "D8865898F99035EC24C5A20569AF31E3E4264FD20F2C8517575139794F042D59",
    "FB07E6D9B19BF275773C4CD9D59F074CC40DEFB9B7D78D88AD572BE3687ADCC3",
    "32303769CFCCD0A73EC56F2C1C30D9029154D0ACB0979D0D187346991F703388",
}

CONFIRMED_TOMATO_NAMES = {
    "20260925_102128_641390_Tomato_Test4.jpg",
    "20260925_130932_343311_tomato-test5.jpg",
}

EXTENSIONS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


# Hash everything already in the gate
gate_hashes = set()

for p in GATE.rglob("*"):
    if p.is_file() and p.suffix in EXTENSIONS:
        gate_hashes.add(sha256(p))


# Deduplicate captured images
unique_captured = {}

for p in sorted(CAPTURED.rglob("*")):
    if not p.is_file() or p.suffix not in EXTENSIONS:
        continue

    h = sha256(p)

    if h not in unique_captured:
        unique_captured[h] = p


# Apply protections
eligible = []

for h, p in unique_captured.items():

    # Locked holdout
    if h in HOLDOUT_HASHES:
        continue

    # Already part of gate dataset
    if h in gate_hashes:
        continue

    eligible.append((h, p))


# Confirmed real-world tomato candidates
tomato_candidates = [
    (h, p)
    for h, p in eligible
    if p.name in CONFIRMED_TOMATO_NAMES
]


print()
print("===== CURRENT REAL-WORLD TOMATO AUDIT =====")
print(f"Captured files scanned:              {sum(1 for p in CAPTURED.rglob('*') if p.is_file() and p.suffix in EXTENSIONS)}")
print(f"Unique captured images:              {len(unique_captured)}")
print(f"Locked holdout hashes excluded:     {sum(1 for h in unique_captured if h in HOLDOUT_HASHES)}")
print(f"Already in gate excluded:            {sum(1 for h in unique_captured if h in gate_hashes)}")
print(f"Eligible unique real-world images:   {len(eligible)}")
print()
print("===== CONFIRMED REAL-WORLD TOMATO =====")

if tomato_candidates:
    print(f"Exact count: {len(tomato_candidates)}")
    for i, (_, p) in enumerate(tomato_candidates, 1):
        print(f"{i}. {p.name}")
else:
    print("Exact count: 0")

print()
print("===== CHECK =====")

missing_confirmed = [
    name
    for name in CONFIRMED_TOMATO_NAMES
    if not any(p.name == name for _, p in tomato_candidates)
]

if missing_confirmed:
    print("Confirmed tomato images not currently eligible:")
    for name in missing_confirmed:
        print(f" - {name}")
else:
    print("Both previously confirmed real-world tomato images are present and eligible.")
