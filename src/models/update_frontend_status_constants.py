from pathlib import Path

path = Path(r".\dashboard\script.js")
text = path.read_text(encoding="utf-8")

old = "const CONFIDENCE_THRESHOLD = 0.70;\nconst REFRESH_INTERVAL_MS = 5000;"

new = """const REFRESH_INTERVAL_MS = 5000;

const PREDICTION_STATUS = Object.freeze({
    ACCEPTED: 'ACCEPTED',
    UNCERTAIN: 'UNCERTAIN',
    NOT_TOMATO: 'NOT_TOMATO'
});"""

if old not in text:
    raise SystemExit(
        "Expected old confidence threshold block was not found. "
        "No changes made."
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("Frontend confidence threshold constant removed.")
print("Backend prediction status constants added.")
