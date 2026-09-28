from pathlib import Path

path = Path(r".\backend\app.py")
text = path.read_text(encoding="utf-8")

old = '''                "confidence": row[
                    "confidence"
                ],
                "device_id": row[
'''

new = '''                "confidence": row[
                    "confidence"
                ],
                "status": get_prediction_status(
                    row["prediction"],
                    row["confidence"],
                ),
                "device_id": row[
'''

if old not in text:
    raise SystemExit(
        "Expected prediction history response block was not found. "
        "No changes made."
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("Prediction history API now includes derived status.")
