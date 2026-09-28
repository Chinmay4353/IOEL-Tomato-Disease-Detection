from pathlib import Path

path = Path(r".\backend\app.py")
text = path.read_text(encoding="utf-8")

marker = "\ndef validate_prediction_payload("

helper = '''

def get_prediction_status(prediction, confidence):
    if prediction == "NOT_TOMATO":
        return "NOT_TOMATO"

    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        return "UNCERTAIN"

    if confidence_value < 0.80:
        return "UNCERTAIN"

    return "ACCEPTED"
'''

if "def get_prediction_status(" in text:
    raise SystemExit(
        "get_prediction_status already exists. No changes made."
    )

if marker not in text:
    raise SystemExit(
        "validate_prediction_payload marker was not found. No changes made."
    )

text = text.replace(
    marker,
    helper + marker,
    1,
)

path.write_text(text, encoding="utf-8")

print("Historical prediction status helper added.")
