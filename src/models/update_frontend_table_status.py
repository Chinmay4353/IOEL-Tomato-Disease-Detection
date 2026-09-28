from pathlib import Path

path = Path(r".\dashboard\script.js")
text = path.read_text(encoding="utf-8")

old = """    const display =
        getPredictionDisplay(
            prediction,
            confidence
        );
"""

new = """    const display =
        getPredictionDisplay(
            prediction,
            confidence,
            item?.status ??
                PREDICTION_STATUS.ACCEPTED
        );
"""

if old not in text:
    raise SystemExit(
        "Expected prediction table display call was not found. "
        "No changes made."
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("Prediction history table now uses backend status.")
