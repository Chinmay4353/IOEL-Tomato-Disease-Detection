from pathlib import Path

path = Path(r".\dashboard\script.js")
text = path.read_text(encoding="utf-8")

old = """function showAnalysisResult(prediction, confidence) {
    if (!analysisResult) {
        return;
    }

    const display = getPredictionDisplay(
        prediction,
        confidence
    );

    clearStatusClasses(analysisResult);

    analysisResult.classList.add(
        display.statusClass
    );

    analysisResult.hidden = false;

    analysisResult.textContent =
        display.isLowConfidence
            ? `⚠ ${display.label} · ${display.confidence} confidence`
            : `${display.label} · ${display.confidence} confidence`;

    setAnalysisStatus(
        display.isLowConfidence
            ? 'Analysis complete. The model is uncertain about this image.'
            : 'Analysis complete.'
    );
}
"""

new = """function showAnalysisResult(
    prediction,
    confidence,
    status,
    recommendation
) {
    if (!analysisResult) {
        return;
    }

    const display = getPredictionDisplay(
        prediction,
        confidence,
        status
    );

    clearStatusClasses(analysisResult);

    analysisResult.classList.add(
        display.statusClass
    );

    analysisResult.hidden = false;

    const prefix = display.isNotTomato
        ? '⚠ '
        : display.isLowConfidence
            ? '⚠ '
            : '';

    analysisResult.textContent =
        `${prefix}${display.label} · ${display.confidence} confidence`;

    if (recommendation) {
        analysisResult.textContent +=
            ` · ${recommendation}`;
    }

    setAnalysisStatus(
        status === PREDICTION_STATUS.NOT_TOMATO
            ? 'This image was not identified as a tomato leaf/plant image.'
            : status === PREDICTION_STATUS.UNCERTAIN
                ? 'Analysis complete. Capture a clearer image for a more reliable result.'
                : 'Analysis complete.'
    );
}
"""

if old not in text:
    raise SystemExit(
        "Expected showAnalysisResult function was not found. "
        "No changes made."
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("Analysis result now displays backend status and recommendation.")
