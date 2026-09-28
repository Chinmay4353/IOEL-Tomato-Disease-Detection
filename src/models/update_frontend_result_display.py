from pathlib import Path

path = Path(r".\dashboard\script.js")
text = path.read_text(encoding="utf-8")

start_marker = "function getPredictionDisplay(prediction, confidence) {"
end_marker = "\n}\n\nfunction clearStatusClasses(element) {"

start = text.find(start_marker)

if start == -1:
    raise SystemExit(
        "getPredictionDisplay function was not found. No changes made."
    )

end = text.find(end_marker, start)

if end == -1:
    raise SystemExit(
        "End of getPredictionDisplay function was not found. No changes made."
    )

new_function = """function getPredictionDisplay(
    prediction,
    confidence,
    status = PREDICTION_STATUS.ACCEPTED
) {
    const confidenceValue = getConfidenceValue(confidence);
    const confidenceText = formatConfidence(confidenceValue);

    if (status === PREDICTION_STATUS.NOT_TOMATO) {
        return {
            label: 'Not a Tomato Leaf / Plant Image',
            confidence: confidenceText,
            statusClass: 'result-uncertain',
            tableClass: 'prediction-status-uncertain',
            confidenceClass: 'prediction-confidence-uncertain',
            isLowConfidence: false,
            isNotTomato: true
        };
    }

    if (status === PREDICTION_STATUS.UNCERTAIN) {
        return {
            label: 'Low Confidence / Uncertain',
            confidence: confidenceText,
            statusClass: 'result-uncertain',
            tableClass: 'prediction-status-uncertain',
            confidenceClass: 'prediction-confidence-uncertain',
            isLowConfidence: true,
            isNotTomato: false
        };
    }

    const normalizedPrediction = String(prediction || '')
        .trim()
        .toLowerCase();

    if (normalizedPrediction === 'healthy') {
        return {
            label: 'Healthy',
            confidence: confidenceText,
            statusClass: 'result-healthy',
            tableClass: 'prediction-status-healthy',
            confidenceClass: 'prediction-confidence-healthy',
            isLowConfidence: false,
            isNotTomato: false
        };
    }

    return {
        label: formatPredictionLabel(prediction),
        confidence: confidenceText,
        statusClass: 'result-disease',
        tableClass: 'prediction-status-disease',
        confidenceClass: 'prediction-confidence-disease',
        isLowConfidence: false,
        isNotTomato: false
    };
}"""

text = text[:start] + new_function + text[end + 2:]

path.write_text(text, encoding="utf-8")

print("Frontend result display now uses backend prediction status.")
print("NOT_TOMATO, UNCERTAIN, HEALTHY, and disease states are supported.")
