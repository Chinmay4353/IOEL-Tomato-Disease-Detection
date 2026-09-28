from pathlib import Path

path = Path(r".\dashboard\script.js")
text = path.read_text(encoding="utf-8")

start_marker = "function updateStatistics(predictions) {"
end_marker = "\n}\n\n/* ============================================================\n   RECENT PREDICTIONS"

start = text.find(start_marker)

if start == -1:
    raise SystemExit(
        "updateStatistics function was not found. No changes made."
    )

end = text.find(end_marker, start)

if end == -1:
    raise SystemExit(
        "End of updateStatistics function was not found. No changes made."
    )

new_function = """function updateStatistics(predictions) {
    const total = predictions.length;

    const confirmedHealthy = predictions.filter(item => {
        const status =
            item?.status ??
            PREDICTION_STATUS.ACCEPTED;

        return (
            status === PREDICTION_STATUS.ACCEPTED &&
            String(item?.prediction || '')
                .trim()
                .toLowerCase() === 'healthy'
        );
    }).length;

    const confirmedDisease = predictions.filter(item => {
        const status =
            item?.status ??
            PREDICTION_STATUS.ACCEPTED;

        const prediction = String(
            item?.prediction || ''
        )
            .trim()
            .toLowerCase();

        return (
            status === PREDICTION_STATUS.ACCEPTED &&
            prediction &&
            prediction !== 'healthy' &&
            prediction !== 'not_tomato'
        );
    }).length;

    const average =
        total > 0
            ? (
                predictions.reduce(
                    (sum, item) => {
                        return (
                            sum +
                            getConfidenceValue(
                                item?.confidence
                            )
                        );
                    },
                    0
                ) / total
            ) * 100
            : 0;

    if (healthyCount) {
        healthyCount.textContent =
            String(confirmedHealthy);
    }

    if (diseaseCount) {
        diseaseCount.textContent =
            String(confirmedDisease);
    }

    if (averageConfidence) {
        averageConfidence.textContent =
            `${average.toFixed(1)}%`;
    }

    if (totalPredictions) {
        totalPredictions.textContent =
            String(total);
    }
}"""

text = text[:start] + new_function + text[end + 2:]

path.write_text(text, encoding="utf-8")

print("Dashboard statistics now use backend prediction status.")
print("Uncertain and non-tomato results are excluded from confirmed counts.")
