from pathlib import Path

path = Path(r".\dashboard\script.js")
text = path.read_text(encoding="utf-8")

old = """        const prediction =
            result?.label ??
            result?.prediction ??
            '';

        const confidence =
            getConfidenceValue(result?.confidence);

        showAnalysisResult(
            prediction,
            confidence
        );

        await fetchDashboardData();
"""

new = """        const prediction =
            result?.prediction ??
            result?.label ??
            '';

        const confidence =
            getConfidenceValue(result?.confidence);

        const status =
            result?.status ??
            PREDICTION_STATUS.UNCERTAIN;

        showAnalysisResult(
            prediction,
            confidence,
            status,
            result?.recommendation
        );

        await fetchDashboardData();
"""

if old not in text:
    raise SystemExit(
        "Expected analyzeSelectedImage response block was not found. "
        "No changes made."
    )

text = text.replace(old, new, 1)

path.write_text(text, encoding="utf-8")

print("Frontend now reads backend status and recommendation.")
