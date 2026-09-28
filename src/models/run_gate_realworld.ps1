param(
    [ValidateSet("before", "after")]
    [string]$RunLabel = "before"
)

Write-Host ""
Write-Host "Running Tomato Gate real-world holdout test..." -ForegroundColor Cyan
Write-Host "Run label: $RunLabel" -ForegroundColor Cyan
Write-Host ""

python ".\src\models\evaluate_gate_realworld.py" $RunLabel

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Holdout test failed." -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "Holdout test completed successfully." -ForegroundColor Green
Write-Host "Results: .\results\gate_realworld_holdout.csv" -ForegroundColor Green
