from pathlib import Path

from PIL import Image

from src.training.evaluate import evaluate_model


def test_evaluate_model_creates_expected_outputs(tmp_path):
    dataset_root = tmp_path / "dataset"
    test_dir = dataset_root / "test"
    for class_name in ["tomato", "not_tomato"]:
        (test_dir / class_name).mkdir(parents=True)
        for idx in range(2):
            image = Image.new("RGB", (224, 224), color=(255, 0, 0) if class_name == "tomato" else (0, 255, 0))
            image.save(test_dir / class_name / f"{class_name}_{idx}.jpg")

    model_path = tmp_path / "model.keras"
    results_dir = tmp_path / "results"

    model = None
    try:
        from src.training.train import build_model
        model = build_model()
        model.save(model_path)
    except Exception:
        model = None

    if model is None:
        raise AssertionError("Model could not be built for evaluation test")

    metrics = evaluate_model(model_path=model_path, dataset_dir=test_dir, results_dir=results_dir)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert (results_dir / "confusion_matrix.png").exists()
    assert (results_dir / "metrics.txt").exists()
