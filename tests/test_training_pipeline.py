from pathlib import Path

from PIL import Image

from src.training.train import build_model, prepare_training_directories, train_model


def test_build_model_has_binary_output():
    model = build_model(input_shape=(224, 224, 3))

    assert model.output_shape[-1] == 1
    assert model.output_shape[0] is None


def test_prepare_training_directories_creates_expected_folders(tmp_path):
    dataset_root = tmp_path / "dataset"
    train_dir, val_dir = prepare_training_directories(dataset_root)

    assert "train" in str(train_dir)
    assert "val" in str(val_dir)
    assert train_dir.parent == dataset_root
    assert val_dir.parent == dataset_root


def test_train_model_runs_on_small_synthetic_dataset(tmp_path):
    dataset_root = tmp_path / "dataset"
    train_dir = dataset_root / "train"
    val_dir = dataset_root / "val"

    for class_name in ["tomato", "not_tomato"]:
        (train_dir / class_name).mkdir(parents=True)
        (val_dir / class_name).mkdir(parents=True)

        for idx in range(2):
            image = Image.new("RGB", (224, 224), color=(255, 0, 0) if class_name == "tomato" else (0, 255, 0))
            image.save(train_dir / class_name / f"{class_name}_{idx}.jpg")
            image.save(val_dir / class_name / f"{class_name}_{idx}_val.jpg")

    metrics = train_model(
        dataset_root=dataset_root,
        epochs=1,
        batch_size=2,
        model_path=tmp_path / "model" / "tomato_detector.keras",
        output_dir=tmp_path / "results",
        verbose=0,
    )

    assert "train_accuracy" in metrics
    assert "val_accuracy" in metrics
    assert (tmp_path / "results" / "training_history.png").exists()
    assert (tmp_path / "results" / "metrics.txt").exists()
