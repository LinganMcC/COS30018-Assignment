
import json

import numpy as np
import pytest
import tensorflow as tf

import datasets
from augmentation import (PRESETS, build_augmenter, get_preset,
                          make_shifted_test_set)
from datasets import DIGIT_LABELS, SYMBOL_LABELS, DatasetSplits, load_dataset
from model_zoo import available_architectures, build_model
from summarise_tuning import freeze, load_summaries, markdown_table, pick_best
from trainer import TrainConfig, load_grid, train


def _synthetic(n_classes: int = 10, n_per_class: int = 20, seed: int = 0) -> DatasetSplits:
    """Class k = a bright 6x6 square at a class-specific position + noise."""
    rng = np.random.default_rng(seed)

    def make(n):
        xs, ys = [], []
        for k in range(n_classes):
            r, c = 2 + (k % 4) * 6, 2 + (k // 4) * 6
            img = rng.random((n, 28, 28, 1)).astype("float32") * 0.1
            img[:, r:r + 6, c:c + 6, :] = 1.0
            xs.append(img)
            ys.append(np.full(n, k))
        return np.concatenate(xs), np.concatenate(ys)

    xt, yt = make(n_per_class)
    xv, yv = make(5)
    xs, ys = make(5)
    labels = (DIGIT_LABELS + SYMBOL_LABELS)[:n_classes]
    return DatasetSplits(xt, yt, xv, yv, xs, ys, labels)


# model zoo

@pytest.mark.parametrize("arch", available_architectures())
@pytest.mark.parametrize("num_classes", [10, 16])
def test_every_arch_builds_with_right_output(arch, num_classes):
    model = build_model(arch, num_classes=num_classes)
    out = model(np.zeros((2, 28, 28, 1), dtype="float32"))
    assert out.shape == (2, num_classes)
    np.testing.assert_allclose(np.sum(out, axis=1), 1.0, atol=1e-5)  # softmax


def test_unknown_arch_rejected():
    with pytest.raises(ValueError):
        build_model("not_a_model")


def test_bad_dropout_rejected():
    with pytest.raises(ValueError):
        build_model("vgg_deep", dropout=1.0)


#  augmentation

def test_none_preset_is_identity():
    assert build_augmenter(get_preset("none")) is None


@pytest.mark.parametrize("preset", [p for p in PRESETS if p != "none"])
def test_augmentation_keeps_shape_and_changes_pixels(preset):
    x = _synthetic().x_train[:8]
    out = np.asarray(build_augmenter(get_preset(preset), seed=1)(x, training=True))
    assert out.shape == x.shape
    assert not np.allclose(out, x)


def test_shifted_test_set_is_deterministic_and_in_range():
    x = _synthetic().x_test
    a = make_shifted_test_set(x, seed=0)
    b = make_shifted_test_set(x, seed=0)
    assert a.shape == x.shape
    assert a.min() >= 0.0 and a.max() <= 1.0
    np.testing.assert_allclose(a, b)


def test_unknown_preset_rejected():
    with pytest.raises(ValueError):
        get_preset("extreme")


#  datasets

def test_symbols_dataset_is_placeholder_for_now():
    with pytest.raises(NotImplementedError):
        load_dataset("mnist+symbols")


def test_label_ordering_agreed():
    assert DIGIT_LABELS == [str(i) for i in range(10)]
    assert len(DIGIT_LABELS + SYMBOL_LABELS) == 16


# trainer

@pytest.fixture
def trained(tmp_path):
    """Train one tiny run into a temp folder and return (summary, cfg)."""
    cfg = TrainConfig(run_id="t_unit", arch="lenet", augmentation="light",
                      max_epochs=2, batch_size=32, output_root=str(tmp_path))
    return train(cfg, data=_synthetic()), cfg, tmp_path


def test_train_writes_all_outputs(trained):
    summary, cfg, root = trained
    assert (root / "checkpoints/tuning/t_unit.keras").exists()
    assert (root / "experiments/tuning/t_unit_history.png").exists()
    assert (root / "experiments/tuning/t_unit_summary.json").exists()
    labels = json.loads((root / "checkpoints/tuning/t_unit.labels.json").read_text())
    assert labels["label_names"] == DIGIT_LABELS
    log = (root / "experiments/tuning_log.csv").read_text()
    assert "t_unit" in log and "aug=light" in log
    for key in ("val_accuracy", "test_accuracy", "robust_test_accuracy"):
        assert 0.0 <= summary[key] <= 1.0


def test_saved_model_reloads_and_predicts(trained):
    _, _, root = trained
    model = tf.keras.models.load_model(root / "checkpoints/tuning/t_unit.keras")
    assert model.predict(np.zeros((3, 28, 28, 1), "float32"), verbose=0).shape == (3, 10)


def test_train_supports_16_classes(tmp_path):
    cfg = TrainConfig(run_id="t16", arch="lenet", max_epochs=1, batch_size=32,
                      output_root=str(tmp_path))
    train(cfg, data=_synthetic(n_classes=16))
    labels = json.loads((tmp_path / "checkpoints/tuning/t16.labels.json").read_text())
    assert labels["label_names"][10:] == SYMBOL_LABELS


def test_quick_runs_are_not_logged(tmp_path):
    cfg = TrainConfig(run_id="t_quick", arch="lenet", quick=True, batch_size=32,
                      output_root=str(tmp_path))
    train(cfg, data=_synthetic())
    assert not (tmp_path / "experiments/tuning_log.csv").exists()
    assert (tmp_path / "experiments/tuning_quick/t_quick_summary.json").exists()


#  grid

def test_sprint3_grid_parses_with_correct_precedence():
    from pathlib import Path
    grid = Path(__file__).parent / "configs" / "sprint3_tuning.json"
    configs = load_grid(grid)
    ids = [c.run_id for c in configs]
    assert len(ids) == len(set(ids)), "run_ids must be unique"
    by_id = {c.run_id: c for c in configs}
    assert by_id["s3_A1_aug_none"].augmentation == "none"
    assert by_id["s3_B1_lr_3e-4"].augmentation == "medium"   # stage default
    assert by_id["s3_B1_lr_3e-4"].learning_rate == 0.0003    # run override
    assert by_id["s3_B1_lr_3e-4"].arch == "vgg_deep"         # file default
    for c in configs:  # every config must reference real presets / archs
        get_preset(c.augmentation)
        assert c.arch in available_architectures()


def test_grid_rejects_typos(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"stages": {"A": {"runs": [
        {"run_id": "x", "learnin_rate": 0.1}]}}}))
    with pytest.raises(ValueError):
        load_grid(bad)


# summarise

def test_best_is_chosen_by_val_not_test(trained):
    _, _, root = trained
    s = load_summaries(root)[0]
    good_val = {**s, "val_accuracy": 0.99, "test_accuracy": 0.90}
    good_test = {**s, "val_accuracy": 0.95, "test_accuracy": 0.999}
    assert pick_best([good_test, good_val]) is good_val
    assert "**(best)**" in markdown_table([good_val], good_val)


def test_freeze_copies_checkpoint_and_labels(trained):
    summary, _, root = trained
    freeze(summary, root)
    assert (root / "checkpoints/cnn_best.keras").exists()
    assert (root / "checkpoints/cnn_best.labels.json").exists()
