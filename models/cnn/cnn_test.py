import os
import csv
import tempfile
from pathlib import Path
import numpy as np
import pytest
import tensorflow as tf

from mnist_loader import load_mnist
from cnn_shallow import build_shallow
from experiment_logger import log_run

# Checkpoints are one level up: models/checkpoints/
CHECKPOINTS = {
    "shallow": Path(__file__).parent.parent / "checkpoints" / "cnn_shallow.keras",
    "lenet":   Path(__file__).parent.parent / "checkpoints" / "cnn_lenet.keras",
    "vgg":     Path(__file__).parent.parent / "checkpoints" / "cnn_vgg_small.keras",
}

THRESHOLDS = {
    "shallow": 0.95,
    "lenet":   0.98,
    "vgg":     0.99,
}


class TestMnistLoader:
    @pytest.fixture(scope="class")
    def splits(self):
        return load_mnist()

    def test_returns_three_splits(self, splits):
        assert len(splits) == 3, "load_mnist() must return (train, val, test)"

    def test_train_shape(self, splits):
        (x_train, y_train), _, _ = splits
        assert x_train.shape == (54_000, 28, 28, 1), \
            f"Expected (54000,28,28,1) got {x_train.shape}"
        assert y_train.shape == (54_000,)

    def test_val_shape(self, splits):
        _, (x_val, y_val), _ = splits
        assert x_val.shape == (6_000, 28, 28, 1), \
            f"Expected (6000,28,28,1) got {x_val.shape}"
        assert y_val.shape == (6_000,)

    def test_test_shape(self, splits):
        _, _, (x_test, y_test) = splits
        assert x_test.shape == (10_000, 28, 28, 1), \
            f"Expected (10000,28,28,1) got {x_test.shape}"
        assert y_test.shape == (10_000,)

    def test_pixel_values_normalised(self, splits):
        (x_train, _), (x_val, _), (x_test, _) = splits
        for name, arr in [("train", x_train), ("val", x_val), ("test", x_test)]:
            assert arr.min() >= 0.0, f"{name}: min below 0 ({arr.min()})"
            assert arr.max() <= 1.0, f"{name}: max above 1 ({arr.max()})"

    def test_dtype_float32(self, splits):
        (x_train, _), _, _ = splits
        assert x_train.dtype == np.float32, \
            f"Expected float32, got {x_train.dtype}"

    def test_labels_are_integers(self, splits):
        (_, y_train), _, _ = splits
        assert y_train.ndim == 1, "Labels should be 1-D, not one-hot"
        assert set(np.unique(y_train)) == set(range(10)), \
            "Labels must cover all 10 digit classes"

    def test_no_data_leakage(self, splits):
        (x_train, _), (x_val, _), (x_test, _) = splits

        def fingerprint(arr):
            return set(map(tuple, arr[:, :10, 0, 0].tolist()))

        train_fp = fingerprint(x_train)
        val_fp   = fingerprint(x_val)
        overlap_tv = train_fp & val_fp
        assert len(overlap_tv) < 10, \
            f"Possible train/val overlap: {len(overlap_tv)} shared fingerprints"

    def test_reproducible_with_seed(self):
        (x1, y1), (v1, _), _ = load_mnist()
        (x2, y2), (v2, _), _ = load_mnist()
        np.testing.assert_array_equal(y1, y2,
            err_msg="Train labels differ between two calls — seed not fixed")
        np.testing.assert_array_equal(v1, v2,
            err_msg="Val split differs between two calls — seed not fixed")


class TestShallowCNN:
    @pytest.fixture(scope="class")
    def model(self):
        return build_shallow()

    def test_model_compiles(self, model):
        assert model.optimizer is not None, "Model has no optimizer — call compile()"
        assert model.loss is not None, "Model has no loss function"

    def test_output_shape(self, model):
        dummy_input = tf.zeros((4, 28, 28, 1))
        output = model(dummy_input, training=False)
        assert output.shape == (4, 10), \
            f"Expected output shape (4, 10), got {output.shape}"

    def test_softmax_outputs_probabilities(self, model):
        dummy = tf.zeros((8, 28, 28, 1))
        preds = model(dummy, training=False).numpy()
        row_sums = preds.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(8), atol=1e-5,
            err_msg="Output rows do not sum to 1 — softmax broken?")

    def test_output_probabilities_in_range(self, model):
        dummy = tf.random.uniform((16, 28, 28, 1))
        preds = model(dummy, training=False).numpy()
        assert preds.min() >= 0.0
        assert preds.max() <= 1.0

    def test_parameter_count_shallow_range(self, model):
        total = model.count_params()
        assert 30_000 <= total <= 150_000, \
            f"Unexpected param count {total:,} — check architecture"

    def test_model_has_trainable_weights(self, model):
        trainable = sum(tf.size(w).numpy() for w in model.trainable_weights)
        assert trainable > 0, "Model has no trainable weights"

    def test_different_inputs_different_outputs(self, model):
        zeros = tf.zeros((1, 28, 28, 1))
        ones  = tf.ones((1, 28, 28, 1))
        out_zeros = model(zeros, training=False).numpy()
        out_ones  = model(ones,  training=False).numpy()
        assert not np.allclose(out_zeros, out_ones), \
            "Model outputs identical values for all inputs — weights may be dead"


class TestOneEpochSmoke:
    def test_one_epoch_does_not_crash(self):
        (x_train, y_train), (x_val, y_val), _ = load_mnist()
        x_mini  = x_train[:512]
        y_mini  = y_train[:512]
        x_vmini = x_val[:128]
        y_vmini = y_val[:128]

        model = build_shallow()
        history = model.fit(
            x_mini, y_mini,
            validation_data=(x_vmini, y_vmini),
            epochs=1,
            batch_size=128,
            verbose=0,
        )
        assert "accuracy" in history.history
        assert "val_accuracy" in history.history
        assert history.history["loss"][0] < 10.0, \
            "Loss suspiciously high — check input normalisation"


def _load_test_data():
    _, _, (x_test, y_test) = load_mnist()
    return x_test, y_test


@pytest.mark.skipif(
    not CHECKPOINTS["shallow"].exists(),
    reason="cnn_shallow.keras not found — run cnn_shallow.py first"
)
def test_shallow_quality_gate():
    model = tf.keras.models.load_model(str(CHECKPOINTS["shallow"]))
    x_test, y_test = _load_test_data()
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    assert acc >= THRESHOLDS["shallow"], \
        f"Shallow CNN test accuracy {acc:.4f} is below threshold {THRESHOLDS['shallow']}"


@pytest.mark.skipif(
    not CHECKPOINTS["lenet"].exists(),
    reason="cnn_lenet.keras not found — run cnn_lenet.py first"
)
def test_lenet_quality_gate():
    model = tf.keras.models.load_model(str(CHECKPOINTS["lenet"]))
    x_test, y_test = _load_test_data()
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    assert acc >= THRESHOLDS["lenet"], \
        f"LeNet test accuracy {acc:.4f} is below threshold {THRESHOLDS['lenet']}"


@pytest.mark.skipif(
    not CHECKPOINTS["vgg"].exists(),
    reason="cnn_vgg_small.keras not found — run cnn_vgg_small.py first"
)
def test_vgg_quality_gate():
    model = tf.keras.models.load_model(str(CHECKPOINTS["vgg"]))
    x_test, y_test = _load_test_data()
    _, acc = model.evaluate(x_test, y_test, verbose=0)
    assert acc >= THRESHOLDS["vgg"], \
        f"VGG-small test accuracy {acc:.4f} is below threshold {THRESHOLDS['vgg']}"


class TestExperimentLogger:
    EXPECTED_COLUMNS = {
        "timestamp", "run_id", "architecture", "learning_rate",
        "batch_size", "epochs", "dropout", "val_accuracy",
        "test_accuracy", "training_time_s", "notes"
    }

    def test_log_creates_file(self, tmp_path, monkeypatch):
        import experiment_logger
        log_file = tmp_path / "experiment_log.csv"
        monkeypatch.setattr(experiment_logger, "LOG_PATH", log_file)
        log_run(run_id="test_run", architecture="test", learning_rate=0.001,
                batch_size=128, epochs=1, dropout=0.0, val_accuracy=0.95,
                test_accuracy=0.94, training_time_s=1.0, notes="unit test")
        assert log_file.exists(), "log_run() did not create the CSV file"

    def test_log_writes_correct_columns(self, tmp_path, monkeypatch):
        import experiment_logger
        log_file = tmp_path / "experiment_log.csv"
        monkeypatch.setattr(experiment_logger, "LOG_PATH", log_file)
        log_run(run_id="test_run", architecture="test", learning_rate=0.001,
                batch_size=128, epochs=1, dropout=0.0, val_accuracy=0.95,
                test_accuracy=0.94, training_time_s=1.0, notes="unit test")
        with open(log_file) as f:
            headers = set(csv.DictReader(f).fieldnames or [])
        missing = self.EXPECTED_COLUMNS - headers
        assert not missing, f"CSV missing columns: {missing}"

    def test_log_appends_not_overwrites(self, tmp_path, monkeypatch):
        import experiment_logger
        log_file = tmp_path / "experiment_log.csv"
        monkeypatch.setattr(experiment_logger, "LOG_PATH", log_file)
        for i in range(2):
            log_run(run_id=f"run_{i}", architecture="test", learning_rate=0.001,
                    batch_size=128, epochs=1, dropout=0.0,
                    val_accuracy=0.90 + i * 0.01, test_accuracy=0.89 + i * 0.01,
                    training_time_s=1.0, notes=f"run {i}")
        with open(log_file) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 2, \
            f"Expected 2 logged rows, got {len(rows)} — logger may be overwriting"
