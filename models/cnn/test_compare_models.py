"""Tests for compare_models.py on fake histories. Run: pytest test_compare_models.py"""

import json

import numpy as np
import pytest

import compare_models as cm


def _history(n, top):
    """A plausible learning curve that rises to `top`."""
    acc = list(top - 0.05 * np.exp(-np.arange(n) / 4))
    loss = list(0.3 * np.exp(-np.arange(n) / 5) + 0.02)
    return {"accuracy": [a + 0.002 for a in acc], "val_accuracy": acc,
            "loss": loss, "val_loss": [l * 1.2 for l in loss]}


def _summary(run_id, val, test, shifted):
    return {"config": {"run_id": run_id, "augmentation": "light", "dropout": 0.3,
                       "batch_size": 128},
            "val_accuracy": val, "test_accuracy": test, "robust_test_accuracy": shifted,
            "epochs_run": 20, "training_time_s": 100.0}


@pytest.fixture
def exp_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "EXP_DIR", tmp_path)
    (tmp_path / "tuning").mkdir()
    for key, *_ in cm.MODELS:
        (tmp_path / f"{key}_history.json").write_text(json.dumps(_history(25, 0.99)))
    for i, (key, *_) in enumerate(cm.BEST_MODELS):
        (tmp_path / f"{key}_history.json").write_text(json.dumps(_history(30, 0.992 + i * 0.001)))
        run_id = key.split("/")[-1]
        (tmp_path / f"{key}_summary.json").write_text(json.dumps(
            _summary(run_id, 0.99 + i * 0.001, 0.991 + i * 0.001, 0.98 + i * 0.002)))
    return tmp_path


def test_same_set_keeps_original_file_names(exp_dir):
    cm.main([])
    assert (exp_dir / "compare_accuracy.png").exists()
    assert (exp_dir / "compare_loss.png").exists()
    assert not (exp_dir / "compare_best_accuracy.png").exists()


def test_best_set_writes_separate_charts_and_table(exp_dir):
    cm.main(["--set", "best"])
    assert (exp_dir / "compare_best_accuracy.png").exists()
    assert (exp_dir / "compare_best_loss.png").exists()
    assert not (exp_dir / "compare_accuracy.png").exists()      # original charts untouched
    table = (exp_dir / "compare_best_table.md").read_text()
    for run in ("pm_mlp_best", "pm_lenet_best", "pm_resnet_best", "s3_B5_bs_64"):
        assert run in table
    assert "Shifted-test acc" in table


def test_missing_runs_are_skipped_not_fatal(exp_dir):
    (exp_dir / "tuning" / "pm_resnet_best_history.json").unlink()
    cm.main(["--set", "best"])
    assert (exp_dir / "compare_best_accuracy.png").exists()


def test_both_sets_use_the_same_models_and_colours():
    same = [(label, colour) for _, label, colour in cm.MODELS]
    best = [(label, colour) for _, label, colour in cm.BEST_MODELS]
    assert same == best
