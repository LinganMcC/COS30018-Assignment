"""Tests for the per-class metrics in evaluate_classes.py (no TensorFlow needed)."""
import csv

import numpy as np
import pytest

import evaluate_classes as ec

LABELS = ["0", "1", "+", "("]


@pytest.fixture
def toy():
    # 0: 3 right            1: 3 right, 1 read as "("
    # +: 2 right            (: 1 right, 1 read as "1"
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 3, 2, 2, 3, 1])
    return y_true, y_pred


def test_confusion_matrix_counts(toy):
    cm = ec.confusion_matrix(*toy, num_classes=4)
    assert cm.sum() == 11
    assert cm[1, 3] == 1 and cm[3, 1] == 1
    assert list(np.diag(cm)) == [3, 3, 2, 1]


def test_per_class_scores(toy):
    s = ec.per_class_scores(ec.confusion_matrix(*toy, num_classes=4))
    assert s["recall"][1] == pytest.approx(3 / 4)
    assert s["precision"][1] == pytest.approx(3 / 4)     # one "(" predicted as 1
    assert s["recall"][3] == pytest.approx(1 / 2)
    assert s["f1"][0] == pytest.approx(1.0)
    assert list(s["support"]) == [3, 4, 2, 2]


def test_never_predicted_class_scores_zero_not_nan():
    cm = ec.confusion_matrix(np.array([0, 1]), np.array([0, 0]), num_classes=2)
    s = ec.per_class_scores(cm)
    assert s["precision"][1] == 0.0 and s["f1"][1] == 0.0
    assert not np.isnan(s["f1"]).any()


def test_top_confusions_sorted_and_off_diagonal(toy):
    cm = ec.confusion_matrix(*toy, num_classes=4)
    conf = ec.top_confusions(cm, LABELS, k=5)
    assert len(conf) == 2                       # only two off-diagonal cells
    assert {(t, p) for t, p, _, _ in conf} == {("1", "("), ("(", "1")}
    assert all(c == 1 for _, _, c, _ in conf)


def test_group_accuracy_splits_digits_and_symbols(toy):
    r = ec.summarise(*toy, LABELS)
    assert r["digit_acc"] == pytest.approx(6 / 7)
    assert r["symbol_acc"] == pytest.approx(3 / 4)
    assert r["overall_acc"] == pytest.approx(9 / 11)


def test_digit_only_model_has_no_symbol_score():
    y = np.array([0, 1, 1])
    r = ec.summarise(y, y, ["0", "1"])
    assert r["symbol_acc"] is None and r["overall_acc"] == 1.0


def test_outputs_written(tmp_path, toy):
    clean = ec.summarise(*toy, LABELS)
    ec.write_per_class_csv(tmp_path / "x.csv", LABELS, clean["scores"], clean["scores"])
    ec.save_confusion_png(clean["cm"], LABELS, "toy", tmp_path / "x.png")
    ec.write_report_md(tmp_path / "x.md", "toy", LABELS, {"clean": clean})
    rows = list(csv.DictReader((tmp_path / "x.csv").open()))
    assert [r["class"] for r in rows] == LABELS and "shifted_recall" in rows[0]
    assert (tmp_path / "x.png").stat().st_size > 0
    assert "Most frequent confusions" in (tmp_path / "x.md").read_text(encoding="utf-8")
