"""Tests for the evaluation lane (no trained model, no MNIST download needed).

Run from the repository root:  python -m pytest evaluation -q
"""
import cv2
import numpy as np
import pytest

from baselines import HOG_CONFIGS, Baseline, hog_features
from common import wilson
from error_analysis import category, dropped_digit
from evaluate_all import levenshtein, mcnemar, number_metrics


#  metrics

def test_levenshtein():
    assert levenshtein("896", "896") == 0
    assert levenshtein("896", "826") == 1          # one wrong
    assert levenshtein("896", "86") == 1           # one missing
    assert levenshtein("896", "8966") == 1         # one extra
    assert levenshtein("", "123") == 3


def test_number_metrics_keeps_per_digit_and_whole_number_apart():
    rows = [
        {"truth": "12", "read": "12", "count_ok": True},     # right
        {"truth": "34", "read": "35", "count_ok": True},     # one digit wrong
        {"truth": "56", "read": "5", "count_ok": False},     # segmentation lost a digit
    ]
    m = number_metrics(rows)
    assert m["exact"] == pytest.approx(1 / 3)
    assert m["count_ok"] == pytest.approx(2 / 3)
    assert m["digit_acc_count_ok"] == pytest.approx(3 / 4)   # only the two right-count numbers
    assert m["exact_count_ok"] == pytest.approx(1 / 2)
    assert m["digit_acc"] == pytest.approx(1 - 2 / 6)        # one wrong + one missing out of 6


def test_mcnemar_counts_discordant_pairs_only():
    ref = np.array([True, True, True, False, True])
    other = np.array([True, False, False, False, True])
    r = mcnemar(ref, other)
    assert (r["ref_only_right"], r["other_only_right"]) == (2, 0)
    assert 0 < r["p_value"] <= 1
    assert mcnemar(ref, ref)["p_value"] == 1.0


def test_wilson_interval_contains_the_estimate():
    lo, hi = wilson(80, 100)
    assert lo < 0.8 < hi and 0 <= lo and hi <= 1
    assert wilson(0, 0) == (0.0, 0.0)


#  baselines

@pytest.mark.parametrize("config,length", [("hog7", 324), ("hog4", 1296)])
def test_hog_features_shape_and_input_formats(config, length):
    img = np.zeros((28, 28), "float32")
    img[6:22, 12:16] = 1.0
    a = hog_features(img[None], config)
    b = hog_features((img * 255)[None, ..., None], config)     # 0-255 with a channel axis
    assert a.shape == (1, length) and config in HOG_CONFIGS
    np.testing.assert_allclose(a, b, atol=1e-5)


def test_baseline_wrapper_round_trip(tmp_path):
    """Saved and reloaded, the wrapper gives exactly what the classifier gives on HOG features."""
    import joblib
    from sklearn.svm import SVC
    x = np.zeros((60, 28, 28), "float32")
    y = np.repeat(np.arange(10), 6)
    rng = np.random.default_rng(0)
    for i, d in enumerate(y):                    # each class: a stroke at its own angle
        angle = np.deg2rad(18 * d + rng.normal(0, 3))
        dx, dy = 9 * np.cos(angle), 9 * np.sin(angle)
        cv2.line(x[i], (int(14 - dx), int(14 - dy)), (int(14 + dx), int(14 + dy)), 1.0, 2)
    features = hog_features(x, "hog7")
    clf = SVC(probability=True, random_state=0).fit(features, y)
    joblib.dump(Baseline("hog_svm", "hog7", clf), tmp_path / "m.joblib")
    loaded = joblib.load(tmp_path / "m.joblib")
    np.testing.assert_allclose(loaded.predict_proba(x), clf.predict_proba(features))
    assert loaded.predict_proba(x).shape == (60, 10)
    assert loaded.size() == clf.n_support_.sum()


#  failure categories

def number_image(digits_at, line=False):
    """White page with dark blobs at the given x positions (one blob = one digit)."""
    img = np.full((80, 300, 3), 255, np.uint8)
    for x in digits_at:
        cv2.rectangle(img, (x, 20), (x + 18, 60), (0, 0, 0), 4)
    if line:
        cv2.line(img, (5, 72), (295, 72), (0, 0, 0), 2)
    return img


def test_dropped_digit_sees_a_digit_with_no_box():
    img = number_image([20, 120, 220])
    boxes = [(18, 18, 24, 46), (118, 18, 24, 46)]            # the third digit got no box
    assert dropped_digit(img, boxes)


def test_merged_digits_are_not_dropped():
    img = number_image([20, 40])                              # two digits close together
    assert not dropped_digit(img, [(18, 18, 44, 46)])        # one box around both


def test_ruled_line_outside_the_boxes_is_not_a_dropped_digit():
    img = number_image([20, 120], line=True)
    assert not dropped_digit(img, [(18, 18, 24, 46), (118, 18, 24, 46)])


def test_category():
    two = [(0, 0, 10, 20), (20, 0, 10, 20)]
    assert category("12", "12", two) == "correct"
    assert category("123", "12", two, dropped=False) == "merged"
    assert category("123", "12", two, dropped=True) == "dropped"
    assert category("1", "12", two) == "split"
    assert category("13", "12", two) == "misread"
