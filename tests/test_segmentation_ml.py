"""Unit tests for the model-based Task 2 methods. Run: python -m pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation_ml import ML_METHODS, MODEL_METHODS, STRATEGY_METHODS

# Every method here needs a trained classifier, which needs the exported digit
# images. Skip the whole module rather than fail on a fresh clone.
pytestmark = pytest.mark.skipif(
    not (ROOT / "data" / "raw" / "0").exists(),
    reason="needs data/raw - run 'python src/prepare_data.py' first",
)

# sliding_cnn needs TensorFlow and Thien's checkpoint; the rest run on numpy.
RUNNABLE = [name for name in ML_METHODS if name != "sliding_cnn"]


def make_number_image(text: str, spacing: int = 90) -> np.ndarray:
    width = 40 + spacing * len(text)
    img = np.zeros((120, width), dtype=np.uint8)
    for i, ch in enumerate(text):
        cv2.putText(img, ch, (20 + i * spacing, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 4)
    return cv2.bitwise_not(img)


def test_four_distinct_models_are_compared():
    """The tutor asked for four different models, not four uses of one model."""
    assert set(MODEL_METHODS) == {"sliding_mlp", "sliding_cnn",
                                  "sliding_svm", "sliding_rf"}


def test_the_four_models_are_genuinely_different_backends():
    from digit_classifier import BACKENDS
    assert set(BACKENDS) == {"mlp", "cnn", "svm", "random_forest"}
    # four distinct classes, not one class registered four times
    assert len({cls.__name__ for cls in BACKENDS.values()}) == 4


def test_strategy_methods_are_registered():
    assert set(STRATEGY_METHODS) == {"cutpoint_mlp", "confidence_split"}
    assert set(ML_METHODS) == set(MODEL_METHODS) | set(STRATEGY_METHODS)


@pytest.mark.parametrize("method", RUNNABLE)
def test_returns_crops_and_boxes_of_equal_length(method):
    crops, boxes = ML_METHODS[method](make_number_image("42"))
    assert len(crops) == len(boxes)


@pytest.mark.parametrize("method", RUNNABLE)
def test_boxes_are_ordered_left_to_right(method):
    _, boxes = ML_METHODS[method](make_number_image("482"))
    xs = [b[0] for b in boxes]
    assert xs == sorted(xs)


@pytest.mark.parametrize("method", RUNNABLE)
def test_crops_are_square(method):
    crops, _ = ML_METHODS[method](make_number_image("18"))
    assert all(c.shape[0] == c.shape[1] for c in crops)


@pytest.mark.parametrize("method", RUNNABLE)
def test_blank_image_returns_nothing(method):
    blank = np.full((100, 200), 255, dtype=np.uint8)
    crops, boxes = ML_METHODS[method](blank)
    assert crops == [] and boxes == []


def test_cut_points_can_split_ink_that_is_joined():
    """The reason this family exists.

    Two bars welded together are one connected region, so connected components
    must see a single digit. The cut-point method proposes boundaries from the
    ink profile and lets the classifier confirm them, so connectivity does not
    limit it.
    """
    from segmentation import segment_digits

    img = np.zeros((120, 200), dtype=np.uint8)
    cv2.rectangle(img, (40, 25), (90, 95), 255, -1)
    cv2.rectangle(img, (110, 25), (160, 95), 255, -1)
    cv2.rectangle(img, (90, 55), (110, 65), 255, -1)      # the weld
    img = cv2.bitwise_not(img)

    assert len(segment_digits(img, method="connected_components")[0]) == 1
    assert len(ML_METHODS["cutpoint_mlp"](img)[0]) >= 2
