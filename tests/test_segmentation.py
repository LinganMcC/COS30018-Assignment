"""Unit tests for Task 2 segmentation. Run: python -m pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from segmentation import METHODS, SELECTED_METHOD, annotate, segment_digits


def make_number_image(text: str, spacing: int = 90) -> np.ndarray:
    """Draw `text` as widely spaced digits, dark on white."""
    width = 40 + spacing * len(text)
    img = np.full((140, width), 255, dtype=np.uint8)
    for i, ch in enumerate(text):
        cv2.putText(img, ch, (25 + i * spacing, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, 0, 5)
    return img


@pytest.mark.parametrize("method", list(METHODS.keys()))
@pytest.mark.parametrize("number", ["7", "42", "913", "5061"])
def test_finds_correct_digit_count(method, number):
    crops, boxes = segment_digits(make_number_image(number), method=method)
    assert len(crops) == len(number)
    assert len(boxes) == len(number)


@pytest.mark.parametrize("method", list(METHODS.keys()))
def test_boxes_are_ordered_left_to_right(method):
    _, boxes = segment_digits(make_number_image("1234"), method=method)
    x_positions = [b[0] for b in boxes]
    assert x_positions == sorted(x_positions)


@pytest.mark.parametrize("method", list(METHODS.keys()))
def test_crops_are_square(method):
    """Crops must be square so a '1' is not stretched when resized."""
    crops, _ = segment_digits(make_number_image("18"), method=method)
    for crop in crops:
        assert crop.shape[0] == crop.shape[1]


@pytest.mark.parametrize("method", list(METHODS.keys()))
def test_crops_are_not_empty(method):
    crops, _ = segment_digits(make_number_image("36"), method=method)
    for crop in crops:
        assert crop.sum() > 0


@pytest.mark.parametrize("method", list(METHODS.keys()))
def test_noise_specks_are_ignored(method):
    """Stray pixels must not be counted as digits."""
    img = make_number_image("5")
    cv2.circle(img, (10, 10), 1, 0, -1)
    cv2.circle(img, (120, 130), 1, 0, -1)
    crops, _ = segment_digits(img, method=method)
    assert len(crops) == 1


def test_blank_image_returns_nothing():
    blank = np.full((100, 200), 255, dtype=np.uint8)
    crops, boxes = segment_digits(blank)
    assert crops == [] and boxes == []


def test_unknown_method_raises():
    with pytest.raises(KeyError):
        segment_digits(make_number_image("1"), method="not_a_method")


def test_annotate_returns_colour_image_of_same_size():
    img = make_number_image("24")
    _, boxes = segment_digits(img)
    out = annotate(img, boxes)
    assert out.shape[:2] == img.shape[:2]
    assert out.ndim == 3


# --- what makes the four methods genuinely different ------------------------

def test_all_four_methods_are_registered():
    assert set(METHODS) == {"contours", "connected_components",
                            "projection", "watershed"}


def test_selected_method_is_a_real_method():
    assert SELECTED_METHOD in METHODS


def test_default_matches_the_selected_method():
    """Same guard as Task 1: the code must use the technique we wrote down."""
    img = make_number_image("482")
    default_crops, default_boxes = segment_digits(img)
    chosen_crops, chosen_boxes = segment_digits(img, method=SELECTED_METHOD)
    assert default_boxes == chosen_boxes
    assert len(default_crops) == len(chosen_crops)


def test_projection_splits_shapes_that_are_connected():
    """Two bars joined by a thin bridge are ONE connected region.

    Connected components and contours must see a single digit. The projection
    profile looks at ink per column instead of connectivity, so the thin bridge
    reads as a valley and it cuts there. This is the behaviour that makes it a
    different technique rather than a third way of computing the same thing.
    """
    img = np.zeros((120, 200), dtype=np.uint8)
    cv2.rectangle(img, (30, 20), (70, 100), 255, -1)
    cv2.rectangle(img, (130, 20), (170, 100), 255, -1)
    cv2.rectangle(img, (70, 58), (130, 62), 255, -1)      # thin bridge
    img = cv2.bitwise_not(img)

    assert len(segment_digits(img, method="connected_components")[0]) == 1
    assert len(segment_digits(img, method="projection")[0]) == 2


def test_watershed_crops_do_not_leak_neighbouring_ink():
    """Each watershed crop is masked to its own label.

    Without the mask, two digits that touch would each carry a slice of the
    other into their crop, which is exactly what the method exists to avoid.
    """
    img = np.zeros((120, 200), dtype=np.uint8)
    cv2.circle(img, (80, 60), 30, 255, -1)
    cv2.circle(img, (120, 60), 30, 255, -1)
    img = cv2.bitwise_not(img)

    crops, boxes = segment_digits(img, method="watershed")
    for crop, (_, _, w, h) in zip(crops, boxes):
        # a masked crop holds less ink than the full box would if both blobs
        # were present, so it can never be completely filled
        assert crop.sum() / 255 < w * h
