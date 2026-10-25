"""Unit tests for Task 2 segmentation. Run: python -m pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from segmentation import (METHODS, SELECTED_METHOD, annotate, segment_digits,
                          segment_expression)


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


def make_division_image(dot_radius: int) -> np.ndarray:
    """'8 ÷ 2' with a handwritten-size division sign (dots ~8 % of digit height)."""
    img = make_number_image("8 2")
    cv2.line(img, (125, 75), (185, 75), 0, 5)
    for y in (50, 100):
        cv2.circle(img, (155, y), dot_radius, 0, -1)
    return img


def test_small_division_dots_join_the_bar():
    """Dots below the operator-blob limit must still make the bar a '÷', not a '-'."""
    _, boxes = segment_expression(make_division_image(dot_radius=3))
    assert len(boxes) == 3
    _, top, _, height = boxes[1]
    assert top <= 50 - 2 and top + height >= 100 + 2   # box spans both dots


def test_small_specks_away_from_a_bar_are_still_dropped():
    img = make_number_image("8 2")
    cv2.line(img, (125, 75), (185, 75), 0, 5)
    cv2.circle(img, (60, 130), 3, 0, -1)      # speck under the 8, no bar there
    _, boxes = segment_expression(img)
    assert len(boxes) == 3
    assert boxes[1][3] < 15                    # the minus stays a thin bar


OPERATOR_METHODS = ["contours", "connected_components", "watershed"]


def make_expression_image(text: str, spacing: int = 70) -> np.ndarray:
    """Draw an expression dark on white. '/' is drawn as a real division sign."""
    width = 40 + spacing * len(text)
    img = np.zeros((160, width), dtype=np.uint8)
    for i, ch in enumerate(text):
        cx = 25 + i * spacing
        if ch == "/":
            cv2.line(img, (cx, 80), (cx + 50, 80), 255, 6)
            cv2.circle(img, (cx + 25, 54), 7, 255, -1)
            cv2.circle(img, (cx + 25, 106), 7, 255, -1)
        else:
            cv2.putText(img, ch, (cx, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.2, 255, 5)
    return cv2.bitwise_not(img)


@pytest.mark.parametrize("method", OPERATOR_METHODS)
@pytest.mark.parametrize("expression", ["26-3", "9*3", "8/2", "(2+3)*4"])
def test_operators_are_found_when_asked_for(method, expression):
    """Every symbol of an expression gets its own box.

    Without keep_operators the digit filter drops '-', '*' and all three pieces
    of a division sign, because each is far shorter than the digits beside it.
    """
    _, boxes = segment_digits(make_expression_image(expression),
                              method=method, keep_operators=True)
    assert len(boxes) == len(expression)


@pytest.mark.parametrize("method", OPERATOR_METHODS)
def test_minus_is_dropped_by_default(method):
    """The digit pipeline must keep rejecting short shapes.

    This is the trade-off the flag exists for: a minus sign and a speck of dust
    are both short, so admitting one admits the other.
    """
    _, boxes = segment_digits(make_expression_image("26-3"), method=method)
    assert len(boxes) == 3


@pytest.mark.parametrize("method", OPERATOR_METHODS)
def test_division_sign_comes_back_as_one_box(method):
    """Bar and both dots must be merged, not returned as three symbols."""
    _, boxes = segment_digits(make_expression_image("8/2"),
                              method=method, keep_operators=True)
    _, _, _, divide_height = boxes[1]
    _, _, _, digit_height = boxes[0]
    assert divide_height > digit_height      # a merged box spans both dots


@pytest.mark.parametrize("method", OPERATOR_METHODS)
def test_operator_mode_still_ignores_noise_specks(method):
    img = make_expression_image("7+1")
    cv2.circle(img, (10, 10), 1, 0, -1)
    cv2.circle(img, (15, 150), 1, 0, -1)
    _, boxes = segment_digits(img, method=method, keep_operators=True)
    assert len(boxes) == 3


def test_segment_expression_orders_symbols_left_to_right():
    _, boxes = segment_expression(make_expression_image("26-3*(17-4)"))
    assert len(boxes) == 11
    assert [b[0] for b in boxes] == sorted(b[0] for b in boxes)


def test_segment_expression_rejects_projection():
    """Named rather than silently wrong: projection cannot see a division sign."""
    with pytest.raises(ValueError, match="division"):
        segment_expression(make_expression_image("8/2"), method="projection")
