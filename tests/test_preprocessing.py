"""Unit tests for Task 1 preprocessing. Run with: python -m pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import (  # noqa: E402
    CONFIGS, SELECTED_CONFIG, binarize_otsu, center_by_mass,
    invert_if_dark_strokes, normalize, preprocess, resize_image, to_grayscale,
)


@pytest.fixture
def colour_digit():
    """A colour image with a digit drawn on it, dark strokes on white."""
    img = np.full((100, 100, 3), 255, dtype=np.uint8)
    cv2.putText(img, "3", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
    return img


def test_to_grayscale_reduces_channels(colour_digit):
    assert to_grayscale(colour_digit).ndim == 2


def test_to_grayscale_passes_through_grey(colour_digit):
    grey = to_grayscale(colour_digit)
    assert to_grayscale(grey).shape == grey.shape


def test_to_grayscale_rejects_none():
    with pytest.raises(ValueError):
        to_grayscale(None)


def test_resize_produces_target_size(colour_digit):
    assert resize_image(to_grayscale(colour_digit), (28, 28)).shape == (28, 28)


def test_binarize_produces_two_values(colour_digit):
    binary = binarize_otsu(to_grayscale(colour_digit))
    assert set(np.unique(binary)).issubset({0, 255})


def test_invert_flips_bright_images():
    bright = np.full((10, 10), 240, dtype=np.uint8)
    assert invert_if_dark_strokes(bright).mean() < 127


def test_invert_leaves_dark_images_alone():
    dark = np.full((10, 10), 10, dtype=np.uint8)
    assert invert_if_dark_strokes(dark).mean() < 127


def test_normalize_scales_to_unit_range():
    out = normalize(np.array([[0, 255]], dtype=np.uint8))
    assert out.min() == 0.0 and out.max() == 1.0


def test_center_by_mass_handles_empty_image():
    """An all-black image has no centre of mass - must not divide by zero."""
    empty = np.zeros((28, 28), dtype=np.uint8)
    assert center_by_mass(empty).shape == (28, 28)


@pytest.mark.parametrize("config", list(CONFIGS.keys()))
def test_every_config_returns_valid_output(colour_digit, config):
    out = preprocess(colour_digit, config=config)
    assert out.shape == (28, 28)
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_flatten_returns_vector(colour_digit):
    assert preprocess(colour_digit, flatten=True).shape == (784,)


def test_unknown_config_raises(colour_digit):
    with pytest.raises(KeyError):
        preprocess(colour_digit, config="does_not_exist")


def test_preprocess_preserves_the_digit(colour_digit):
    """Output must not be blank - a pipeline that erases the digit is broken."""
    assert preprocess(colour_digit).sum() > 0


def test_selected_config_is_a_real_config():
    """The technique we selected must actually exist, not be a typo."""
    assert SELECTED_CONFIG in CONFIGS


def test_default_matches_the_selected_config(colour_digit):
    """Calling preprocess() with no config must use the selected technique.

    This is the guard against the codebase and the written selection drifting
    apart - the marking scheme asks us to select a technique AND implement it.
    """
    assert (preprocess(colour_digit) == preprocess(colour_digit, config=SELECTED_CONFIG)).all()
