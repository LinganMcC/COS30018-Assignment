"""Unit tests for Task 1 preprocessing. Run: python -m pytest tests/ -v"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from preprocessing import (
    CONFIGS, SELECTED_CONFIG, binarize_otsu, center_by_mass, fit_to_mnist_box,
    invert_if_dark_strokes, normalize, preprocess, resize_image, to_grayscale,
)


@pytest.fixture
def colour_digit():
    """Colour image of a digit, dark on white."""
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
    """Empty image has no centre of mass, so it must not divide by zero."""
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
    """The digit must not be erased by the pipeline."""
    assert preprocess(colour_digit).sum() > 0


def test_selected_config_is_a_real_config():
    """SELECTED_CONFIG must be a real key in CONFIGS."""
    assert SELECTED_CONFIG in CONFIGS


def test_default_matches_the_selected_config(colour_digit):
    """preprocess() with no config must use SELECTED_CONFIG."""
    assert (preprocess(colour_digit) == preprocess(colour_digit, config=SELECTED_CONFIG)).all()


# --- MNIST 20x20 box -------------------------------------------------------

@pytest.fixture
def white_on_black_digit():
    """A digit drawn white on black, which is what fit_to_mnist_box expects."""
    img = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(img, "7", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)
    return img


def test_mnist_box_output_is_target_size(white_on_black_digit):
    assert fit_to_mnist_box(white_on_black_digit).shape == (28, 28)


def test_mnist_box_keeps_the_digit_inside_20_pixels(white_on_black_digit):
    """The whole point: neither side of the digit may exceed the 20px box."""
    out = fit_to_mnist_box(white_on_black_digit)
    _, _, w, h = cv2.boundingRect(cv2.findNonZero(out))
    assert w <= 20 and h <= 20


def test_mnist_box_leaves_a_border(white_on_black_digit):
    """A digit in a 20x20 box inside 28x28 cannot touch the frame edge."""
    out = fit_to_mnist_box(white_on_black_digit)
    assert out[0, :].sum() == 0 and out[-1, :].sum() == 0
    assert out[:, 0].sum() == 0 and out[:, -1].sum() == 0


def test_mnist_box_preserves_aspect_ratio():
    """A tall thin stroke must stay tall and thin, not be stretched square."""
    tall = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(tall, (48, 10), (52, 90), 255, -1)
    out = fit_to_mnist_box(tall)
    _, _, w, h = cv2.boundingRect(cv2.findNonZero(out))
    assert h > w


def test_mnist_box_handles_blank_image():
    blank = np.zeros((50, 50), dtype=np.uint8)
    out = fit_to_mnist_box(blank)
    assert out.shape == (28, 28) and out.sum() == 0


def test_mnist_box_configs_run(colour_digit):
    for name in ("grayscale_mnist_box", "adaptive_mnist_box", "otsu_mnist_box"):
        assert preprocess(colour_digit, config=name).shape == (28, 28)


@pytest.fixture
def photo_like_digit():
    """Off-white textured paper with a shading gradient, like a phone photo.

    The paper is never pure white, so after inversion no background pixel is
    exactly 0. That is the case clean MNIST-style tests never exercise.
    """
    rng = np.random.default_rng(0)
    h, w = 600, 800
    paper = 200 + rng.normal(0, 8, (h, w)) + np.linspace(-25, 25, w)[None, :]
    paper = paper.clip(0, 255).astype(np.uint8)
    cv2.putText(paper, "7", (300, 420), cv2.FONT_HERSHEY_SIMPLEX, 9, 40, 22)
    return cv2.cvtColor(paper, cv2.COLOR_GRAY2BGR)


def test_mnist_box_finds_the_digit_on_a_photo(photo_like_digit):
    """Regression: the box used to become the whole frame on non-black paper."""
    out = (preprocess(photo_like_digit, config="grayscale_mnist_box") * 255).astype(np.uint8)
    ys, xs = np.where(out > 127)
    assert max(np.ptp(xs), np.ptp(ys)) + 1 >= 18      # digit fills the 20px box


def test_adaptive_keeps_thick_strokes_on_a_large_image(photo_like_digit):
    """Regression: a fixed 11px block hollowed out strokes wider than itself."""
    from preprocessing import binarize_adaptive, denoise
    grey = denoise(to_grayscale(photo_like_digit))
    binary = binarize_adaptive(grey)
    stroke = grey < 100
    assert binary[stroke].mean() / 255 > 0.9


# --- model input shape -----------------------------------------------------

def test_add_channel_matches_what_conv2d_expects(colour_digit):
    """Thien's models take (28, 28, 1), not (28, 28)."""
    assert preprocess(colour_digit, add_channel=True).shape == (28, 28, 1)


def test_flatten_and_add_channel_are_mutually_exclusive(colour_digit):
    with pytest.raises(ValueError):
        preprocess(colour_digit, flatten=True, add_channel=True)


def test_tight_crop_of_a_thick_digit_is_still_inverted():
    """Regression: a crop that is mostly ink used to be left dark-on-light.

    The old rule inverted only when the whole image was bright on average. A
    tight box around a thick stroke can be more than half ink, so it stayed
    black-on-white and the model saw the opposite of what it was trained on.
    The rule now looks at the border, which is paper however much ink is inside.
    """
    crop = np.full((60, 40), 230, dtype=np.uint8)
    cv2.rectangle(crop, (4, 0), (36, 59), 30, -1)          # ink fills most of the box
    assert crop.mean() < 127                                # the old rule would not invert
    out = invert_if_dark_strokes(crop)
    assert out[30, 20] > 127                                # stroke is now light
