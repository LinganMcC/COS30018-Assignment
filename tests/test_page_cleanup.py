"""Tests for photo cleanup before segmentation: ruled lines and slant."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from page_cleanup import deskew, remove_ruled_lines, ruled_line_mask, slant


def page(h=200, w=300):
    return np.full((h, w, 3), 255, np.uint8)


def dark(image):
    return (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) < 128)


def test_ruled_line_is_removed_and_the_digit_kept():
    im = page()
    cv2.line(im, (0, 150), (299, 154), (40, 40, 40), 3)          # line edge to edge
    cv2.putText(im, "4", (110, 145), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8)
    out = remove_ruled_lines(im)
    assert not dark(out)[148:157, :60].any()                        # line gone at the side
    assert dark(out)[60:140, 100:200].sum() > 0.9 * dark(im)[60:140, 100:200].sum()


def test_long_digit_stroke_with_margin_is_not_a_line():
    im = page()
    cv2.line(im, (60, 150), (240, 150), (0, 0, 0), 6)               # base of a 2, inside the margin
    assert not ruled_line_mask(im).any()


def test_no_line_means_no_change():
    im = page()
    cv2.putText(im, "38", (60, 140), cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 8)
    assert np.array_equal(remove_ruled_lines(im), im)


def test_slant_sign_and_deskew():
    im = page()
    cv2.line(im, (130, 180), (170, 20), (0, 0, 0), 6)               # leans right
    assert slant(im) < 0
    ys, xs = np.nonzero(dark(deskew(im)))
    assert abs(xs[ys < 40].mean() - xs[ys > 160].mean()) < 6       # now upright


def test_upright_writing_is_left_alone():
    im = page()
    cv2.line(im, (150, 20), (150, 180), (0, 0, 0), 6)
    assert deskew(im) is im
