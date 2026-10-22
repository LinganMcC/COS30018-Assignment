"""Tests for src/number_from_folder.py - image acquisition from a folder of digit images."""
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import number_from_folder as N  # noqa: E402

REAL_DIGITS = Path(__file__).resolve().parent.parent / "data" / "custom_samples" / "digits"


def mnist_like(d: int) -> np.ndarray:
    """28x28, white digit on black, like data/raw/."""
    img = np.zeros((28, 28), np.uint8)
    cv2.putText(img, str(d), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
    return img


def photo_like(d: int, line=False, neighbour=False) -> np.ndarray:
    """Dark digit on light paper, high resolution, thin pen; optionally a ruled line / cut-off neighbour."""
    img = np.full((300, 220), 225, np.uint8)
    cv2.putText(img, str(d), (40, 240), cv2.FONT_HERSHEY_SIMPLEX, 8, 40, 6)
    if line:
        cv2.line(img, (0, 270), (219, 270), 60, 3)
    if neighbour:
        cv2.rectangle(img, (205, 80), (219, 200), 40, -1)      # edge of the next digit
    return img


@pytest.fixture
def subfolder_layout(tmp_path):
    tmp_path = tmp_path / "mnist"
    tmp_path.mkdir()
    for d in range(10):
        (tmp_path / str(d)).mkdir()
        cv2.imwrite(str(tmp_path / str(d) / "a.png"), mnist_like(d))
    return tmp_path


@pytest.fixture
def filename_layout(tmp_path):
    tmp_path = tmp_path / "photos"
    tmp_path.mkdir()
    for d in (1, 2, 3):
        cv2.imwrite(str(tmp_path / f"{d}_01.jpg"), photo_like(d))
    (tmp_path / "notes.txt").write_text("ignored")
    cv2.imwrite(str(tmp_path / "x_01.jpg"), photo_like(4))   # no digit label: ignored
    return tmp_path


def test_both_folder_layouts_are_read(subfolder_layout, filename_layout):
    assert sorted(N.load_folder(subfolder_layout)) == list("0123456789")
    assert sorted(N.load_folder(filename_layout)) == ["1", "2", "3"]


def test_empty_or_missing_folder_is_a_clear_error(tmp_path):
    (tmp_path / "notes.txt").write_text("no images here")
    with pytest.raises(FileNotFoundError):
        N.load_folder(tmp_path)
    with pytest.raises(FileNotFoundError):
        N.load_folder(tmp_path / "nope")


def test_typed_number_is_built_dark_on_white(subfolder_layout):
    img, text = N.make_number(subfolder_layout, "4096", seed=0)
    assert text == "4096" and img.dtype == np.uint8 and img.ndim == 2
    assert img.mean() > 200 and img.min() < 50                 # paper with dark ink
    ink_cols = np.flatnonzero((img < 128).any(axis=0))
    gaps = np.flatnonzero(np.diff(ink_cols) > 1)
    assert len(gaps) == 3                                       # four separate digits


def test_random_number_has_the_length_asked_and_no_leading_zero(subfolder_layout):
    for seed in range(20):
        _, text = N.make_number(subfolder_layout, "", n_digits=5, seed=seed)
        assert len(text) == 5 and text[0] != "0" and text.isdigit()


def test_digit_missing_from_the_folder_is_reported(filename_layout):
    with pytest.raises(ValueError, match="digit 9"):
        N.make_number(filename_layout, "19")
    _, text = N.make_number(filename_layout, "", n_digits=6, seed=1)
    assert set(text) <= {"1", "2", "3"}                          # random uses what is there


def test_text_is_checked(subfolder_layout):
    with pytest.raises(ValueError):
        N.make_number(subfolder_layout, "12a")
    with pytest.raises(ValueError):
        N.make_number(subfolder_layout, "3+4")                   # not an expression
    assert N.parse("3 x (4 ÷ 2)", expression=True) == list("3*(4/2)")


def test_expressions(subfolder_layout):
    img, text = N.make_number(subfolder_layout, "(3+4)*5", expression=True, seed=0)
    assert text == "(3+4)*5" and img.shape[1] > 300
    _, text = N.make_number(subfolder_layout, "", expression=True, seed=3)
    assert set(text) - set(N.DIGITS) <= set(N.OPERATORS)


def test_brackets_face_the_right_way():
    """'(' bulges to the left: its middle rows have ink further left than its ends."""
    for symbol, sign in (("(", 1), (")", -1)):
        tile = N.operator_tile(symbol)
        rows = [np.flatnonzero(r).mean() for r in tile if r.any()]
        ends, middle = (rows[0] + rows[-1]) / 2, rows[len(rows) // 2]
        assert sign * (ends - middle) > 2, symbol


def test_photo_digit_is_cleaned_and_thickened():
    clean = N.ink_tile(photo_like(7))
    messy = N.ink_tile(photo_like(7, line=True, neighbour=True))
    assert abs(messy.shape[1] - clean.shape[1]) <= 4            # line and neighbour removed
    assert abs(messy.shape[0] - clean.shape[0]) <= 4
    tile = N.normalise_digit(clean)
    assert tile.shape[0] == N.DIGIT_HEIGHT
    assert N.stroke_width(tile > 128) >= N.DIGIT_HEIGHT / 10 - 1.5   # not a hairline


@pytest.mark.skipif(not REAL_DIGITS.is_dir(), reason="team digit photos not present")
def test_team_photo_folder_builds_every_digit():
    img, text = N.make_number(REAL_DIGITS, "0123456789", seed=0)
    assert text == "0123456789" and img.shape[0] > N.DIGIT_HEIGHT
