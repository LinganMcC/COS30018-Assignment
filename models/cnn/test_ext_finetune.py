"""Tests for the Sprint 4 retraining pieces (no training, no photos needed)."""
import numpy as np
import pytest

from augmentation import cross_sevens, flag_ones, style_augment
from real_symbols import split_of


def test_writer_split_keeps_test_sheets_out_of_training():
    assert split_of("3", "1") == "test" and split_of("4", "7") == "test"
    assert split_of("1", "7") == "val" and split_of("2", "7") == "val"
    assert split_of("1", "1") == "train" and split_of("2", "6") == "train"


def test_style_augment_keeps_shape_and_range():
    x = np.zeros((32, 28, 28, 1), "float32")
    x[:, 6:22, 12:16] = 1.0
    out = style_augment(x).numpy()
    assert out.shape == x.shape
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert (out.reshape(32, -1).max(1) > 0.5).all()      # the stroke survives


@pytest.mark.parametrize("make", [cross_sevens, flag_ones])
def test_synthetic_styles_only_add_ink(make):
    x = np.zeros((4, 28, 28, 1), "float32")
    x[:, 5:23, 13:15] = 1.0                             # a plain vertical stroke
    out = make(x, np.random.default_rng(0))
    assert out.shape == x.shape
    assert (out.sum(axis=(1, 2, 3)) > x.sum(axis=(1, 2, 3))).all()
    assert (out[x > 0.5] > 0.9).all()                    # original ink kept
