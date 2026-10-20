"""Tests for the second-round segmentation repairs (merge, split, crop)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import segmentation_split as S
from segmentation_split import (Piece, cut_piece, dropfall_path, estimate_count,
                                merge_stacked, photo_crop, segment_pieces)


def piece_from(mask: np.ndarray, shape) -> Piece:
    ys, xs = np.nonzero(mask)
    x0, y0 = xs.min(), ys.min()
    return Piece(int(x0), int(y0), int(xs.max() - x0 + 1), int(ys.max() - y0 + 1),
                 mask[y0:ys.max() + 1, x0:xs.max() + 1])


def page_with(*draw) -> np.ndarray:
    """A white page with black strokes, as a photo would be."""
    page = np.full((200, 400, 3), 255, np.uint8)
    for f in draw:
        f(page)
    return page


def test_stacked_fragments_are_merged():
    shape = (200, 200)
    bar = np.zeros(shape, bool); bar[40:46, 50:100] = True           # top bar of a 5
    body = np.zeros(shape, bool); body[60:120, 50:100] = True         # its body, below
    out = merge_stacked([piece_from(bar, shape), piece_from(body, shape)], shape)
    assert len(out) == 1


def test_side_by_side_digits_are_not_merged():
    # a slanted 1 leaning over the 0 next to it shares columns but not rows
    shape = (200, 200)
    one = np.zeros(shape, bool); one[40:120, 60:90] = True
    zero = np.zeros(shape, bool); zero[40:120, 75:130] = True
    out = merge_stacked([piece_from(one, shape), piece_from(zero, shape)], shape)
    assert len(out) == 2


def test_count_estimate_from_shape():
    narrow = Piece(0, 0, 50, 100, np.ones((100, 50), bool))
    wide = Piece(0, 0, 200, 100, np.ones((100, 200), bool))
    assert estimate_count(narrow) == 1
    assert estimate_count(wide) == 3


def test_dropfall_goes_round_a_stroke_and_stays_in_its_window():
    mask = np.zeros((40, 40), bool)
    mask[10:30, 18:22] = True                  # a stroke in the drop's way
    mask[10:30, 23:24] = False
    path = dropfall_path(mask, start=20, lo=12, hi=28)
    assert path.min() >= 12 and path.max() <= 28
    assert len(path) == 40


def test_cut_gives_one_piece_per_digit():
    shape = (100, 200)
    full = np.zeros(shape, bool)
    full[20:80, 20:70] = True                  # two blobs joined by a thin bridge
    full[20:80, 90:140] = True
    full[48:52, 70:90] = True
    p = piece_from(full, shape)
    parts = cut_piece(p, [p.w // 2], shape)
    assert len(parts) == 2
    assert all(part.cut for part in parts)


def test_uncut_piece_is_a_plain_crop():
    image = page_with(lambda im: cv2.rectangle(im, (50, 50), (90, 150), (0, 0, 0), -1))
    binary, pieces = segment_pieces(image, merge=False, split=None)
    crop = photo_crop(image, binary, pieces[0])
    x, y, w, h = pieces[0].box
    assert np.array_equal(crop, image[y:y + h, x:x + w])


def test_cut_piece_paints_out_its_neighbour():
    def two_blobs(im):
        cv2.rectangle(im, (50, 50), (90, 150), (0, 0, 0), -1)
        cv2.rectangle(im, (120, 50), (160, 150), (0, 0, 0), -1)
    image = page_with(two_blobs)
    binary, pieces = segment_pieces(image, merge=False, split=None)
    left, right = pieces
    # a cut piece whose box reaches over its neighbour, as a slanted cut can
    mask = np.zeros((left.h, right.x + right.w - left.x), bool)
    mask[:, :left.w] = left.mask
    wide = S.Piece(left.x, left.y, mask.shape[1], left.h, mask, cut=True)
    crop = photo_crop(image, binary, wide)
    assert crop[:, -10:].min() > 200          # the neighbour is painted out
    assert crop[:, :10].max() < 60            # its own ink is kept


def test_recognition_split_follows_the_model():
    def two_touching(im):                      # two digit-shaped loops, touching
        cv2.ellipse(im, (100, 100), (25, 50), 0, 0, 360, (0, 0, 0), 6)
        cv2.ellipse(im, (150, 100), (25, 50), 0, 0, 360, (0, 0, 0), 6)
    image = page_with(two_touching)

    def confident_on_halves(crops):
        # sure about crops one loop wide, unsure about wider ones
        out = np.full((len(crops), 10), 0.05)
        for i, c in enumerate(crops):
            out[i, 0] = 0.95 if c.shape[1] < 0.8 * c.shape[0] else 0.2
        return out

    _, pieces = segment_pieces(image, merge=True, split="recognition",
                               scorer=confident_on_halves)
    assert len(pieces) == 2
