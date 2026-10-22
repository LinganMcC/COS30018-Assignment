"""Cleaning a photo before it is segmented: ruled lines, and slanted writing.

Both work on the photo itself and return a photo, so everything after them -
segmentation, cropping, preprocessing - runs unchanged on a cleaner input.

    remove_ruled_lines   paints the lines of lined paper out with the paper colour
    deskew               shears slanted writing upright

Why these two. After the second-round segmentation repair, 32 of the 99 real
numbers were still read wrongly: 11 of them on the one lined sheet, and 17
because touching digits were not separated. A ruled line joins every digit on
it into one region and, inside a crop, looks like the base of a 2. Slanted
writing makes neighbouring digits share columns, so no vertical cut can part
them; sheared upright, the gap between them often reappears.
"""

from __future__ import annotations

import cv2
import numpy as np

from preprocessing import binarize_otsu, to_grayscale

LINE_SPAN = 0.85        # a ruled line crosses at least this share of the image width
LINE_DRIFT = 0.04       # it may rise or fall by this share of the image height
MAX_SHEAR = 0.6         # deskew never shears by more than this (about 31 degrees)


def _ink(image: np.ndarray) -> np.ndarray:
    return binarize_otsu(to_grayscale(image))


def _paper_colour(image: np.ndarray, ink: np.ndarray):
    paper = image[ink == 0]
    return np.median(paper, axis=0) if len(paper) else 255


def ruled_line_mask(image: np.ndarray, span: float = LINE_SPAN,
                    drift: float = LINE_DRIFT) -> np.ndarray:
    """Pixels that belong to a ruled line, as a bool mask.

    A ruled line runs almost the whole width of a crop cut from a lined sheet,
    edge to edge. No stroke of a digit does: the crops keep a margin of paper
    round the writing, so even the long base of a 2 stops well short of both
    edges. The test is therefore a horizontal run of ink, allowed to tilt a
    little, that covers `span` of the width.
    """
    ink = _ink(image)
    h, w = ink.shape
    length = max(15, int(span * w))
    rise = max(1, int(drift * h))
    tall = cv2.dilate(ink, np.ones((2 * rise + 1, 1), np.uint8))
    runs = cv2.morphologyEx(tall, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    line = cv2.bitwise_and(cv2.dilate(runs, np.ones((3, 3), np.uint8)), ink) > 0

    # Keep only line ink that is thin at each column: where a digit crosses the
    # line, the stroke above and below the line is the digit's and stays.
    thickness = max(2, int(0.03 * h))
    keep = np.zeros_like(line)
    for x in np.nonzero(line.any(axis=0))[0]:
        ys = np.nonzero(line[:, x])[0]
        if len(ys) <= 3 * thickness:
            keep[ys, x] = True
    return keep


def remove_ruled_lines(image: np.ndarray) -> np.ndarray:
    """The photo with any ruled line painted over in the paper colour."""
    mask = ruled_line_mask(image)
    if not mask.any():
        return image
    out = image.copy()
    grown = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    out[grown] = _paper_colour(image, _ink(image))
    return out


def slant(image: np.ndarray) -> float:
    """Horizontal shift per row of the writing, from the digits themselves.

    For each region of ink tall enough to be a digit, mu11 / mu02 is how far
    its ink moves sideways per row; writing that leans right (top further right
    than base) gives a negative value, as x falls while y grows down the image.
    This is the deskewing used on MNIST-style digits. It is measured per
    region and the median taken, not over the whole number at once: a number
    written along a rising line would otherwise read as slanted even if every
    digit in it stood upright.
    """
    ink = _ink(image)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    if n <= 1:
        return 0.0
    tallest = stats[1:, cv2.CC_STAT_HEIGHT].max()
    slants, weights = [], []
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_HEIGHT] < 0.5 * tallest:
            continue
        m = cv2.moments((labels == i).astype(np.uint8), binaryImage=True)
        if m["mu02"] > 1e-6:
            slants.append(m["mu11"] / m["mu02"])
            weights.append(stats[i, cv2.CC_STAT_AREA])
    if not slants:
        return 0.0
    order = np.argsort(slants)
    cum = np.cumsum(np.array(weights)[order])
    return float(np.array(slants)[order][np.searchsorted(cum, cum[-1] / 2)])


def deskew(image: np.ndarray, max_shear: float = MAX_SHEAR) -> np.ndarray:
    """Shear the photo so its writing stands upright.

    The image is widened so nothing is sheared off its edge, and the new
    corners are filled with the paper colour.
    """
    s = float(np.clip(slant(image), -max_shear, max_shear))
    if abs(s) < 0.05:                       # already upright enough
        return image
    h, w = image.shape[:2]
    pad = int(abs(s) * h) + 2
    # x' = x - s * (y - h / 2), moved right by pad / 2 so both sides have room
    matrix = np.float32([[1, -s, s * h / 2 + pad / 2], [0, 1, 0]])
    paper = np.atleast_1d(_paper_colour(image, _ink(image))).astype(float)
    border = tuple(paper.tolist()) if image.ndim == 3 else float(paper[0])
    return cv2.warpAffine(image, matrix, (w + pad, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def clean(image: np.ndarray, lines: bool = True, straighten: bool = True) -> np.ndarray:
    """Both steps, lines first: a ruled line would distort the slant estimate."""
    if lines:
        image = remove_ruled_lines(image)
    if straighten:
        image = deskew(image)
    return image
