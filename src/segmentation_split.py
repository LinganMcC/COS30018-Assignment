"""Task 2, second round - repairing the boxes that connected-ink segmentation gets wrong.

`contours` finds each connected region of ink. On 99 real numbers that is
right 75 % of the time, and nearly every miss has one of two causes:

    too few boxes    two or more digits touch, so they are one region (22 numbers)
    too many boxes   one digit is drawn in pieces, e.g. the bar of a 5 (3 numbers)

This module keeps connected-ink segmentation as the first pass and repairs its
output in two steps:

    1. merge   pieces that sit in the same columns are joined into one digit
    2. split   a box too wide to be one digit is cut into several

Three ways of choosing where to cut are implemented and compared:

    projection   the emptiest column near each expected cut, inside the box only
    dropfall     a path that falls from the top edge through the gap between
                 two digits, sliding round strokes instead of cutting them
    recognition  several candidate cuts are read by the digit model and the
                 split it is most confident about is kept

Every piece carries its own ink mask, not just a box. When a piece is read,
ink that belongs to other pieces is painted out of its crop (`photo_crop`), so
a box cut from a touching pair does not hand the model half of its neighbour.

Parameters were chosen on the development set only (John's two sheets and the
generated numbers); the other four sheets are held out for the final figure.
See experiments/evaluate_split.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import cv2
import numpy as np

from segmentation import _filter_boxes, _pad_to_square, _prepare_binary

# Chosen on the development set (experiments/evaluate_split.py --tune).
MERGE_OVERLAP = 0.5      # share of the narrower piece's width that must overlap
STACKED_OVERLAP = 0.2    # pieces sharing less of their height than this are stacked
SPLIT_RATIO = 1.25       # width / height above which a box is cut by shape alone
DIGIT_RATIO = 0.65       # typical width / height of one handwritten digit
CUT_WINDOW = 0.3         # a cut is searched within +/- this share of one digit's width
RECOG_GATE = 0.75        # recognition: boxes wider than this (w / h) are tried as 2+ digits
PRIOR_WEIGHT = 0.6       # recognition: cost of disagreeing with the width estimate

# The classifier used by the recognition strategy: a function that takes a
# list of crops (BGR or grey photo crops) and returns an (n, 10) array of
# digit probabilities.
Scorer = Callable[[Sequence[np.ndarray]], np.ndarray]


@dataclass
class Piece:
    """One digit candidate: its box in the image and its own ink."""
    x: int
    y: int
    w: int
    h: int
    mask: np.ndarray              # bool, shape (h, w): this piece's ink inside its box
    cut: bool = False             # True if it came from splitting a wider region

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


# ---------------------------------------------------------------------------
# Building and combining pieces
# ---------------------------------------------------------------------------

def _piece_from_mask(full: np.ndarray) -> Optional[Piece]:
    """A Piece from a full-image bool mask, cropped to its ink."""
    ys, xs = np.nonzero(full)
    if len(xs) == 0:
        return None
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return Piece(int(x0), int(y0), int(x1 - x0), int(y1 - y0), full[y0:y1, x0:x1].copy())


def _paste(piece: Piece, shape: tuple[int, int]) -> np.ndarray:
    full = np.zeros(shape, dtype=bool)
    full[piece.y:piece.y + piece.h, piece.x:piece.x + piece.w] = piece.mask
    return full


def find_pieces(binary: np.ndarray, min_area_ratio: float = 0.002,
                min_height_ratio: float = 0.15) -> list[Piece]:
    """Connected regions of ink, filtered exactly as `connected_components` is."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    boxes = [tuple(int(v) for v in stats[i, :4]) for i in range(1, n)]
    kept = set(_filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio))
    pieces = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in stats[i, :4])
        if (x, y, w, h) in kept:
            pieces.append(Piece(x, y, w, h, labels[y:y + h, x:x + w] == i))
    return sorted(pieces, key=lambda p: p.x)


def merge_stacked(pieces: list[Piece], shape: tuple[int, int],
                  min_overlap: Optional[float] = None) -> list[Piece]:
    """Join pieces that occupy the same columns: parts of one digit.

    A digit written in more than one stroke - a 5 whose top bar does not
    touch its body, a 4 drawn as two strokes - arrives as two regions one
    above the other. Two different digits written left to right share few
    columns, so a large horizontal overlap is the signal. The pieces must also
    be stacked - one above the other, sharing little of their height. A
    slanted 1 leaning over the 0 beside it shares columns too, but it stands
    next to the 0, at the same height, and is left alone.
    """
    min_overlap = MERGE_OVERLAP if min_overlap is None else min_overlap
    pieces = sorted(pieces, key=lambda p: p.x)
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces) - 1):
            a, b = pieces[i], pieces[i + 1]
            overlap = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
            v_overlap = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
            stacked = v_overlap < STACKED_OVERLAP * min(a.h, b.h)
            if stacked and overlap >= min_overlap * min(a.w, b.w):
                merged = _piece_from_mask(_paste(a, shape) | _paste(b, shape))
                pieces[i:i + 2] = [merged]
                changed = True
                break
    return pieces


# ---------------------------------------------------------------------------
# Where to cut
# ---------------------------------------------------------------------------

def estimate_count(piece: Piece, split_ratio: Optional[float] = None,
                   digit_ratio: Optional[float] = None) -> int:
    """How many digits a box probably holds, from its shape alone."""
    split_ratio = SPLIT_RATIO if split_ratio is None else split_ratio
    digit_ratio = DIGIT_RATIO if digit_ratio is None else digit_ratio
    ratio = piece.w / max(1, piece.h)
    if ratio <= split_ratio:
        return 1
    return max(2, int(round(ratio / digit_ratio)))


def _windows(width: int, k: int, window: Optional[float] = None) -> list[tuple[int, int]]:
    """Column ranges in which cut 1..k-1 is searched."""
    window = CUT_WINDOW if window is None else window
    step = width / k
    return [(max(1, int((i - window) * step)), min(width - 1, int((i + window) * step)))
            for i in range(1, k)]


def _profile(mask: np.ndarray) -> np.ndarray:
    """Ink per column, lightly smoothed so one stray pixel does not decide a cut."""
    col = mask.sum(axis=0).astype(float)
    return np.convolve(col, np.ones(3) / 3, mode="same")


def projection_cuts(piece: Piece, k: int, candidates: int = 1) -> list[list[int]]:
    """For each of the k-1 cuts, the `candidates` emptiest columns in its window."""
    prof = _profile(piece.mask)
    options = []
    for lo, hi in _windows(piece.w, k):
        order = lo + np.argsort(prof[lo:hi + 1], kind="stable")
        picked: list[int] = []
        for c in order:                       # keep candidates a few columns apart
            if all(abs(int(c) - p) > 2 for p in picked):
                picked.append(int(c))
            if len(picked) == candidates:
                break
        options.append(picked or [(lo + hi) // 2])
    return options


def dropfall_path(mask: np.ndarray, start: int, lo: int, hi: int) -> np.ndarray:
    """Column of the cut on every row, falling from the top like a drop of water.

    From the top edge the drop moves down while the pixel below is paper. On
    ink it slides down-left or down-right if that pixel is paper, and only when
    both are ink does it cut straight down through the stroke. It is kept
    inside [lo, hi] so it cannot wander into the middle of a neighbouring digit.
    This follows the curved gap between two touching digits, where a straight
    vertical cut would slice the end off one of them.
    """
    h, _ = mask.shape
    path = np.empty(h, dtype=int)
    x = start
    for y in range(h):
        path[y] = x
        if y == h - 1:
            break
        below = mask[y + 1]
        if not below[x]:
            continue
        if x - 1 >= lo and not below[x - 1]:
            x -= 1
        elif x + 1 <= hi and not below[x + 1]:
            x += 1
    return path


def cut_piece(piece: Piece, cuts: Sequence[int], shape: tuple[int, int],
              dropfall: bool = False) -> list[Piece]:
    """Cut a piece at the given columns (box coordinates), straight or by drop-fall."""
    h, w = piece.mask.shape
    cols = np.arange(w)[None, :]
    bounds = []
    windows = _windows(w, len(cuts) + 1)
    for c, (lo, hi) in zip(cuts, windows):
        if dropfall:
            bounds.append(dropfall_path(piece.mask, int(c), lo, hi)[:, None])
        else:
            bounds.append(np.full((h, 1), int(c)))
    edges = [np.full((h, 1), -1)] + bounds + [np.full((h, 1), w)]
    parts = []
    for left, right in zip(edges, edges[1:]):
        local = piece.mask & (cols > left) & (cols <= right)
        if local.sum() == 0:
            continue
        full = np.zeros(shape, dtype=bool)
        full[piece.y:piece.y + h, piece.x:piece.x + w] = local
        part = _piece_from_mask(full)
        if part is not None:
            part.cut = True
            parts.append(part)
    return parts


# ---------------------------------------------------------------------------
# Reading a piece from the photo
# ---------------------------------------------------------------------------

def photo_crop(image: np.ndarray, binary: np.ndarray, piece: Piece) -> np.ndarray:
    """The piece's box cut from the photo; for a cut piece, the neighbour painted out.

    A piece cut from a touching pair has its neighbour's ink inside its box,
    so those pixels are replaced by the paper colour and the classifier sees
    this digit alone. The piece's own pixels keep their photographed grey
    values, which is what the preprocessing expects.

    A piece that was not cut is returned as a plain box crop, exactly as the
    contours pipeline crops it. Painting out on every piece was measured and
    rejected: on the development set it lowered exact reading of separate
    numbers from 79 % to 74 %, because a digit drawn in two strokes - the stem
    of a 9 apart from its loop - lost the stroke that sat in another region.
    """
    x, y, w, h = piece.box
    crop = image[y:y + h, x:x + w].copy()
    if not piece.cut:
        return crop
    # The binary mask holds the core of each stroke; its anti-aliased grey edge
    # lies just outside. Growing the foreign ink by a few pixels removes that
    # edge too, so no faint outline of the neighbour is left for Otsu to find.
    other = ((binary[y:y + h, x:x + w] > 0) & ~piece.mask).astype(np.uint8)
    k = max(1, int(round(h / 40)))       # about the stroke edge at this scale
    grown = cv2.dilate(other, np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0
    own = cv2.dilate(piece.mask.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    foreign = grown & ~own
    if foreign.any():
        paper_pixels = image[binary == 0]
        paper = np.median(paper_pixels, axis=0) if len(paper_pixels) else 255
        crop[foreign] = paper
    return crop


# ---------------------------------------------------------------------------
# The segmenter
# ---------------------------------------------------------------------------

def _recognition_split(image: np.ndarray, binary: np.ndarray, piece: Piece,
                       scorer: Scorer, prior_weight: Optional[float] = None,
                       max_digits: int = 4) -> list[Piece]:
    """Try 1..max_digits digits, several cuts each; keep what the model reads best.

    Score of one option = log-confidence of its least certain piece, minus a penalty for
    disagreeing with the count the box's width suggests. The penalty stops the
    model from cutting a clear 0 into two confident-looking 1s.
    """
    prior_weight = PRIOR_WEIGHT if prior_weight is None else prior_weight
    shape = binary.shape
    expected = estimate_count(piece)
    options: list[list[Piece]] = [[piece]]
    for k in range(2, max_digits + 1):
        if piece.w / max(1, piece.h) < 0.3 * k:        # far too narrow for k digits
            break
        per_cut = projection_cuts(piece, k, candidates=2)
        combos = [[]]
        for choices in per_cut:
            combos = [c + [x] for c in combos for x in choices]
        for cuts in combos:
            for drop in (False, True):
                parts = cut_piece(piece, cuts, shape, dropfall=drop)
                if len(parts) == k:
                    options.append(parts)
    crops = [photo_crop(image, binary, p) for opt in options for p in opt]
    probs = scorer(crops)
    best, best_score, i = options[0], -np.inf, 0
    for opt in options:
        conf = probs[i:i + len(opt)].max(axis=1)
        i += len(opt)
        # the weakest piece decides: one implausible piece sinks the option
        score = float(np.log(conf.min() + 1e-9)) - prior_weight * abs(len(opt) - expected)
        if score > best_score:
            best, best_score = opt, score
    return best


def segment_pieces(image: np.ndarray, merge: bool = True, split: Optional[str] = "dropfall",
                   scorer: Optional[Scorer] = None) -> tuple[np.ndarray, list[Piece]]:
    """Connected ink, then the two repairs. Returns (binary, pieces left to right).

    split: None, "projection", "dropfall" or "recognition" (needs `scorer`).
    """
    binary = _prepare_binary(image)
    pieces = find_pieces(binary)
    if merge:
        pieces = merge_stacked(pieces, binary.shape)
    if split:
        out = []
        for p in pieces:
            if split == "recognition":
                if scorer is None:
                    raise ValueError("split='recognition' needs a scorer")
                if p.w / max(1, p.h) > RECOG_GATE:    # only boxes wide enough to hold two
                    out.extend(_recognition_split(image, binary, p, scorer))
                else:
                    out.append(p)
                continue
            k = estimate_count(p)
            if k == 1:
                out.append(p)
                continue
            cuts = [c[0] for c in projection_cuts(p, k)]
            out.extend(cut_piece(p, cuts, binary.shape, dropfall=(split == "dropfall")))
        pieces = out
    return binary, sorted(pieces, key=lambda p: p.x)


def _as_method(merge: bool, split: Optional[str]):
    """Wrap a configuration in the (crops, boxes) interface of segmentation.METHODS."""
    def segment(image: np.ndarray, margin: int = 4, **_ignored):
        binary, pieces = segment_pieces(image, merge=merge, split=split)
        crops = [_pad_to_square((p.mask * 255).astype(np.uint8), margin) for p in pieces]
        return crops, [p.box for p in pieces]
    segment.__name__ = f"segment_{'merge_' if merge else ''}{split or 'none'}"
    return segment


# The model-free configurations, usable anywhere a segmentation method is.
# "recognition" needs a model, so it is called through segment_pieces().
SPLIT_METHODS = {
    "merge": _as_method(True, None),
    "merge_projection": _as_method(True, "projection"),
    "merge_dropfall": _as_method(True, "dropfall"),
}


def segment_digits_repaired(image: np.ndarray, scorer: Scorer
                            ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """The full repaired segmenter, for the system: ruled lines out, merge, split.

    Returns (photo crops, boxes) left to right. The crops are already cut from
    the photo (with a cut piece's neighbour painted out), so they go straight
    to preprocess() - unlike segment_digits(), whose crops are binary and whose
    boxes the caller cuts from the photo itself.

    scorer = any function from a list of photo crops to an (n, 10) array of
    digit probabilities, e.g. the team model behind preprocess(...).
    """
    from page_cleanup import remove_ruled_lines
    image = remove_ruled_lines(image)          # boxes refer to this cleaned photo
    binary, pieces = segment_pieces(image, merge=True, split="recognition", scorer=scorer)
    return [photo_crop(image, binary, p) for p in pieces], [p.box for p in pieces]
