"""Task 2 - segmentation techniques that use a trained model.

The methods in segmentation.py are pure image processing: they decide where a
digit ends by looking at pixels. The ones here ask a classifier instead, which
is the point of the comparison. A method that relies on connectivity cannot
separate two digits whose ink touches, no matter how it is tuned. A method
that asks "does this window look like a digit" can.

Four of them hold the search strategy fixed and vary only the model, so the
results isolate the model:

    sliding_mlp   sliding_cnn   sliding_svm   sliding_rf

Two more vary the strategy instead, and answer a different question:

    cutpoint_mlp       over-segment first, let the model choose which cuts to keep
    confidence_split   split wide blobs where the model is most confident

All of them return (crops, boxes) ordered left to right, exactly like the methods
in segmentation.py, so demo_segmentation.py can score them side by side.
"""

from __future__ import annotations

import numpy as np

from digit_classifier import NOT_A_DIGIT, get_classifier, to_patch
from segmentation import _filter_boxes, _pad_to_square, _prepare_binary

import cv2

PATCH = 28


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ink_bounds(binary: np.ndarray):
    """The region actually containing ink, as (x, y, w, h)."""
    ink = cv2.findNonZero(binary)
    return None if ink is None else cv2.boundingRect(ink)


def _score_windows(binary: np.ndarray, boxes, classifier) -> np.ndarray:
    """Classify a batch of candidate boxes. Returns one probability row each."""
    if not boxes:
        return np.zeros((0, NOT_A_DIGIT + 1))
    patches = [to_patch(binary[y:y + h, x:x + w]) for (x, y, w, h) in boxes]
    return classifier.predict_proba(np.array(patches))


def _digit_confidence(probs: np.ndarray) -> np.ndarray:
    """Confidence that a patch is some digit, i.e. anything but the null class."""
    return probs[:, :NOT_A_DIGIT].max(axis=1)


def _suppress_overlaps(boxes, scores, iou_limit: float = 0.3):
    """Keep the highest-scoring box in any cluster of overlapping boxes.

    Sliding windows fire several times around the same digit. Without this the
    method would report one digit as three.
    """
    order = np.argsort(scores)[::-1]
    kept = []
    for i in order:
        x1, _, w1, _ = boxes[i]
        if all(_x_overlap((x1, w1), (boxes[j][0], boxes[j][2])) <= iou_limit for j in kept):
            kept.append(i)
    return sorted(kept, key=lambda i: boxes[i][0])


def _x_overlap(a, b) -> float:
    """Horizontal overlap of two spans, as a fraction of the narrower one."""
    (x1, w1), (x2, w2) = a, b
    left, right = max(x1, x2), min(x1 + w1, x2 + w2)
    if right <= left:
        return 0.0
    return (right - left) / min(w1, w2)


def _crops_for(binary: np.ndarray, boxes, margin: int = 4):
    return [_pad_to_square(binary[y:y + h, x:x + w], margin) for (x, y, w, h) in boxes]


def _row_extent(binary: np.ndarray, x_start: int, x_end: int):
    """Vertical extent of the ink inside a column band."""
    band = binary[:, x_start:x_end]
    rows = np.where(band.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return int(rows[0]), int(rows[-1] - rows[0] + 1)


# ---------------------------------------------------------------------------
# 1 and 2 - sliding window, MLP or CNN
# ---------------------------------------------------------------------------

def segment_by_sliding_window(image: np.ndarray, classifier: str = "mlp",
                              confidence: float = 0.5, floor: float = 0.15,
                              step_ratio: float = 0.12,
                              width_ratios=(0.55, 0.75, 0.95, 1.15),
                              min_area_ratio: float = 0.002,
                              min_height_ratio: float = 0.15, margin: int = 4):
    """Slide a window across the ink and keep the positions the model likes.

    Window widths are expressed as fractions of the ink height, because a digit
    is roughly as tall as it is wide and the ink height is the one measurement
    available before knowing where the digits are.

    `confidence` is relative, not absolute: a window is kept if it scores at
    least that fraction of the best score in the same image. An absolute
    threshold cannot work across four different models, because their
    probabilities are on different scales. The MLP's softmax peaks near 1.0,
    while the SVM's Platt scaling and the forest's vote-averaging both top out
    around 0.5, so a fixed 0.55 cut-off silently returned nothing at all for
    two of the four models. Scoring relative to the best window in the image
    sidesteps the calibration problem entirely. `floor` is the absolute minimum
    that keeps a blank page from promoting its own noise.
    """
    model = get_classifier(classifier)
    binary = _prepare_binary(image)
    bounds = _ink_bounds(binary)
    if bounds is None:
        return [], []

    ink_x, ink_y, ink_w, ink_h = bounds
    step = max(1, int(step_ratio * ink_h))

    candidates = []
    for ratio in width_ratios:
        win = max(4, int(ratio * ink_h))
        for x in range(ink_x, ink_x + ink_w - win // 2, step):
            w = min(win, ink_x + ink_w - x)
            if w < 4:
                continue
            extent = _row_extent(binary, x, x + w)
            if extent is None:
                continue
            y, h = extent
            candidates.append((x, y, w, h))

    if not candidates:
        return [], []

    probs = _score_windows(binary, candidates, model)
    scores = _digit_confidence(probs)

    cutoff = max(floor, confidence * float(scores.max()))
    strong = [i for i, s in enumerate(scores) if s >= cutoff]
    if not strong:
        return [], []

    keep = _suppress_overlaps([candidates[i] for i in strong], scores[strong])
    boxes = [candidates[strong[i]] for i in keep]
    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio)
    boxes.sort(key=lambda b: b[0])
    return _crops_for(binary, boxes, margin), boxes


def _sliding_with(classifier: str):
    """Build a sliding-window method bound to one model.

    The search is identical in all four, so the only thing that differs between
    them is the model doing the scoring. That is what makes the comparison mean
    something: any gap in the results is the model, not the strategy.
    """
    def method(image: np.ndarray, **kwargs):
        kwargs.setdefault("classifier", classifier)
        return segment_by_sliding_window(image, **kwargs)

    method.__name__ = f"segment_by_sliding_{classifier}"
    method.__doc__ = f"Sliding-window segmentation scored by the {classifier} model."
    return method


segment_by_sliding_mlp = _sliding_with("mlp")
segment_by_sliding_cnn = _sliding_with("cnn")
segment_by_sliding_svm = _sliding_with("svm")
segment_by_sliding_rf = _sliding_with("random_forest")


# ---------------------------------------------------------------------------
# 3 - over-segment, then let the model pick the cuts
# ---------------------------------------------------------------------------

def _candidate_cuts(binary: np.ndarray, ink_x: int, ink_w: int,
                    max_cuts: int = 24) -> list[int]:
    """Columns that could plausibly be a boundary: the local minima of ink."""
    profile = (binary > 0).sum(axis=0).astype(float)
    span = profile[ink_x:ink_x + ink_w]
    if len(span) < 3:
        return []

    minima = []
    for i in range(1, len(span) - 1):
        if span[i] <= span[i - 1] and span[i] <= span[i + 1]:
            minima.append((span[i], ink_x + i))

    minima.sort(key=lambda pair: pair[0])
    chosen = sorted({x for _, x in minima[:max_cuts]})
    return chosen


def segment_by_cut_points(image: np.ndarray, classifier: str = "mlp",
                          min_area_ratio: float = 0.002,
                          min_height_ratio: float = 0.15, margin: int = 4):
    """Over-segment at every plausible boundary, then keep the best set of cuts.

    This is the classical approach for touching handwriting, used in cheque
    readers. The image processing proposes; the model disposes. A dynamic
    program walks left to right and picks the sequence of cuts whose resulting
    pieces the model is most confident about, so a cut that splits a digit in
    half is rejected because both halves score badly.
    """
    model = get_classifier(classifier)
    binary = _prepare_binary(image)
    bounds = _ink_bounds(binary)
    if bounds is None:
        return [], []

    ink_x, _, ink_w, ink_h = bounds
    cuts = [ink_x] + _candidate_cuts(binary, ink_x, ink_w) + [ink_x + ink_w]
    cuts = sorted(set(cuts))
    if len(cuts) < 2:
        return [], []

    # Score every piece between any two cut points, within a plausible width.
    min_w, max_w = max(3, int(0.25 * ink_h)), int(1.6 * ink_h)
    pieces, spans = [], []
    for a in range(len(cuts) - 1):
        for b in range(a + 1, len(cuts)):
            width = cuts[b] - cuts[a]
            if not (min_w <= width <= max_w):
                continue
            extent = _row_extent(binary, cuts[a], cuts[b])
            if extent is None:
                continue
            y, h = extent
            pieces.append((cuts[a], y, width, h))
            spans.append((a, b))

    if not pieces:
        return [], []

    scores = _digit_confidence(_score_windows(binary, pieces, model))
    reward = {span: float(np.log(max(s, 1e-6))) for span, s in zip(spans, scores)}
    piece_at = {span: box for span, box in zip(spans, pieces)}

    # best[i] = (total score of the best path from cut i to the end, first span)
    best: dict[int, tuple[float, tuple | None]] = {len(cuts) - 1: (0.0, None)}
    for a in range(len(cuts) - 2, -1, -1):
        options = [(reward[(a, b)] + best[b][0], (a, b))
                   for b in range(a + 1, len(cuts)) if (a, b) in reward and b in best]
        best[a] = max(options) if options else (-np.inf, None)

    boxes, node = [], 0
    while node in best and best[node][1] is not None:
        span = best[node][1]
        boxes.append(piece_at[span])
        node = span[1]

    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio)
    boxes.sort(key=lambda b: b[0])
    return _crops_for(binary, boxes, margin), boxes


# ---------------------------------------------------------------------------
# 4 - split blobs that are too wide, where the model says to
# ---------------------------------------------------------------------------

def segment_by_confidence_split(image: np.ndarray, classifier: str = "mlp",
                                min_area_ratio: float = 0.002,
                                min_height_ratio: float = 0.15, margin: int = 4):
    """Start from connected blobs, then split the ones that hold several digits.

    A blob much wider than it is tall is almost certainly more than one digit.
    Rather than cutting it into equal parts, every candidate cut column is
    tried and the one that leaves the model most confident about both sides
    wins. Cheaper than the sliding window because the model is only consulted
    where there is a real decision to make.
    """
    model = get_classifier(classifier)
    binary = _prepare_binary(image)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    blobs = [(stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
              stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
             for i in range(1, n_labels)]
    blobs = _filter_boxes(blobs, binary.shape, min_area_ratio, min_height_ratio)
    if not blobs:
        return [], []

    reference = max(h for (_, _, _, h) in blobs)      # tallest blob sets the scale

    boxes = []
    for (x, y, w, h) in blobs:
        expected = max(1, int(round(w / (0.85 * reference))))
        boxes.extend(_split_blob(binary, (x, y, w, h), expected, model))

    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio)
    boxes.sort(key=lambda b: b[0])
    return _crops_for(binary, boxes, margin), boxes


def _split_blob(binary: np.ndarray, box, parts: int, model, search: int = 6):
    """Cut one blob into `parts` pieces, nudging each cut to the best column."""
    x, y, w, h = box
    if parts <= 1:
        return [box]

    edges = [x + round(i * w / parts) for i in range(parts + 1)]
    for i in range(1, parts):
        lo, hi = max(x + 2, edges[i] - search), min(x + w - 2, edges[i] + search)
        if hi <= lo:
            continue
        options = []
        for cut in range(lo, hi + 1):
            left = _piece(binary, edges[i - 1], cut)
            right = _piece(binary, cut, edges[i + 1])
            if left is None or right is None:
                continue
            probs = _score_windows(binary, [left, right], model)
            options.append((float(_digit_confidence(probs).mean()), cut))
        if options:
            edges[i] = max(options)[1]

    out = []
    for i in range(parts):
        piece = _piece(binary, edges[i], edges[i + 1])
        if piece is not None:
            out.append(piece)
    return out


def _piece(binary: np.ndarray, x_start: int, x_end: int):
    if x_end - x_start < 3:
        return None
    extent = _row_extent(binary, x_start, x_end)
    if extent is None:
        return None
    y, h = extent
    return (x_start, y, x_end - x_start, h)


# The four-model comparison: one search strategy, four different models.
MODEL_METHODS = {
    "sliding_mlp": segment_by_sliding_mlp,
    "sliding_cnn": segment_by_sliding_cnn,
    "sliding_svm": segment_by_sliding_svm,
    "sliding_rf": segment_by_sliding_rf,
}

# Two further model-based strategies, kept separate because they change the
# search rather than the model, and so answer a different question.
STRATEGY_METHODS = {
    "cutpoint_mlp": segment_by_cut_points,
    "confidence_split": segment_by_confidence_split,
}

ML_METHODS = {**MODEL_METHODS, **STRATEGY_METHODS}
