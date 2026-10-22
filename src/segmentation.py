"""Task 2 - split an image of a number into single-digit images (contours or connected components)."""

from __future__ import annotations

import cv2
import numpy as np

from preprocessing import to_grayscale, binarize_otsu, denoise


def _prepare_binary(image: np.ndarray, apply_denoise: bool = True) -> np.ndarray:
    grey = to_grayscale(image)
    if apply_denoise:
        grey = denoise(grey)
    return binarize_otsu(grey)


def _pad_to_square(crop: np.ndarray, margin: int = 4) -> np.ndarray:
    """Pad a crop to a square so a thin '1' is not stretched when resized."""
    h, w = crop.shape
    side = max(h, w) + 2 * margin
    square = np.zeros((side, side), dtype=crop.dtype)
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    square[y0:y0 + h, x0:x0 + w] = crop
    return square


def _is_operator_shape(box: tuple[int, int, int, int], reference_height: int) -> bool:
    """True for the two shapes an operator takes that the digit filter rejects.

    The digit filter measures a box against the image, which works because a
    digit always fills a good share of the frame. An operator does not: a minus
    sign is 5% of the image height, an asterisk 13%, and the bar and dots of a
    division sign less again. They are measured against the tallest box in the
    same image instead - the digits set the scale, whatever the photo's margins.

    bar  - a minus sign, or the middle stroke of a division sign: several times
           wider than it is tall, which no digit and no operator else is.
    blob - an asterisk, or one of the two dots of a division sign: roughly
           square, clearly smaller than a digit but far larger than dust.
    """
    _, _, w, h = box
    if h == 0 or reference_height == 0:
        return False
    if w >= 2.0 * h and w >= 0.3 * reference_height:
        return True
    return (0.15 * reference_height <= h <= 0.6 * reference_height
            and 0.5 <= w / h <= 2.0
            and w >= 0.15 * reference_height)


def _merge_stacked(boxes: list[tuple[int, int, int, int]], min_overlap: float = 0.6
                   ) -> list[tuple[int, int, int, int]]:
    """Merge boxes that sit above one another in one column into a single box.

    A division sign is three separate ink regions - bar, dot, dot - so every
    method here returns it as three boxes. In a left-to-right expression no two
    different symbols share a column, so horizontal overlap alone is enough to
    put the pieces back together. Overlap is measured against the narrower of
    the two boxes, because the dots are much narrower than the bar they belong to.

    Only ever called on the operator-shaped boxes. Run over digits as well it
    cost 2 of 27 real numbers, because slanted handwriting puts one digit far
    enough over its neighbour's column to look like a stacked pair.
    """
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                (x1, y1, w1, h1), (x2, y2, w2, h2) = merged[i], merged[j]
                overlap = min(x1 + w1, x2 + w2) - max(x1, x2)
                if overlap <= 0 or overlap < min_overlap * min(w1, w2):
                    continue
                left, top = min(x1, x2), min(y1, y2)
                merged[i] = (left, top,
                             max(x1 + w1, x2 + w2) - left,
                             max(y1 + h1, y2 + h2) - top)
                del merged[j]
                changed = True
                break
            if changed:
                break
    return merged


def _attach_dots(bars: list[tuple[int, int, int, int]],
                 dots: list[tuple[int, int, int, int]], reference_height: int
                 ) -> list[tuple[int, int, int, int]]:
    """Grow each bar to cover the small dots directly above or below it.

    Handwritten division dots are often only 6-17 % of the digit height (median
    10 % on the team's 359 photographed divisions), below the blob limit in
    _is_operator_shape, so a division sign arrived as its bar alone and was read
    as a minus. Such a dot is kept only when its centre lies within the bar's
    columns and close above or below it; every other small speck is still dropped.
    """
    out = []
    for (x, y, w, h) in bars:
        if w < 2 * h:                      # only bars can be division signs
            out.append((x, y, w, h))
            continue
        x0, y0, x1, y1 = x, y, x + w, y + h
        for (dx, dy, dw, dh) in dots:
            cx, cy = dx + dw / 2, dy + dh / 2
            if x <= cx <= x + w and abs(cy - (y + h / 2)) <= 0.6 * reference_height:
                x0, y0 = min(x0, dx), min(y0, dy)
                x1, y1 = max(x1, dx + dw), max(y1, dy + dh)
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def _is_small_dot(box: tuple[int, int, int, int], reference_height: int) -> bool:
    """A roughly round mark 4-15 % of the digit height: a possible division dot."""
    _, _, w, h = box
    return (h > 0 and 0.4 <= w / h <= 2.5
            and 0.04 * reference_height <= max(w, h) < 0.15 * reference_height)


def _filter_boxes(boxes: list[tuple[int, int, int, int]], image_shape: tuple[int, int],
                  min_area_ratio: float, min_height_ratio: float,
                  keep_operators: bool = False
                  ) -> list[tuple[int, int, int, int]]:
    """Drop boxes that are too small to be digits (limits are fractions of the image size).

    keep_operators also admits the short, small shapes that make up the
    operators of the arithmetic extension, and rejoins the pieces of a division
    sign. It is off by default: it admits noise that the digit pipeline wants
    rejected (see SELECTED_METHOD).

    The operator boxes are merged among themselves and never with a digit box,
    so turning the flag on can only add boxes - a digit that was found before
    is still found, as its own box, afterwards.
    """
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w
    reference_height = max((h for (_, _, _, h) in boxes), default=0)
    digits, operators, dots = [], [], []
    for (x, y, w, h) in boxes:
        digit_sized = (w * h >= min_area_ratio * img_area
                       and h >= min_height_ratio * img_h)
        if digit_sized:
            digits.append((x, y, w, h))
        elif keep_operators and _is_operator_shape((x, y, w, h), reference_height):
            operators.append((x, y, w, h))
        elif keep_operators and _is_small_dot((x, y, w, h), reference_height):
            dots.append((x, y, w, h))
    return digits + _attach_dots(_merge_stacked(operators), dots, reference_height)


def segment_by_contours(image: np.ndarray, min_area_ratio: float = 0.002,
                        min_height_ratio: float = 0.15, margin: int = 4,
                        keep_operators: bool = False,
                        ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Find digits from the outer contour of each ink region. Returns (crops, boxes) ordered left to right."""
    binary = _prepare_binary(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio,
                          keep_operators)
    boxes.sort(key=lambda b: b[0])   # left to right keeps the digit order

    crops = []
    for (x, y, w, h) in boxes:
        crop = binary[y:y + h, x:x + w]
        crops.append(_pad_to_square(crop, margin))
    return crops, boxes


def segment_by_connected_components(image: np.ndarray, min_area_ratio: float = 0.002,
                                    min_height_ratio: float = 0.15, margin: int = 4,
                                    keep_operators: bool = False,
                                    ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Find digits by labelling connected white regions. Returns (crops, boxes) ordered left to right."""
    binary = _prepare_binary(image)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    boxes = []
    for i in range(1, n_labels):        # label 0 is the background
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        boxes.append((x, y, w, h))

    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio,
                          keep_operators)
    boxes.sort(key=lambda b: b[0])

    crops = [_pad_to_square(binary[y:y + h, x:x + w], margin) for (x, y, w, h) in boxes]
    return crops, boxes


def segment_by_projection(image: np.ndarray, min_area_ratio: float = 0.002,
                          min_height_ratio: float = 0.15, margin: int = 4,
                          valley_ratio: float = 0.20,
                          keep_operators: bool = False,
                          ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Split on the vertical ink profile: count ink per column and cut at the valleys.

    Unlike the two methods above this does not care whether pixels are
    connected. It only looks at how much ink each column holds, so a pair of
    digits joined by a thin stroke can still be cut at the waist between them.
    The cost is that it assumes digits do not overlap horizontally.

    valley_ratio sets how empty a column has to be to count as a gap, as a
    fraction of the busiest column. 0.20 was chosen by sweeping it against the
    generated set: below it the method only finds true gaps and scores 20% on
    touching digits, above it single digits start getting cut in half.

        ratio   overall   tight   touching   wide
        0.05      70%      90%       20%     100%
        0.15      80%     100%       40%     100%
        0.20      83%     100%       60%      90%     <- selected
        0.30      70%      70%       60%      80%
        0.40      67%      60%       90%      50%
    """
    binary = _prepare_binary(image)
    column_ink = (binary > 0).sum(axis=0)
    if column_ink.max() == 0:
        return [], []

    cutoff = valley_ratio * column_ink.max()
    is_ink = column_ink > cutoff

    boxes = []
    start = None
    for x, filled in enumerate(is_ink):
        if filled and start is None:
            start = x
        elif not filled and start is not None:
            boxes.append(_box_from_column_band(binary, start, x))
            start = None
    if start is not None:
        boxes.append(_box_from_column_band(binary, start, len(is_ink)))

    boxes = [b for b in boxes if b is not None]
    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio,
                          keep_operators)
    boxes.sort(key=lambda b: b[0])

    crops = [_pad_to_square(binary[y:y + h, x:x + w], margin) for (x, y, w, h) in boxes]
    return crops, boxes


def _box_from_column_band(binary: np.ndarray, x_start: int, x_end: int):
    """Turn a band of ink-bearing columns into a tight (x, y, w, h) box."""
    band = binary[:, x_start:x_end]
    rows = np.where(band.any(axis=1))[0]
    if len(rows) == 0:
        return None
    return (x_start, int(rows[0]), x_end - x_start, int(rows[-1] - rows[0] + 1))


def segment_by_watershed(image: np.ndarray, min_area_ratio: float = 0.002,
                         min_height_ratio: float = 0.15, margin: int = 4,
                         foreground_ratio: float = 0.45,
                         keep_operators: bool = False,
                         ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Split touching digits with a distance transform and watershed.

    Added specifically for the failure case the other methods share: two digits
    that touch are one connected shape, so contours and connected components
    both return them as a single box. The distance transform peaks near the
    centre of each digit, so thresholding it gives one seed per digit even when
    the ink is joined, and watershed grows those seeds back out to the boundary.

    Each crop is masked to its own label, so a neighbouring digit's ink does not
    leak into the crop.
    """
    binary = _prepare_binary(image)
    if not binary.any():
        return [], []

    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)

    # Threshold the distance map per blob, not globally. One fat blob would
    # otherwise set a cutoff that erases every thinner blob in the image.
    n_blobs, blob_labels = cv2.connectedComponents(binary)
    seeds = np.zeros_like(binary)
    for blob in range(1, n_blobs):
        in_blob = blob_labels == blob
        peak = distance[in_blob].max()
        seeds[in_blob & (distance > foreground_ratio * peak)] = 255

    n_seeds, markers = cv2.connectedComponents(seeds)
    if n_seeds <= 1:                       # no seed survived, nothing to grow
        return [], []

    markers = markers + 1                  # watershed reserves 0 for "unknown"
    markers[cv2.subtract(binary, seeds) == 255] = 0
    markers = cv2.watershed(cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR), markers)

    boxes, masks = [], []
    for label in range(2, n_seeds + 1):    # 1 is background, -1 is the boundary
        mask = np.where(markers == label, binary, 0).astype(np.uint8)
        ink = cv2.findNonZero(mask)
        if ink is None:
            continue
        boxes.append(cv2.boundingRect(ink))
        masks.append(mask)

    keep = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio,
                         keep_operators)
    # A kept box is normally one of the boxes above and keeps that label's mask.
    # The exception is a division sign, which the filter merged out of three
    # boxes: its crop has to come from all three masks, or it would show the
    # bar without the dots.
    original = list(zip(boxes, masks))
    pairs = [(box, _union_masks(original, box)) for box in keep]

    pairs.sort(key=lambda pair: pair[0][0])

    crops = [_pad_to_square(m[y:y + h, x:x + w], margin) for (x, y, w, h), m in pairs]
    return crops, [b for b, _ in pairs]


def _union_masks(pairs: list[tuple[tuple[int, int, int, int], np.ndarray]],
                 box: tuple[int, int, int, int]) -> np.ndarray:
    """The mask for one kept box: its own, or the union of the pieces it merged.

    A box that survived the filter unchanged keeps its own mask. Only a merged
    box has no mask of its own, and then the pieces inside it are combined.
    Taking the union in both cases would be wrong: one watershed label's
    bounding box can enclose another, and unioning those two would put a
    neighbour's ink into the crop.
    """
    for original_box, mask in pairs:
        if original_box == box:
            return mask

    x, y, w, h = box
    total = np.zeros_like(pairs[0][1])
    for (bx, by, bw, bh), mask in pairs:
        if bx >= x and by >= y and bx + bw <= x + w and by + bh <= y + h:
            total = cv2.bitwise_or(total, mask)
    return total


METHODS = {
    "contours": segment_by_contours,
    "connected_components": segment_by_connected_components,
    "projection": segment_by_projection,
    "watershed": segment_by_watershed,
}


# Selected technique (Task 2).
#
# Correct-digit-count accuracy, first on 30 generated numbers
# (reports/segmentation_comparison.csv), then on 99 real photographed numbers
# by four writers (reports/custom_segmentation.csv, 25 of them with digits
# that touch). The last column reads the whole number with the team CNN:
#
#                          generated   real photos   real, read exactly
#   contours       <-         70%         75%              62%
#   connected_components      70%         75%              62%
#   watershed                 50%         52%              34%
#   projection                83%         17%               5%
#
# projection was selected on the generated set and it did not survive contact
# with real handwriting: a real pen stroke is thin, so a column through the
# middle of a 0 or 6 holds only a few pixels of ink and the valley threshold
# treats it as a gap between digits. "60" came out as 11 pieces. The tuned
# threshold was fitted to MNIST-thick strokes and did not transfer.
#
# contours and connected_components tie on both sets because they find the
# same thing - connected regions of ink - computed two ways. contours is kept
# as the selection. The six model-based segmenters in segmentation_ml.py
# (sliding window with MLP, CNN, SVM, RF; cut points; confidence split) were
# run on the same 99: confidence_split ties contours at 62%, the rest read
# 3-22%, so none replaces it.
#
# Touching digits remain the open limitation. Separate digits: 96% counted
# right, 81% read exactly (74 numbers). Touching digits: 12% and 4% (25
# numbers). 22 of the 38 numbers read wrongly are touching ones split into
# too few pieces.
#
# keep_operators (the extension) is off by default, and the reason is measured.
# On the first 27 real numbers, correct-count accuracy with the flag on:
#
#                          off    on
#   contours       <-      74%   70%
#   connected_components   74%   70%
#   watershed              52%   37%
#   projection             19%   19%
#
# contours loses one number, to one extra box: the flag exists to admit shapes
# much shorter than a digit, and a speck of dirt on the paper is also much
# shorter than a digit. Watershed loses far more because it already splits
# ink into fragments, and the flag then accepts those fragments as operators.
# Expression input turns the flag on and pays that cost; number input does not.
SELECTED_METHOD = "contours"


def segment_digits(image: np.ndarray, method: str = SELECTED_METHOD, **kwargs
                   ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Split a number image into ordered single-digit images.

    method: one of METHODS - 'contours', 'connected_components', 'projection'
    or 'watershed'. Defaults to the technique selected for the project.

    keep_operators=True also returns the arithmetic operators, for the
    extension. Use segment_expression() for that rather than passing it here.
    """
    if method not in METHODS:
        raise KeyError(f"Unknown method '{method}'. Available: {list(METHODS.keys())}")
    return METHODS[method](image, **kwargs)


def segment_expression(image: np.ndarray, method: str = SELECTED_METHOD, **kwargs
                       ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Split an arithmetic expression into ordered symbol images (digits and operators).

    The entry point for the extension. Same contract as segment_digits - crops
    and boxes, ordered left to right - but the boxes also cover '+ - * / ( )'.
    Classify the crops with ext_best.keras, whose 16 classes are listed in
    ext_best.labels.json; classes 10-15 are the operators.

    'projection' is rejected here. It cuts on vertical gaps in the ink, and a
    division sign's dots sit in the same columns as its bar, so the three
    pieces arrive as one band that cannot be told from a minus sign. The other
    three methods find all six operators (tests/test_segmentation.py).
    """
    if method == "projection":
        raise ValueError(
            "'projection' cannot separate the pieces of a division sign. "
            f"Use one of {[m for m in METHODS if m != 'projection']}."
        )
    return segment_digits(image, method=method, keep_operators=True, **kwargs)


def annotate(image: np.ndarray, boxes: list[tuple[int, int, int, int]],
             labels: list[str] | None = None) -> np.ndarray:
    """Draw the boxes on a copy of the image."""
    canvas = image.copy()
    if len(canvas.shape) == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(canvas, (x, y), (x + w, y + h), (0, 200, 0), 2)
        text = labels[i] if labels and i < len(labels) else str(i + 1)
        cv2.putText(canvas, text, (x, max(y - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    return canvas


if __name__ == "__main__":
    demo = np.zeros((120, 300), dtype=np.uint8)
    for i, ch in enumerate("482"):
        cv2.putText(demo, ch, (20 + i * 90, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 4)
    demo = cv2.bitwise_not(demo)   # dark on light, like a real photo

    for name in METHODS:
        crops, boxes = segment_digits(demo, method=name)
        print(f"{name:22s} -> found {len(crops)} digit(s), "
              f"crop sizes: {[c.shape[0] for c in crops]}")
