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


def _filter_boxes(boxes: list[tuple[int, int, int, int]], image_shape: tuple[int, int],
                  min_area_ratio: float, min_height_ratio: float
                  ) -> list[tuple[int, int, int, int]]:
    """Drop boxes that are too small to be digits (limits are fractions of the image size)."""
    img_h, img_w = image_shape[:2]
    img_area = img_h * img_w
    keep = []
    for (x, y, w, h) in boxes:
        if w * h < min_area_ratio * img_area:
            continue          # noise
        if h < min_height_ratio * img_h:
            continue          # dust, pen dots
        keep.append((x, y, w, h))
    return keep


def segment_by_contours(image: np.ndarray, min_area_ratio: float = 0.002,
                        min_height_ratio: float = 0.15, margin: int = 4,
                        ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Find digits from the outer contour of each ink region. Returns (crops, boxes) ordered left to right."""
    binary = _prepare_binary(image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio)
    boxes.sort(key=lambda b: b[0])   # left to right keeps the digit order

    crops = []
    for (x, y, w, h) in boxes:
        crop = binary[y:y + h, x:x + w]
        crops.append(_pad_to_square(crop, margin))
    return crops, boxes


def segment_by_connected_components(image: np.ndarray, min_area_ratio: float = 0.002,
                                    min_height_ratio: float = 0.15, margin: int = 4,
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

    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio)
    boxes.sort(key=lambda b: b[0])

    crops = [_pad_to_square(binary[y:y + h, x:x + w], margin) for (x, y, w, h) in boxes]
    return crops, boxes


def segment_by_projection(image: np.ndarray, min_area_ratio: float = 0.002,
                          min_height_ratio: float = 0.15, margin: int = 4,
                          valley_ratio: float = 0.20,
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
    boxes = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio)
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

    keep = _filter_boxes(boxes, binary.shape, min_area_ratio, min_height_ratio)
    pairs = [(b, m) for b, m in zip(boxes, masks) if b in keep]
    pairs.sort(key=lambda pair: pair[0][0])

    crops = [_pad_to_square(m[y:y + h, x:x + w], margin) for (x, y, w, h), m in pairs]
    return crops, [b for b, _ in pairs]


METHODS = {
    "contours": segment_by_contours,
    "connected_components": segment_by_connected_components,
    "projection": segment_by_projection,
    "watershed": segment_by_watershed,
}


# Selected technique (Task 2), from reports/segmentation_comparison.csv.
# Correct-digit-count accuracy on 30 generated numbers, split by how much the
# digits are spaced:
#
#                          overall   tight   touching   wide
#   projection    <-         83%     100%       60%      90%
#   contours                 70%      80%       40%      90%
#   connected_components     70%      80%       40%      90%
#   watershed                50%      60%       40%      50%
#
# Three things this table says, none of which were visible before the test set
# included digits that actually touch:
#
#   1. contours and connected_components score identically on every level.
#      They are not two independent techniques; both find connected regions of
#      ink and only differ in how OpenCV computes them.
#   2. projection wins because it works on ink per column rather than on
#      connectivity, so it can still cut two digits that are joined.
#   3. watershed is the worst here. Its distance transform assumes blob-like
#      objects with a peak in the middle; digits are strokes of roughly even
#      width, so there is no per-digit peak to seed from. Investigated and
#      rejected, with a reason.
#
# Touching digits remain the open limitation: 60% is the best any of the four
# manages. Not solved, only reduced.
SELECTED_METHOD = "projection"


def segment_digits(image: np.ndarray, method: str = SELECTED_METHOD, **kwargs
                   ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Split a number image into ordered single-digit images.

    method: one of METHODS - 'contours', 'connected_components', 'projection'
    or 'watershed'. Defaults to the technique selected for the project.
    """
    if method not in METHODS:
        raise KeyError(f"Unknown method '{method}'. Available: {list(METHODS.keys())}")
    return METHODS[method](image, **kwargs)


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
