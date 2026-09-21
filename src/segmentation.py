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


METHODS = {
    "contours": segment_by_contours,
    "connected_components": segment_by_connected_components,
}


def segment_digits(image: np.ndarray, method: str = "contours", **kwargs
                   ) -> tuple[list[np.ndarray], list[tuple[int, int, int, int]]]:
    """Split a number image into ordered single-digit images. method: 'contours' or 'connected_components'."""
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
