"""Task 1 - image preprocessing."""

from __future__ import annotations

import cv2
import numpy as np

TARGET_SIZE = (28, 28)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a colour image to grayscale."""
    if image is None:
        raise ValueError("to_grayscale() received None - check the image path is correct.")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, kernel_size)


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """Otsu threshold, strokes white on black."""
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive(image: np.ndarray, block_size: int | None = None,
                      c: int = 2) -> np.ndarray:
    """Adaptive threshold for uneven lighting.

    The neighbourhood has to be wider than a pen stroke, otherwise the inside
    of a thick stroke is compared only against itself and comes out as paper.
    A fixed 11 pixels suits a small scan but recovered only about a quarter of
    the stroke on an 800x600 test photo, so by default the block scales with
    the image (an eighth of the shorter side, never below 11, always odd).

    c=2 suits clean images. On a noisy test photo it marked about 30% of the
    paper as ink, and c=10 brought that to almost zero. Left at 2 until the
    real handwritten set says otherwise.
    """
    if block_size is None:
        block_size = max(11, min(image.shape[:2]) // 8)
        block_size += 1 - block_size % 2          # must be odd
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c,
    )


def invert_if_dark_strokes(image: np.ndarray) -> np.ndarray:
    """Invert if the background is bright, so strokes end up light on dark.

    The background is judged from the image border, not the whole image. A
    tight crop around a thick digit can be more than half ink, so its overall
    mean is dark even though the paper around it is white - the old mean test
    left those crops un-inverted and the model saw a black digit on white. The
    border of an image is almost always paper, however much ink is inside.
    """
    border = np.concatenate([image[0], image[-1], image[:, 0], image[:, -1]])
    if np.median(border) > 127:
        return cv2.bitwise_not(image)
    return image


def resize_image(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def center_by_mass(image: np.ndarray) -> np.ndarray:
    """Shift the digit so its centre of mass is in the middle."""
    moments = cv2.moments(image)
    if moments["m00"] == 0:
        return image
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    rows, cols = image.shape
    shift_x = cols // 2 - cx
    shift_y = rows // 2 - cy
    translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, translation, (cols, rows))


def fit_to_mnist_box(image: np.ndarray, box: int = 20,
                     size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Scale the digit into a 20x20 box inside a 28x28 frame, centred by mass.

    This is how MNIST itself was built: the digit was size-normalised to fit a
    20x20 box with its aspect ratio kept, then placed in a 28x28 frame with its
    centre of mass in the middle. A plain resize to 28x28 makes the digit fill
    the whole frame, which is a shape the model never saw in training.

    Expects white strokes on a black background.

    The ink is located with an Otsu mask rather than "any non-zero pixel". On a
    clean scan the two are the same, but on a photo the inverted paper is never
    exactly 0, so every pixel counts as non-zero and the box silently becomes
    the whole frame. The mask only decides where to crop; the pixels copied
    into the box keep their original grey values.
    """
    width, height = size
    if not image.any():                   # nothing drawn, nothing to fit
        return np.zeros((height, width), dtype=image.dtype)

    _, mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink = cv2.findNonZero(mask)
    if ink is None:                       # flat image, no ink to separate
        return np.zeros((height, width), dtype=image.dtype)

    x, y, w, h = cv2.boundingRect(ink)
    pad = 2                               # keep the faint anti-aliased edge
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1 = min(image.shape[1], x + w + pad)
    y1 = min(image.shape[0], y + h + pad)
    digit = image[y0:y1, x0:x1]
    h, w = digit.shape

    scale = box / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    digit = cv2.resize(digit, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width), dtype=digit.dtype)
    top = (height - new_h) // 2
    left = (width - new_w) // 2
    canvas[top:top + new_h, left:left + new_w] = digit

    return center_by_mass(canvas)


def normalize(image: np.ndarray) -> np.ndarray:
    return image.astype("float32") / 255.0


CONFIGS = {
    "grayscale_only": {
        "denoise": False, "threshold": None, "center": False, "mnist_box": False,
        "description": "Grayscale + resize only (no thresholding)",
    },
    "otsu": {
        "denoise": False, "threshold": "otsu", "center": False, "mnist_box": False,
        "description": "Grayscale + Otsu binarization",
    },
    "otsu_denoised": {
        "denoise": True, "threshold": "otsu", "center": False, "mnist_box": False,
        "description": "Grayscale + median denoise + Otsu binarization",
    },
    "otsu_denoised_centered": {
        "denoise": True, "threshold": "otsu", "center": True, "mnist_box": False,
        "description": "Grayscale + denoise + Otsu + centre of mass alignment",
    },
    "adaptive": {
        "denoise": True, "threshold": "adaptive", "center": False, "mnist_box": False,
        "description": "Grayscale + denoise + adaptive thresholding",
    },
    # The two below use MNIST's own 20x20-in-28x28 convention instead of a
    # plain resize, so the comparison can measure whether matching the
    # dataset's geometry is worth anything.
    "grayscale_mnist_box": {
        "denoise": False, "threshold": None, "center": False, "mnist_box": True,
        "description": "Grayscale + 20x20 MNIST box (no thresholding)",
    },
    "adaptive_mnist_box": {
        "denoise": True, "threshold": "adaptive", "center": False, "mnist_box": True,
        "description": "Grayscale + denoise + adaptive + 20x20 MNIST box",
    },
    # Added after the first run on real photos: Otsu was the clear winner on
    # single digits (83% against 33% for grayscale_only), and the 20x20 box is
    # what keeps a tight crop from segmentation from being stretched. This is
    # the combination of the two.
    "otsu_mnist_box": {
        "denoise": False, "threshold": "otsu", "center": False, "mnist_box": True,
        "description": "Grayscale + Otsu binarization + 20x20 MNIST box",
    },
}

# Selected configuration (Task 1). Chosen on three measurements, because the
# first one could not tell the options apart:
#
#                        MNIST, 5 seeds    70 real digits   27 real numbers,
#                        (team LeNet)      (single crops)   read end to end*
#   otsu_mnist_box  <-   not yet run          82.9%              63%
#   otsu                 0.967 +/- 0.006      82.9%               7%
#   otsu_denoised        0.959 +/- 0.005      82.9%               7%
#   adaptive_mnist_box   0.965 +/- 0.003      20.0%              56%
#   adaptive             0.967 +/- 0.003      51.4%               4%
#   grayscale_only       0.967 +/- 0.005      32.9%               0%
#   grayscale_mnist_box  0.962 +/- 0.003      28.6%              19%
#
#   * segmented with SELECTED_METHOD (contours), each crop classified by the
#     team CNN, number counted right only if every digit is right.
#
# 1. MNIST ties the top three within one standard deviation. It cannot choose.
# 2. On real photos thresholding is essential: Otsu forces the paper to exactly
#    0, as in MNIST, while grayscale_only leaves a grey haze the CNN never saw.
# 3. End to end, the 20x20 box is essential: segmentation hands over tight
#    crops, and a plain resize stretches them to fill 28x28. Otsu alone falls
#    from 83% on single digits to 7% on numbers; with the box it reads 63%.
#
# otsu_mnist_box is the only configuration at the top of both real tests.
# With 27 numbers the end-to-end figure is +/- about 9 points, so the margin
# over adaptive_mnist_box (56%) alone is not decisive; its 83% against 20% on
# single digits is. The remaining end-to-end errors are mostly segmentation:
# 7 of the 10 misread numbers were split into the wrong number of pieces, 6 of
# them because the digits touch.
SELECTED_CONFIG = "otsu_mnist_box"


def preprocess(image: np.ndarray, config: str = SELECTED_CONFIG,
               size: tuple[int, int] = TARGET_SIZE,
               flatten: bool = False,
               add_channel: bool = False) -> np.ndarray:
    """Run the pipeline for one named config; returns float32 in [0, 1].

    flatten gives a 784-vector for k-NN and the other classical baselines.
    add_channel gives (28, 28, 1), which is the shape the team's Conv2D models
    expect. The two are mutually exclusive.
    """
    if config not in CONFIGS:
        raise KeyError(
            f"Unknown config '{config}'. Available: {list(CONFIGS.keys())}"
        )
    if flatten and add_channel:
        raise ValueError("flatten and add_channel cannot both be True.")
    settings = CONFIGS[config]

    result = to_grayscale(image)

    if settings["denoise"]:
        result = denoise(result)

    if settings["threshold"] == "otsu":
        result = binarize_otsu(result)
    elif settings["threshold"] == "adaptive":
        result = binarize_adaptive(result)
    else:
        # no thresholding, but still need light strokes on dark
        result = invert_if_dark_strokes(result)

    if settings["mnist_box"]:
        # fit_to_mnist_box does its own crop, resize and centring
        result = fit_to_mnist_box(result, size=size)
    else:
        result = resize_image(result, size)
        if settings["center"]:
            result = center_by_mass(result)

    result = normalize(result)
    if flatten:
        return result.flatten()
    if add_channel:
        return result[:, :, np.newaxis]
    return result


def preprocess_batch(images: list[np.ndarray], config: str = SELECTED_CONFIG,
                     flatten: bool = False,
                     add_channel: bool = False) -> np.ndarray:
    return np.array([preprocess(img, config=config, flatten=flatten,
                                add_channel=add_channel) for img in images])


if __name__ == "__main__":
    # quick check on a synthetic image
    demo = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(demo, "5", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

    print("Preprocessing configurations available:")
    for name, cfg in CONFIGS.items():
        out = preprocess(demo, config=name)
        print(f"  {name:26s} -> shape {out.shape}, "
              f"range [{out.min():.2f}, {out.max():.2f}]  | {cfg['description']}")
