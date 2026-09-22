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


def binarize_adaptive(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """Adaptive threshold for uneven lighting."""
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c,
    )


def invert_if_dark_strokes(image: np.ndarray) -> np.ndarray:
    """Invert if the image is mostly bright."""
    if image.mean() > 127:
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
    """
    width, height = size
    ink = cv2.findNonZero(image)
    if ink is None:                       # nothing drawn, nothing to fit
        return np.zeros((height, width), dtype=image.dtype)

    x, y, w, h = cv2.boundingRect(ink)
    digit = image[y:y + h, x:x + w]

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
}

# Selected configuration, based on reports/preprocessing_comparison.csv
# (accuracy of the team's LeNet, retrained per configuration):
#   grayscale_only          0.9660   <- selected
#   otsu                    0.9650
#   adaptive                0.9603
#   otsu_denoised           0.9557
#   otsu_denoised_centered  0.9480
#
# Caveats:
#  - 3000 test images gives a standard error of about 0.3 percentage points,
#    so grayscale_only and otsu are tied. What the results show is that the
#    two simplest configurations beat the ones that denoise or re-centre.
#  - MNIST is clean and evenly lit, which is the worst case for thresholding
#    and denoising. Re-run on photographed handwriting before the final
#    report; if adaptive wins there, switch to it.
#  - The two mnist_box configurations were added after this run and have not
#    been scored yet.
SELECTED_CONFIG = "grayscale_only"


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
