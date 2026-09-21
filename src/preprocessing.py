"""
Task 1 - Image Preprocessing
COS30018 Option B - Handwritten Number Recognition System
Owner: John (Person A)

The spec requires more than "make the image the right size": it asks us to
research and experiment with different preprocessing techniques, compare them,
and select the appropriate one for the project.

This module therefore exposes each technique as its own function so they can be
switched on and off independently, and a single `preprocess()` entry point that
takes a named configuration. `experiments/compare_preprocessing.py` uses that to
measure which configuration actually performs best.

Pipeline position:
    raw image -> [THIS MODULE] -> segmentation -> recognition
"""

from __future__ import annotations

import cv2
import numpy as np

# Target size expected by the digit classifier. MNIST is 28x28, so matching it
# lets us train on MNIST and predict on our own images without rescaling twice.
TARGET_SIZE = (28, 28)


# ---------------------------------------------------------------------------
# Individual techniques - each one is deliberately small and testable on its own
# ---------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Collapse a colour image to a single intensity channel.

    Colour carries no information about which digit was written, so removing it
    cuts the input size by two thirds with no loss of useful signal.
    Already-grey images are passed through unchanged.
    """
    if image is None:
        raise ValueError("to_grayscale() received None - check the image path is correct.")
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def denoise(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Remove speckle noise with a median blur.

    A median filter is used rather than a Gaussian because it removes isolated
    bright/dark pixels (camera sensor noise, paper texture) while keeping stroke
    edges sharp - Gaussian blur would soften the strokes themselves.
    """
    return cv2.medianBlur(image, kernel_size)


def binarize_otsu(image: np.ndarray) -> np.ndarray:
    """Convert to pure black/white using an automatically chosen threshold.

    Otsu's method picks the threshold that best separates the two intensity
    clusters (ink vs paper), so it adapts to different lighting instead of us
    hard-coding a value. THRESH_BINARY_INV makes the strokes white on a black
    background, matching how MNIST is stored.
    """
    _, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def binarize_adaptive(image: np.ndarray, block_size: int = 11, c: int = 2) -> np.ndarray:
    """Threshold each region using its own local average.

    Included as the third comparison point: unlike Otsu, this copes with photos
    where one side of the page is shadowed, because the threshold varies across
    the image instead of being global.
    """
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block_size, c,
    )


def invert_if_dark_strokes(image: np.ndarray) -> np.ndarray:
    """Ensure strokes are light on a dark background (MNIST convention).

    A photo of black pen on white paper arrives the opposite way round from
    MNIST. If the image is mostly bright, we assume it is dark-on-light and flip
    it, so downstream code never has to care which way the source was.
    """
    if image.mean() > 127:
        return cv2.bitwise_not(image)
    return image


def resize_image(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Resize to the classifier's input size.

    INTER_AREA is the correct interpolation for shrinking - it averages over the
    source pixels rather than sampling one of them, so thin strokes survive
    instead of disappearing.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def center_by_mass(image: np.ndarray) -> np.ndarray:
    """Shift the digit so its centre of mass sits in the middle of the frame.

    This is how the original MNIST images were normalised. Applying the same
    normalisation to our own images removes a systematic difference between
    training data and real input.
    """
    moments = cv2.moments(image)
    if moments["m00"] == 0:  # empty image - nothing to centre
        return image
    cx = int(moments["m10"] / moments["m00"])
    cy = int(moments["m01"] / moments["m00"])
    rows, cols = image.shape
    shift_x = cols // 2 - cx
    shift_y = rows // 2 - cy
    translation = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, translation, (cols, rows))


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale pixel values from 0-255 into 0.0-1.0.

    Neural networks train far more stably on small inputs; leaving values at
    0-255 makes gradients large and learning erratic.
    """
    return image.astype("float32") / 255.0


# ---------------------------------------------------------------------------
# Named configurations - these are what the comparison experiment evaluates
# ---------------------------------------------------------------------------

CONFIGS = {
    "grayscale_only": {
        "denoise": False, "threshold": None, "center": False,
        "description": "Grayscale + resize only (no thresholding)",
    },
    "otsu": {
        "denoise": False, "threshold": "otsu", "center": False,
        "description": "Grayscale + Otsu binarization",
    },
    "otsu_denoised": {
        "denoise": True, "threshold": "otsu", "center": False,
        "description": "Grayscale + median denoise + Otsu binarization",
    },
    "otsu_denoised_centered": {
        "denoise": True, "threshold": "otsu", "center": True,
        "description": "Grayscale + denoise + Otsu + centre of mass alignment",
    },
    "adaptive": {
        "denoise": True, "threshold": "adaptive", "center": False,
        "description": "Grayscale + denoise + adaptive thresholding",
    },
}


# ---------------------------------------------------------------------------
# Selected technique (Task 1)
# ---------------------------------------------------------------------------
#
# The specification asks us to compare techniques and then select one. This is
# that selection, kept as a single named constant so there is exactly one place
# in the codebase that answers "which configuration does the system use?".
#
# Chosen: grayscale_only, on the evidence in reports/preprocessing_comparison.csv
#
#   grayscale_only          0.902   <- selected
#   adaptive                0.901
#   otsu_denoised           0.894
#   otsu_denoised_centered  0.888
#   otsu                    0.886
#
# Two caveats recorded deliberately, because they change what this number means:
#
#   1. With 1000 test images the standard error is about +/-1 percentage point,
#      so grayscale_only and adaptive are tied within error. The honest claim is
#      that both beat plain Otsu, not that grayscale beats adaptive.
#   2. The comparison ran on MNIST, which is clean, evenly lit and already
#      normalised - the conditions least favourable to thresholding and
#      denoising. This selection is therefore provisional and must be re-run on
#      photographed handwriting before the final report. If adaptive wins there,
#      that is the configuration that should ship.
SELECTED_CONFIG = "grayscale_only"


def preprocess(image: np.ndarray, config: str = SELECTED_CONFIG,
               size: tuple[int, int] = TARGET_SIZE,
               flatten: bool = False) -> np.ndarray:
    """Run the full preprocessing pipeline for a named configuration.

    Args:
        image:  input image (colour or grayscale, any size)
        config: key from CONFIGS - which combination of techniques to apply
        size:   output dimensions, defaults to MNIST's 28x28
        flatten: if True return a 1-D vector (needed by SVM/MLP baselines);
                 if False return a 2-D image (needed by the CNN)

    Returns:
        float32 array with values in [0, 1]

    Raises:
        KeyError: if `config` is not a known configuration
    """
    if config not in CONFIGS:
        raise KeyError(
            f"Unknown config '{config}'. Available: {list(CONFIGS.keys())}"
        )
    settings = CONFIGS[config]

    result = to_grayscale(image)

    if settings["denoise"]:
        result = denoise(result)

    if settings["threshold"] == "otsu":
        result = binarize_otsu(result)
    elif settings["threshold"] == "adaptive":
        result = binarize_adaptive(result)
    else:
        # No thresholding: still need MNIST's light-on-dark convention.
        result = invert_if_dark_strokes(result)

    result = resize_image(result, size)

    if settings["center"]:
        result = center_by_mass(result)

    result = normalize(result)
    return result.flatten() if flatten else result


def preprocess_batch(images: list[np.ndarray], config: str = SELECTED_CONFIG,
                     flatten: bool = False) -> np.ndarray:
    """Apply `preprocess` to a list of images and stack the results.

    Convenience wrapper for Liam and Thien, who need whole arrays at a time for
    training and evaluation rather than one image per call.
    """
    return np.array([preprocess(img, config=config, flatten=flatten) for img in images])


if __name__ == "__main__":
    # Smoke test with a synthetic image so this runs even before real data exists.
    demo = np.zeros((100, 100), dtype=np.uint8)
    cv2.putText(demo, "5", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 3)

    print("Preprocessing configurations available:")
    for name, cfg in CONFIGS.items():
        out = preprocess(demo, config=name)
        print(f"  {name:26s} -> shape {out.shape}, "
              f"range [{out.min():.2f}, {out.max():.2f}]  | {cfg['description']}")
