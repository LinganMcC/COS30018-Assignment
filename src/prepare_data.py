"""Export MNIST as single-digit PNGs in data/raw/<digit>/.

Run once after setup:
    python src/prepare_data.py
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
SAMPLES_PER_DIGIT = 50


def load_mnist():
    """Load MNIST from Keras, falling back to a direct download (the uni network can block the Keras URL)."""
    try:
        from tensorflow.keras.datasets import mnist
        (x_train, y_train), _ = mnist.load_data()
        print("Loaded MNIST via tensorflow.keras")
        return x_train, y_train
    except Exception as exc:
        print(f"Keras unavailable ({type(exc).__name__}), trying direct download...")

    import urllib.request
    cache = ROOT / "data" / "mnist.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
        print(f"Downloading MNIST from {url} ...")
        urllib.request.urlretrieve(url, cache)
    with np.load(cache, allow_pickle=True) as data:
        print("Loaded MNIST from local cache")
        return data["x_train"], data["y_train"]


def export_digit_images(x, y, samples_per_digit: int = SAMPLES_PER_DIGIT) -> int:
    """Write `samples_per_digit` PNGs per digit into data/raw/<digit>/."""
    written = 0
    for digit in range(10):
        out_dir = RAW_DIR / str(digit)
        out_dir.mkdir(parents=True, exist_ok=True)
        indices = np.where(y == digit)[0][:samples_per_digit]
        for n, idx in enumerate(indices):
            cv2.imwrite(str(out_dir / f"{digit}_{n:03d}.png"), x[idx])
            written += 1
        print(f"  digit {digit}: {len(indices)} images -> {out_dir.relative_to(ROOT)}")
    return written


def main() -> None:
    print("Preparing single-digit image dataset...")
    x, y = load_mnist()
    print(f"MNIST loaded: {x.shape[0]} training images of shape {x.shape[1:]}")

    total = export_digit_images(x, y)

    print(f"\nDone. {total} single-digit images written to {RAW_DIR.relative_to(ROOT)}/")
    print("Next: python src/generate_number.py")


if __name__ == "__main__":
    main()
