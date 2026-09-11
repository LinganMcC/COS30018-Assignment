"""
Shared MNIST data loader for all CNN experiments.

Every model in this project loads MNIST through this module so that the
train / validation / test protocol is identical across variants. That is
what makes the three-way comparison in the final report credible - the
same rows of data, split the same way, normalised the same way, evaluated
on the same held-out test set.

Split protocol
--------------
MNIST ships with 60 000 training images and 10 000 test images. We
further split the training set into 54 000 training + 6 000 validation
so that hyperparameters (early stopping, checkpoint selection) are tuned
without ever peeking at the test set. Test-set accuracy is reported once
per model, at the end.
"""

from typing import Tuple
import numpy as np
import tensorflow as tf


def load_mnist(
    validation_size: int = 6_000,
    seed: int = 42,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray],
]:
    """
    Load MNIST and return ((x_train, y_train), (x_val, y_val), (x_test, y_test)).

    Images are reshaped to (N, 28, 28, 1) and normalised to [0.0, 1.0].
    Labels are kept as integers (use sparse_categorical_crossentropy loss).

    Parameters
    ----------
    validation_size : int
        How many samples to hold out from the training set for validation.
        Default 6 000 matches the standard 10% split of the 60k training
        set.
    seed : int
        Random seed for the deterministic train/val shuffle so all three
        model runs see exactly the same split.

    Returns
    -------
    Three (x, y) tuples: train, validation, test.
    """
    (x_train_full, y_train_full), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Add the channel dimension expected by Conv2D layers.
    x_train_full = x_train_full.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # Deterministic shuffle so every model sees the same train/val split.
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(x_train_full))
    x_train_full = x_train_full[idx]
    y_train_full = y_train_full[idx]

    # Split off the validation portion from the tail of the shuffled train set.
    x_val = x_train_full[-validation_size:]
    y_val = y_train_full[-validation_size:]
    x_train = x_train_full[:-validation_size]
    y_train = y_train_full[:-validation_size]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


if __name__ == "__main__":
    # Smoke test - run this file directly to verify the loader works.
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist()
    print(f"Train : {x_train.shape}  labels {y_train.shape}")
    print(f"Val   : {x_val.shape}  labels {y_val.shape}")
    print(f"Test  : {x_test.shape}  labels {y_test.shape}")
    print(f"Pixel range: [{x_train.min():.2f}, {x_train.max():.2f}]")
