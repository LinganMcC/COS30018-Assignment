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
