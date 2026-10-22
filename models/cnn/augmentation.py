from dataclasses import dataclass, asdict
from typing import Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


@dataclass(frozen=True)
class AugmentConfig:
    """Strength of each random transform.

    rotation_deg: max rotation in degrees (either direction).
    shift:        max translation as a fraction of image size.
    zoom:         max zoom in/out as a fraction (0.1 = +/-10 %).
    """
    rotation_deg: float = 0.0
    shift: float = 0.0
    zoom: float = 0.0

    def is_identity(self) -> bool:
        return self.rotation_deg == 0 and self.shift == 0 and self.zoom == 0

    def to_dict(self) -> dict:
        return asdict(self)


# Named presets used in the tuning grid. "none" is the Sprint 2 behaviour.
PRESETS: Dict[str, AugmentConfig] = {
    "none":   AugmentConfig(0.0, 0.00, 0.00),
    "light":  AugmentConfig(8.0, 0.08, 0.08),
    "medium": AugmentConfig(12.0, 0.10, 0.10),
    "strong": AugmentConfig(18.0, 0.15, 0.15),
}

# Perturbation used to build the shifted test set. Deliberately a bit
# stronger than "medium" so it actually stresses the models.
ROBUST_TEST_AUGMENT = AugmentConfig(rotation_deg=15.0, shift=0.12, zoom=0.12)


def get_preset(name: str) -> AugmentConfig:
    """Look up a preset by name ('none', 'light', 'medium', 'strong')."""
    if name not in PRESETS:
        raise ValueError(f"Unknown augmentation preset '{name}'. "
                         f"Choose from {sorted(PRESETS)}")
    return PRESETS[name]


def build_augmenter(cfg: AugmentConfig,
                    seed: Optional[int] = None) -> Optional[tf.keras.Sequential]:
    """Return a Sequential of random layers, or None when cfg does nothing."""
    if cfg.is_identity():
        return None

    aug_layers = []
    if cfg.rotation_deg > 0:
        # RandomRotation takes a fraction of a full turn, not degrees.
        aug_layers.append(layers.RandomRotation(
            factor=cfg.rotation_deg / 360.0,
            fill_mode="constant", fill_value=0.0, seed=seed))
    if cfg.shift > 0:
        aug_layers.append(layers.RandomTranslation(
            height_factor=cfg.shift, width_factor=cfg.shift,
            fill_mode="constant", fill_value=0.0, seed=seed))
    if cfg.zoom > 0:
        aug_layers.append(layers.RandomZoom(
            height_factor=(-cfg.zoom, cfg.zoom),
            fill_mode="constant", fill_value=0.0, seed=seed))
    return tf.keras.Sequential(aug_layers, name="augmentation")


def make_shifted_test_set(x: np.ndarray, seed: int = 0,
                          cfg: AugmentConfig = ROBUST_TEST_AUGMENT,
                          batch_size: int = 1024) -> np.ndarray:
    """Apply one fixed random perturbation to every test image.

    The same seed gives the same perturbed images, so all runs are compared
    on an identical shifted test set.
    """
    augmenter = build_augmenter(cfg, seed=seed)
    if augmenter is None:
        return x.copy()
    out = []
    for start in range(0, len(x), batch_size):
        batch = augmenter(x[start:start + batch_size], training=True)
        out.append(np.asarray(batch))
    # Interpolation can push values a hair outside [0, 1]; clip back.
    return np.clip(np.concatenate(out, axis=0), 0.0, 1.0).astype("float32")


# Photo crops from preprocess() have hard, binary edges and pen-dependent
# stroke widths; MNIST and the symbol set have soft edges. ext_best learned
# edge softness as a cue (brackets read as "1" until the input was blurred).
# These random per-image transforms make edge style and stroke width useless
# as cues, so the model has to use shape.

def _gaussian_kernel(sigma: float, size: int = 5) -> np.ndarray:
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype("float32")[:, :, None, None]


_BLUR_KERNELS = [tf.constant(_gaussian_kernel(s)) for s in (0.5, 0.8, 1.1)]


def _blur(x: tf.Tensor, kernel: tf.Tensor) -> tf.Tensor:
    return tf.nn.conv2d(x, kernel, strides=1, padding="SAME")


def _per_image_choice(options, probs, n):
    """Pick one of `options` (same-shape tensors) independently for each image."""
    logits = tf.math.log(tf.constant([probs], tf.float32))
    idx = tf.random.categorical(tf.tile(logits, [n, 1]), 1)[:, 0]
    onehot = tf.one_hot(idx, len(options))[:, :, None, None, None]   # (n, k, 1, 1, 1)
    return tf.reduce_sum(tf.stack(options, axis=1) * onehot, axis=1)


@tf.function
def style_augment(x: tf.Tensor) -> tf.Tensor:
    """Random stroke width, edge style and contrast for a (N, 28, 28, 1) batch in [0, 1]."""
    n = tf.shape(x)[0]
    # stroke width: unchanged / thicker / thinner
    dilate = tf.nn.max_pool2d(x, 2, 1, "SAME")
    erode = -tf.nn.max_pool2d(-x, 2, 1, "SAME")
    x = _per_image_choice([x, dilate, erode], [0.6, 0.25, 0.15], n)
    # edge style: as is / hard binary (like photo crops) / sharpened / three blur levels
    t = tf.random.uniform([n, 1, 1, 1], 0.25, 0.6)
    binary = tf.sigmoid((x - t) * 30.0)
    sharp = tf.clip_by_value(x + 1.0 * (x - _blur(x, _BLUR_KERNELS[1])), 0.0, 1.0)
    blurs = [_blur(x, k) for k in _BLUR_KERNELS]
    x = _per_image_choice([x, binary, sharp] + blurs, [0.25, 0.3, 0.15, 0.1, 0.1, 0.1], n)
    # contrast: peak between 0.7 and 1, then renormalise most images back to 1
    peak = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True) + 1e-6
    gain = tf.where(tf.random.uniform([n, 1, 1, 1]) < 0.8, 1.0 / peak,
                    tf.random.uniform([n, 1, 1, 1], 0.7, 1.0) / peak)
    return tf.clip_by_value(x * gain, 0.0, 1.0)


# Synthetic writing styles MNIST hardly contains (continental 7 and 1).

def cross_sevens(x7: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a short horizontal bar across the stem of each 7 (28x28x1, [0, 1])."""
    import cv2
    out = []
    for img in x7:
        im = (img[..., 0] * 255).astype(np.uint8)
        ys, xs = np.nonzero(im > 128)
        if len(ys) == 0:
            out.append(img)
            continue
        y = int(np.percentile(ys, rng.uniform(45, 60)))
        row = xs[np.abs(ys - y) <= 1]
        cx = int(row.mean()) if len(row) else int(xs.mean())
        half = int(rng.integers(4, 7))
        tilt = int(rng.integers(-1, 2))
        cv2.line(im, (cx - half, y + tilt), (cx + half, y - tilt), 255, int(rng.integers(2, 4)))
        out.append((im.astype("float32") / 255.0)[..., None])
    return np.stack(out)


def flag_ones(x1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a long flag from the top of each 1 down to the left (the continental 1)."""
    import cv2
    out = []
    for img in x1:
        im = (img[..., 0] * 255).astype(np.uint8)
        ys, xs = np.nonzero(im > 128)
        if len(ys) == 0:
            out.append(img)
            continue
        top = int(ys.min())
        tx = int(xs[ys <= top + 1].mean())
        length = int(rng.integers(6, 12))
        drop = int(rng.integers(4, 9))
        cv2.line(im, (tx, top), (max(tx - length, 1), top + drop), 255, int(rng.integers(2, 4)))
        out.append((im.astype("float32") / 255.0)[..., None])
    return np.stack(out)
