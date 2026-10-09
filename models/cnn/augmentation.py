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
