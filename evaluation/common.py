"""Shared paths, data loaders and the six models compared in Tasks 3 and 4.

Every model is wrapped so it answers the same question in the same way:
given a batch of 28x28 images (float 0-1, white ink on black, the format
preprocess() produces), return class probabilities for the digits 0-9.
That is what lets one harness score a scikit-learn SVM and a Keras CNN
with identical code, on identical inputs.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for sub in ("src", "models/cnn"):
    if str(ROOT / sub) not in sys.path:
        sys.path.insert(0, str(ROOT / sub))

CHECKPOINTS = ROOT / "models" / "checkpoints"
RESULTS = ROOT / "evaluation" / "results"
SAMPLES = ROOT / "data" / "custom_samples"
DIGITS = [str(d) for d in range(10)]


@dataclass
class Model:
    """One model in the comparison."""
    key: str                       # short id used in file names
    name: str                      # display name
    family: str                    # classical / simple neural / CNN
    predict_proba: Callable[[np.ndarray], np.ndarray]   # (N, 28, 28[, 1]) -> (N, 10)
    params: Optional[int] = None   # trainable parameters / support vectors / tree nodes


def _keras(name: str) -> Callable[[np.ndarray], np.ndarray]:
    import tensorflow as tf
    model = tf.keras.models.load_model(str(CHECKPOINTS / f"{name}.keras"))

    def predict(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, "float32")
        if x.ndim == 3:
            x = x[..., None]
        return model.predict(x, batch_size=512, verbose=0)

    predict.n_params = int(model.count_params())
    return predict


# (key, display name, family, loader). Order = simplest to most complex.
MODEL_SPECS = [
    ("hog_svm", "HOG + SVM (RBF)", "classical", "baseline"),
    ("hog_rf", "HOG + Random Forest", "classical", "baseline"),
    ("mlp_best", "MLP", "simple neural", "keras"),
    ("lenet_best", "LeNet-5", "CNN", "keras"),
    ("resnet_best", "ResNet (small)", "CNN", "keras"),
    ("cnn_best", "VGG-deep (cnn_best)", "CNN", "keras"),
]
FINAL_MODEL = "cnn_best"


def load_models(keys: Optional[List[str]] = None) -> List[Model]:
    """Load the compared models; a baseline that has not been trained is skipped with a note."""
    from baselines import load_baseline
    out = []
    for key, name, family, kind in MODEL_SPECS:
        if keys and key not in keys:
            continue
        try:
            if kind == "keras":
                fn = _keras(key)
                out.append(Model(key, name, family, fn, fn.n_params))
            else:
                b = load_baseline(key)
                out.append(Model(key, name, family, b.predict_proba, b.size()))
        except FileNotFoundError as exc:
            print(f"  skip {name}: {exc}")
    return out


def load_mnist_splits():
    """The fixed 54k / 6k / 10k split every model in the project was trained with."""
    from mnist_loader import load_mnist
    return load_mnist()


def wilson(k: int, n: int, z: float = 1.96):
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
