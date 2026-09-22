"""A digit classifier with one interface, so the ML segmentation methods can
swap the model underneath without changing any of their own code.

Four backends, chosen to be four genuinely different model families rather
than four settings of one:

    mlp             fully-connected network   (scikit-learn, trained here)
    cnn             Thien's LeNet             (needs TensorFlow)
    svm             RBF kernel machine        (scikit-learn, trained here)
    random_forest   ensemble of trees         (scikit-learn, trained here)

All four answer the same question: given a 28x28 patch, how confident are you that
it is each of the ten digits, and how confident are you that it is not a digit
at all? The eleventh class is what makes sliding-window segmentation possible -
without it the model would label an empty patch of paper as a confident '1'.

Run directly to train and cache the three scikit-learn models:
    python src/digit_classifier.py
"""

from __future__ import annotations

import pickle
import random
from pathlib import Path

import cv2
import numpy as np

from segmentation import _pad_to_square

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
CHECKPOINTS = ROOT / "models" / "checkpoints"
CNN_PATH = CHECKPOINTS / "cnn_lenet.keras"

PATCH = (28, 28)
NOT_A_DIGIT = 10          # the eleventh class


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def to_patch(crop: np.ndarray, margin: int = 2) -> np.ndarray:
    """Turn any crop into the 28x28 patch shape the models are trained on.

    Training and inference must run a crop through exactly the same steps. The
    sliding window pads its crops to a square before resizing, so the training
    tiles have to be padded the same way. Skipping this costs the MLP a few
    points and breaks the SVM and the random forest outright, because they are
    far less tolerant of input that sits slightly off the distribution they
    were fitted on.
    """
    return cv2.resize(_pad_to_square(crop, margin), PATCH, interpolation=cv2.INTER_AREA)


def _load_digit_tiles() -> tuple[list[np.ndarray], list[int]]:
    """Read the exported single-digit PNGs as white-on-black 28x28 tiles."""
    tiles, labels = [], []
    for digit in range(10):
        for path in sorted((RAW_DIR / str(digit)).glob("*.png")):
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            tiles.append(to_patch(img))
            labels.append(digit)
    if not tiles:
        raise FileNotFoundError(
            f"No digit images under {RAW_DIR}. Run 'python src/prepare_data.py' first."
        )
    return tiles, labels


def _make_negatives(tiles: list[np.ndarray], count: int,
                    rng: random.Random) -> list[np.ndarray]:
    """Build patches that are deliberately NOT a single centred digit.

    Three kinds, because a sliding window meets all three:
      - blank paper
      - the join between two digits, so half of one and half of another
      - a digit sliced down the middle, which is what a badly placed window sees
    """
    negatives = []
    while len(negatives) < count:
        kind = rng.randrange(3)
        if kind == 0:
            negatives.append(np.zeros(PATCH, dtype=np.uint8))
        elif kind == 1:
            left, right = rng.choice(tiles), rng.choice(tiles)
            patch = np.zeros(PATCH, dtype=np.uint8)
            cut = rng.randint(8, 20)
            patch[:, :cut] = left[:, PATCH[1] - cut:]
            patch[:, cut:] = right[:, :PATCH[1] - cut]
            negatives.append(patch)
        else:
            tile = rng.choice(tiles)
            shift = rng.choice([-11, -9, 9, 11])
            patch = np.roll(tile, shift, axis=1)
            if shift > 0:
                patch[:, :shift] = 0
            else:
                patch[:, shift:] = 0
            negatives.append(patch)
    return negatives


def build_training_set(seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Ten digit classes plus a negative class, flattened and scaled to 0-1."""
    rng = random.Random(seed)
    tiles, labels = _load_digit_tiles()
    negatives = _make_negatives(tiles, count=len(tiles) // 2, rng=rng)

    x = np.array([t.flatten() for t in tiles + negatives], dtype="float32") / 255.0
    y = np.array(labels + [NOT_A_DIGIT] * len(negatives))
    return x, y


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class SklearnBackend:
    """Shared plumbing for the scikit-learn models.

    They differ only in which estimator they build, so training, caching and
    the patch-to-probability path live here once.
    """

    name = "sklearn"
    filename = "sklearn_digit.pkl"

    def __init__(self, model):
        self.model = model

    @staticmethod
    def _build(seed: int):
        raise NotImplementedError

    @classmethod
    def cache_path(cls) -> Path:
        return CHECKPOINTS / cls.filename

    @classmethod
    def train(cls, seed: int = 42) -> "SklearnBackend":
        x, y = build_training_set(seed)
        model = cls._build(seed)
        model.fit(x, y)
        return cls(model)

    @classmethod
    def load(cls, retrain: bool = False) -> "SklearnBackend":
        path = cls.cache_path()
        if path.exists() and not retrain:
            with open(path, "rb") as fh:
                return cls(pickle.load(fh))
        backend = cls.train()
        backend.save()
        return backend

    def save(self) -> None:
        path = self.cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self.model, fh)

    def predict_proba(self, patches: np.ndarray) -> np.ndarray:
        flat = patches.reshape(len(patches), -1).astype("float32")
        if flat.max() > 1.0:
            flat = flat / 255.0
        return self.model.predict_proba(flat)


class MLPBackend(SklearnBackend):
    """A fully-connected neural network over flattened 28x28 patches."""

    name = "mlp"
    filename = "mlp_digit.pkl"

    @staticmethod
    def _build(seed: int):
        from sklearn.neural_network import MLPClassifier
        return MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                             random_state=seed, early_stopping=True)


class SVMBackend(SklearnBackend):
    """Support vector machine with an RBF kernel.

    A kernel method rather than a network: it draws the decision boundary from
    a handful of support vectors instead of learning a weighted representation.
    The strongest classical model on MNIST, so it is the one that tests whether
    segmentation quality is limited by the model or by the search around it.
    """

    name = "svm"
    filename = "svm_digit.pkl"

    @staticmethod
    def _build(seed: int):
        from sklearn.svm import SVC
        return SVC(kernel="rbf", C=10.0, gamma="scale",
                   probability=True, random_state=seed)


class RandomForestBackend(SklearnBackend):
    """An ensemble of decision trees.

    Included because it is the one model here that never multiplies pixels by
    weights. It splits on individual pixel values, so it fails differently from
    the other three, which is the point of having it in the comparison.
    """

    name = "random_forest"
    filename = "rf_digit.pkl"

    @staticmethod
    def _build(seed: int):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)


class CNNBackend:
    """Thien's LeNet. Has no negative class, so 'not a digit' is inferred from
    how flat the softmax is: a patch the model cannot commit to is not a digit.
    """

    name = "cnn"

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: Path = CNN_PATH) -> "CNNBackend":
        import tensorflow as tf

        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Ask Thien to merge feature/thien-sprint2-cnn, "
                "or train it with 'python models/cnn/cnn_lenet.py'."
            )
        return cls(tf.keras.models.load_model(str(path)))

    def predict_proba(self, patches: np.ndarray) -> np.ndarray:
        x = patches.astype("float32")
        if x.max() > 1.0:
            x = x / 255.0
        x = x.reshape(-1, 28, 28, 1)
        probs = self.model.predict(x, verbose=0)

        # Turn ten classes into eleven. The confidence the model has in its best
        # guess becomes the digit probability; whatever is left over becomes the
        # not-a-digit probability.
        best = probs.max(axis=1, keepdims=True)
        eleventh = np.clip(1.0 - best, 0.0, 1.0)
        return np.hstack([probs * best, eleventh])


# Four genuinely different model families, so the segmentation comparison
# varies the model and nothing else:
#
#   mlp             fully-connected neural network      (weights over pixels)
#   cnn             convolutional network, Thien's      (learned local filters)
#   svm             RBF kernel machine                  (support vectors)
#   random_forest   ensemble of decision trees          (pixel threshold splits)
#
# The CNN is deliberately the same architecture Thien uses for Task 3. Sharing
# it is the point: if the strongest recogniser also segments best, that is
# worth knowing, and if it does not, that says the bottleneck is the search
# strategy rather than the model.
BACKENDS = {
    "mlp": MLPBackend,
    "cnn": CNNBackend,
    "svm": SVMBackend,
    "random_forest": RandomForestBackend,
}

_LOADED: dict[str, object] = {}


def get_classifier(name: str = "mlp"):
    """Return a backend by name, loading it once per process."""
    if name in _LOADED:
        return _LOADED[name]
    if name not in BACKENDS:
        raise KeyError(f"Unknown classifier '{name}'. Available: {list(BACKENDS)}")
    backend = BACKENDS[name].load()
    _LOADED[name] = backend
    return backend


if __name__ == "__main__":
    x, y = build_training_set()
    print("Training the digit classifiers (10 digits + not-a-digit)")
    print(f"  training patches: {len(x)}  ({int((y == NOT_A_DIGIT).sum())} negatives)\n")

    for name, cls in BACKENDS.items():
        if name == "cnn":
            print(f"  {name:14s} skipped - trained by Thien, loaded from "
                  f"{CNN_PATH.name}")
            continue
        backend = cls.train()
        backend.save()
        print(f"  {name:14s} training accuracy {backend.model.score(x, y):.3f}  "
              f"-> {backend.cache_path().relative_to(ROOT)}")
