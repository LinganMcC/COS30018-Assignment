
from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from mnist_loader import load_mnist

# Class names in output-index order. The trainer saves these next to each
# checkpoint so the GUI / expression parser never has to hard-code indices.
DIGIT_LABELS: List[str] = [str(d) for d in range(10)]
# Planned extension classes (indices 10-15). Kept here so the whole team
# agrees on ONE ordering before any extended model is trained.
SYMBOL_LABELS: List[str] = ["+", "-", "*", "/", "(", ")"]


@dataclass
class DatasetSplits:
    """Train / validation / test arrays plus human-readable class names.

    Images are float32 in [0, 1] with shape (N, 28, 28, 1); labels are int
    class indices (sparse labels).
    """
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    label_names: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.label_names)

    def subsample(self, n_train: int, n_val: int, n_test: int,
                  seed: int = 0) -> "DatasetSplits":
        """Return a smaller random copy - used by --quick smoke runs."""
        rng = np.random.default_rng(seed)

        def pick(x, y, n):
            idx = rng.permutation(len(x))[:min(n, len(x))]
            return x[idx], y[idx]

        xt, yt = pick(self.x_train, self.y_train, n_train)
        xv, yv = pick(self.x_val, self.y_val, n_val)
        xs, ys = pick(self.x_test, self.y_test, n_test)
        return DatasetSplits(xt, yt, xv, yv, xs, ys, list(self.label_names))


_LOADERS: Dict[str, Callable[[], DatasetSplits]] = {}


def register_dataset(name: str):
    """Decorator: make a loader available as load_dataset(name)."""
    def wrapper(fn: Callable[[], DatasetSplits]):
        _LOADERS[name] = fn
        return fn
    return wrapper


def available_datasets() -> List[str]:
    return sorted(_LOADERS)


def load_dataset(name: str) -> DatasetSplits:
    """Load a registered dataset by name."""
    if name not in _LOADERS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from {available_datasets()}")
    return _LOADERS[name]()



# Registered datasets


@register_dataset("mnist")
def _load_mnist_splits() -> DatasetSplits:
    """Standard MNIST via the shared loader (same 54k/6k/10k split as Sprint 2)."""
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = load_mnist()
    return DatasetSplits(x_tr, y_tr, x_va, y_va, x_te, y_te, list(DIGIT_LABELS))


@register_dataset("mnist+symbols")
def _load_mnist_symbols() -> DatasetSplits:
    """Placeholder for the Sprint 4 extension dataset (digits + operators)."""
    raise NotImplementedError(
        "The 'mnist+symbols' dataset is not built yet (Sprint 4). It must return "
        "DatasetSplits with label_names = DIGIT_LABELS + SYMBOL_LABELS, and the "
        "symbol images must go through the same preprocessing as the digits.")
