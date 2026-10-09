from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

from datasets import DIGIT_LABELS, SYMBOL_LABELS, DatasetSplits

REPO_ROOT = Path(__file__).resolve().parents[2]
SYMBOLS_DIR = REPO_ROOT / "data" / "symbols"

MAX_PER_CLASS = 6000          # cap per symbol class, close to MNIST's ~6000 per digit
MIN_TRAIN_PER_CLASS = 3000    # small classes are repeated up to this in TRAIN only
SPLIT = (0.8, 0.1, 0.1)       # train / val / test, per class
BULK_SOURCE = "kaggle"        # capped first; every other source is kept in full
CACHE_VERSION = "v2"          # bump when the cache contents change
WIDTH_TOLERANCE = 0.75        # thicken if symbol strokes < 75 % of MNIST width
MAX_DILATIONS = 2



# Stroke width


def normalise_brightness(images: np.ndarray) -> np.ndarray:
    """Rescale every image so its brightest pixel is 1.0 (empty images unchanged)."""
    peak = images.reshape(len(images), -1).max(axis=1)
    peak = np.where(peak > 0, peak, 1.0).reshape(-1, 1, 1, 1)
    return (images / peak).astype("float32")


def stroke_width(img: np.ndarray) -> float:
    """Average stroke width in pixels = ink area / skeleton length.

    Ink is anything above half of the image's own peak, so faint strokes are
    still measured.
    """
    from skimage.morphology import skeletonize
    img = img.squeeze()
    if img.max() <= 0:
        return 0.0
    ink = img > 0.5 * img.max()
    skeleton = skeletonize(ink).sum()
    return float(ink.sum() / skeleton) if skeleton else 0.0


def median_stroke_width(images: np.ndarray, sample: int = 500, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(images), size=min(sample, len(images)), replace=False)
    widths = [stroke_width(images[i]) for i in idx]
    widths = [w for w in widths if w > 0]
    return float(np.median(widths)) if widths else 0.0


def thicken(images: np.ndarray, times: int) -> np.ndarray:
    """Dilate every image `times` times with a 2x2 kernel (adds ~1 px each)."""
    import cv2
    if times <= 0:
        return images
    kernel = np.ones((2, 2), np.uint8)
    out = [cv2.dilate(img.squeeze(), kernel, iterations=times) for img in images]
    return np.stack(out)[..., np.newaxis].astype("float32")


def match_stroke_width(symbols: np.ndarray, mnist: np.ndarray) -> tuple[np.ndarray, dict]:
    """Thicken the symbols if they are clearly thinner than MNIST digits."""
    target = median_stroke_width(mnist)
    before = median_stroke_width(symbols)
    times = 0
    current = before
    while current < WIDTH_TOLERANCE * target and times < MAX_DILATIONS:
        times += 1
        current = median_stroke_width(thicken(symbols, times))
    info = {"mnist_width": target, "symbol_width_before": before,
            "symbol_width_after": current, "dilations": times}
    print(f"Stroke width (px): MNIST {target:.2f} | symbols {before:.2f}"
          + (f" -> {current:.2f} after {times} dilation(s)" if times else " (no change needed)"))
    return thicken(symbols, times), info


# Reading and preprocessing the symbol images

def read_manifest(symbols_dir: Path) -> list[dict]:
    manifest = symbols_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"{manifest} not found. Run `python src/download_symbols.py` from the "
            "repo root first.")
    with manifest.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["label"] in SYMBOL_LABELS]
    if not rows:
        raise ValueError(f"{manifest} lists no images with labels {SYMBOL_LABELS}.")
    return rows


def _preprocess_fn(config: str):
    """John's preprocess(), imported from src/."""
    src = str(REPO_ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from preprocessing import preprocess
    return lambda img: preprocess(img, config=config, add_channel=True)


def load_symbol_images(symbols_dir: Path = SYMBOLS_DIR,
                       config: str | None = None,
                       use_cache: bool = True,
                       return_sources: bool = False):
    """Return (images float32 (N, 28, 28, 1), label strings (N,)).

    With return_sources=True also returns each image's source ("hasy",
    "kaggle", ...) as a third array.
    """
    import cv2
    if config is None:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from preprocessing import SELECTED_CONFIG
        config = SELECTED_CONFIG

    manifest = symbols_dir / "manifest.csv"
    rows = read_manifest(symbols_dir)
    stat = manifest.stat()
    key = f"{CACHE_VERSION}-{stat.st_size}-{int(stat.st_mtime)}-{config}"
    cache = symbols_dir / f"_cache_{config}.npz"
    if use_cache and cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            if str(data["key"]) == key:
                print(f"Loaded {len(data['labels'])} preprocessed symbols from cache")
                x, y, src = data["x"].astype("float32") / 255.0, data["labels"], data["sources"]
                return (x, y, src) if return_sources else (x, y)

    preprocess = _preprocess_fn(config)
    images, labels, sources, unreadable = [], [], [], 0
    print(f"Preprocessing {len(rows)} symbol images with config '{config}' ...")
    for row in rows:
        img = cv2.imread(str(symbols_dir / row["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            unreadable += 1
            continue
        images.append(preprocess(img))
        labels.append(row["label"])
        sources.append(row.get("source", ""))
    if unreadable:
        print(f"Skipped {unreadable} unreadable image files.")
    x = np.stack(images).astype("float32")
    y = np.array(labels)
    src = np.array(sources)
    if use_cache:
        np.savez_compressed(cache, x=np.round(x * 255).astype(np.uint8),
                            labels=y, sources=src, key=np.array(key))
    return (x, y, src) if return_sources else (x, y)

# Splitting, balancing, combining

def cap_class(idx: np.ndarray, sources: np.ndarray | None, limit: int,
              rng: np.random.Generator) -> np.ndarray:
    if sources is None:
        return rng.permutation(idx)[:limit]
    is_bulk = sources[idx] == BULK_SOURCE
    keep = rng.permutation(idx[~is_bulk])[:limit]
    fill = rng.permutation(idx[is_bulk])[:max(0, limit - len(keep))]
    return rng.permutation(np.concatenate([keep, fill]))


def split_symbols(x: np.ndarray, labels: np.ndarray, seed: int = 42,
                  max_per_class: int = MAX_PER_CLASS,
                  min_train_per_class: int = MIN_TRAIN_PER_CLASS,
                  sources: np.ndarray | None = None) -> dict:
    """Per-class 80/10/10 split, then cap / repeat the TRAIN part only.

    Returns {"train": (x, y), "val": (x, y), "test": (x, y)} with integer
    labels 10-15 and a per-class count table under "counts".
    """
    rng = np.random.default_rng(seed)
    parts = {"train": ([], []), "val": ([], []), "test": ([], [])}
    counts = {}
    for offset, label in enumerate(SYMBOL_LABELS):
        idx = np.flatnonzero(labels == label)
        if len(idx) == 0:
            raise ValueError(f"No images for symbol '{label}' - check data/symbols/.")
        idx = cap_class(idx, sources, max_per_class, rng)
        n_val = max(1, int(round(len(idx) * SPLIT[1])))
        n_test = max(1, int(round(len(idx) * SPLIT[2])))
        test, val, train = idx[:n_test], idx[n_test:n_test + n_val], idx[n_test + n_val:]
        n_unique = len(train)
        if len(train) < min_train_per_class:   # repeat small classes (train only)
            extra = rng.choice(train, size=min_train_per_class - len(train), replace=True)
            train = np.concatenate([train, extra])
        class_id = len(DIGIT_LABELS) + offset
        for name, sel in (("train", train), ("val", val), ("test", test)):
            parts[name][0].append(x[sel])
            parts[name][1].append(np.full(len(sel), class_id))
        kept_other = int((sources[idx] != BULK_SOURCE).sum()) if sources is not None else 0
        counts[label] = {"available": int((labels == label).sum()), "train_unique": n_unique,
                         "kept_non_kaggle": kept_other,
                         "train_after_balance": len(train), "val": len(val), "test": len(test)}
    out = {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in parts.items()}
    out["counts"] = counts
    return out


def combine(mnist: DatasetSplits, symbols: dict, seed: int = 42) -> DatasetSplits:
    """Concatenate MNIST and symbol splits and shuffle each split."""
    rng = np.random.default_rng(seed)

    def merge(xm, ym, xs, ys):
        x = np.concatenate([xm, xs]).astype("float32")
        y = np.concatenate([ym.astype(np.int64), ys.astype(np.int64)])
        order = rng.permutation(len(x))
        return x[order], y[order]

    xt, yt = merge(mnist.x_train, mnist.y_train, *symbols["train"])
    xv, yv = merge(mnist.x_val, mnist.y_val, *symbols["val"])
    xs, ys = merge(mnist.x_test, mnist.y_test, *symbols["test"])
    return DatasetSplits(xt, yt, xv, yv, xs, ys, list(DIGIT_LABELS + SYMBOL_LABELS))


def print_counts(counts: dict) -> None:
    print(f"{'symbol':>6} {'available':>9} {'non-kaggle':>10} {'train':>7} "
          f"{'train(bal)':>10} {'val':>6} {'test':>6}")
    for label, c in counts.items():
        print(f"{label:>6} {c['available']:>9} {c['kept_non_kaggle']:>10} {c['train_unique']:>7} "
              f"{c['train_after_balance']:>10} {c['val']:>6} {c['test']:>6}")


def load_mnist_symbols(mnist: DatasetSplits, symbols_dir: Path = SYMBOLS_DIR,
                       seed: int = 42) -> DatasetSplits:
    """The full pipeline used by datasets.load_dataset('mnist+symbols')."""
    x, labels, sources = load_symbol_images(symbols_dir, return_sources=True)
    x = normalise_brightness(x)
    x, _ = match_stroke_width(x, mnist.x_train)
    parts = split_symbols(x, labels, seed=seed, sources=sources)
    print_counts(parts["counts"])
    return combine(mnist, parts, seed=seed)
