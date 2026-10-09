import csv

import cv2
import numpy as np
import pytest

import symbols_data as sd
from datasets import DIGIT_LABELS, SYMBOL_LABELS, DatasetSplits

FOLDERS = {"+": "plus", "-": "minus", "*": "times", "/": "div", "(": "lparen", ")": "rparen"}


def _draw(label: str, rng) -> np.ndarray:
    """A 45x45 white image with a thin (1 px) black symbol, slightly jittered."""
    img = np.full((45, 45), 255, np.uint8)
    j = int(rng.integers(-3, 4))
    c = 22 + j
    if label in "+-*/":
        cv2.line(img, (8, c), (36, c), 0, 1)                      # horizontal bar
    if label in "+*":
        cv2.line(img, (c, 8), (c, 36), 0, 1)                      # vertical bar
    if label == "*":
        cv2.line(img, (10, 10), (34, 34), 0, 1)
    if label == "/":
        cv2.circle(img, (22, 12), 2, 0, -1)
        cv2.circle(img, (22, 32), 2, 0, -1)
    if label in "()":
        start, end = (90, 270) if label == "(" else (-90, 90)
        cv2.ellipse(img, (c, 22), (8, 16), 0, start, end, 0, 1)
    return img


@pytest.fixture
def symbols_dir(tmp_path):
    """data/symbols/ with 40 images per class, but only 12 for division."""
    rng = np.random.default_rng(0)
    rows = []
    for label, folder in FOLDERS.items():
        (tmp_path / folder).mkdir()
        for i in range(12 if label == "/" else 40):
            name = f"{folder}/kaggle_{i:05d}.png"
            cv2.imwrite(str(tmp_path / name), _draw(label, rng))
            rows.append({"file": name, "label": label, "folder": folder,
                         "source": "kaggle", "writer": ""})
    with (tmp_path / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "folder", "source", "writer"])
        w.writeheader()
        w.writerows(rows)
    return tmp_path


@pytest.fixture
def fake_mnist():
    """Thick-stroke 'digits' (3 px lines) so the width check has a target."""
    def make(n):
        x = np.zeros((n, 28, 28), np.float32)
        for k in range(n):
            cv2.line(x[k], (8, 4 + k % 5), (20, 24), 1.0, 3)
        return x[..., None], np.arange(n) % 10
    (xt, yt), (xv, yv), (xs, ys) = make(100), make(20), make(20)
    return DatasetSplits(xt, yt, xv, yv, xs, ys, list(DIGIT_LABELS))


def test_symbols_are_preprocessed_to_mnist_format(symbols_dir):
    x, labels = sd.load_symbol_images(symbols_dir, use_cache=False)
    assert x.shape == (40 * 5 + 12, 28, 28, 1)
    assert x.dtype == np.float32 and 0.0 <= x.min() and x.max() <= 1.0
    # white strokes on black: the border (background) must be dark
    border = np.concatenate([x[:, 0], x[:, -1], x[:, :, 0], x[:, :, -1]], axis=1)
    assert border.mean() < 0.05
    assert set(labels) == set(SYMBOL_LABELS)


def test_thin_symbols_are_thickened_towards_mnist(symbols_dir, fake_mnist):
    x, _ = sd.load_symbol_images(symbols_dir, use_cache=False)
    thick, info = sd.match_stroke_width(x, fake_mnist.x_train)
    assert info["dilations"] >= 1
    assert info["symbol_width_after"] > info["symbol_width_before"]
    assert thick.shape == x.shape


def test_no_thickening_when_widths_already_match(fake_mnist):
    _, info = sd.match_stroke_width(fake_mnist.x_train, fake_mnist.x_train)
    assert info["dilations"] == 0


def test_split_has_no_overlap_and_balances_train_only(symbols_dir):
    x, labels = sd.load_symbol_images(symbols_dir, use_cache=False)
    # tag every image with a unique id in a corner pixel to track it
    ids = np.arange(len(x), dtype=np.float32)
    tagged = x.copy()
    tagged[:, 0, 0, 0] = ids / 1000.0
    parts = sd.split_symbols(tagged, labels, min_train_per_class=50)
    get = lambda name: set(np.round(parts[name][0][:, 0, 0, 0] * 1000).astype(int))
    assert not (get("train") & get("test"))
    assert not (get("train") & get("val"))
    assert not (get("val") & get("test"))
    c = parts["counts"]["/"]
    assert c["train_unique"] < 50 and c["train_after_balance"] == 50   # repeated
    assert c["val"] + c["test"] + c["train_unique"] == 12               # val/test untouched
    assert set(np.unique(parts["train"][1])) == set(range(10, 16))


def test_cap_per_class(symbols_dir):
    x, labels = sd.load_symbol_images(symbols_dir, use_cache=False)
    parts = sd.split_symbols(x, labels, max_per_class=20, min_train_per_class=0)
    c = parts["counts"]["+"]
    assert c["train_unique"] + c["val"] + c["test"] == 20


def test_combined_dataset_has_16_classes(symbols_dir, fake_mnist):
    ds = sd.load_mnist_symbols(fake_mnist, symbols_dir)
    assert ds.label_names == DIGIT_LABELS + SYMBOL_LABELS
    assert ds.num_classes == 16
    for y in (ds.y_train, ds.y_val, ds.y_test):
        assert y.min() >= 0 and y.max() <= 15
    assert set(np.unique(ds.y_test)) == set(range(16))
    assert len(ds.x_test) == len(fake_mnist.x_test) + sum(
        c["test"] for c in sd.split_symbols(*sd.load_symbol_images(symbols_dir))["counts"].values())


def test_cache_is_reused_and_matches(symbols_dir):
    a, la = sd.load_symbol_images(symbols_dir)                 # builds cache
    assert list(symbols_dir.glob("_cache_*.npz"))
    b, lb = sd.load_symbol_images(symbols_dir)                 # reads cache
    np.testing.assert_allclose(a, b, atol=1 / 255)
    assert (la == lb).all()


def test_missing_manifest_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_symbols.py"):
        sd.load_symbol_images(tmp_path)


def test_trainer_runs_on_16_classes(symbols_dir, fake_mnist, tmp_path):
    from trainer import TrainConfig, train
    data = sd.load_mnist_symbols(fake_mnist, symbols_dir)
    cfg = TrainConfig(run_id="sym", arch="lenet", dataset="mnist+symbols",
                      max_epochs=1, batch_size=64, output_root=str(tmp_path / "out"))
    summary = train(cfg, data=data)
    assert 0.0 <= summary["test_accuracy"] <= 1.0


def test_dim_symbols_are_brightened_to_mnist_range(symbols_dir):
    x, _ = sd.load_symbol_images(symbols_dir, use_cache=False)
    assert np.median(x.reshape(len(x), -1).max(axis=1)) < 0.9   # thin lines come out grey
    bright = sd.normalise_brightness(x)
    peaks = bright.reshape(len(bright), -1).max(axis=1)
    np.testing.assert_allclose(peaks[peaks > 0], 1.0, rtol=1e-6)


def test_faint_strokes_still_get_a_width():
    img = np.zeros((28, 28, 1), np.float32)
    img[14, 4:24, 0] = 0.3                      # a faint 1-px line
    assert sd.stroke_width(img) == pytest.approx(1.0, abs=0.2)


def test_cap_keeps_every_non_kaggle_image_first():
    rng = np.random.default_rng(0)
    sources = np.array(["hasy"] * 5 + ["kaggle"] * 95)
    picked = sd.cap_class(np.arange(100), sources, 20, rng)
    assert len(picked) == 20
    assert set(range(5)) <= set(picked)                 # all 5 HASYv2 kept
    assert len(set(picked)) == 20                       # no duplicates
    assert list(picked[:5]) != list(range(5))           # shuffled, not sorted


def test_split_reports_kept_hasy_count(symbols_dir):
    x, labels, sources = sd.load_symbol_images(symbols_dir, use_cache=False, return_sources=True)
    sources = sources.copy()
    plus = np.flatnonzero(labels == "+")
    sources[plus[:3]] = "hasy"                          # 3 of 40 '+' from HASYv2
    parts = sd.split_symbols(x, labels, max_per_class=10, min_train_per_class=0,
                             sources=sources)
    assert parts["counts"]["+"]["kept_non_kaggle"] == 3


def test_old_cache_without_sources_is_rebuilt(symbols_dir):
    x, labels = sd.load_symbol_images(symbols_dir)     # writes the cache
    cache = next(symbols_dir.glob("_cache_*.npz"))
    np.savez_compressed(cache, x=np.zeros((1, 28, 28, 1), np.uint8),
                        labels=np.array(["+"]), key=np.array("old-format"))
    x2, l2, s2 = sd.load_symbol_images(symbols_dir, return_sources=True)
    assert len(x2) == len(x) and len(s2) == len(x)
