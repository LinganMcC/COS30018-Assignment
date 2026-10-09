"""Run the whole pipeline on ONE image and print every intermediate result.

For showing exactly what segmentation and preprocessing produce: the box
coordinates of each digit, the crop sizes, the 28x28 input handed to the CNN,
and the digit it reads.

Run:
    python experiments/demo_one.py                                  # default example
    python experiments/demo_one.py data/custom_samples/numbers/896_t1.jpg
    python experiments/demo_one.py IMAGE --method sliding_cnn       # any segmentation method
    python experiments/demo_one.py IMAGE --compare                  # contours vs the 4 models

Output:
    printed table in the terminal
    reports/demo_one.png       the image with boxes, every crop, and every 28x28 input
    reports/demo_compare.png   with --compare: one row per method on the same image
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from preprocessing import SELECTED_CONFIG, preprocess
from segmentation import METHODS as CLASSICAL, SELECTED_METHOD, annotate
from segmentation_ml import ML_METHODS

ALL_METHODS = {**CLASSICAL, **ML_METHODS}
COMPARE = ["contours", "sliding_mlp", "sliding_cnn", "sliding_svm", "sliding_rf"]

DEFAULT = ROOT / "data" / "custom_samples" / "numbers" / "5189_01.jpg"
CNN_PATH = ROOT / "models" / "checkpoints" / "cnn_lenet.keras"


def load_reader():
    """Thien's CNN if TensorFlow is available, otherwise the MLP."""
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(CNN_PATH))

        def read(x):
            p = model.predict(x[np.newaxis, :, :, np.newaxis], verbose=0)[0]
            return int(p.argmax()), float(p.max())
        return "CNN LeNet (Thien)", read
    except Exception:                                    # noqa: BLE001
        from digit_classifier import get_classifier
        mlp = get_classifier("mlp")

        def read(x):
            p = mlp.predict_proba(x.reshape(1, -1))[0, :10]
            return int(p.argmax()), float(p.max() / p.sum())
        return "MLP (TensorFlow not available)", read


def read_number(image, boxes, read):
    """Crop each box from the original photo, preprocess it, and read it."""
    inputs, digits, confs = [], [], []
    for (x, y, w, h) in boxes:
        model_input = preprocess(image[y:y + h, x:x + w])     # 28x28, values 0..1
        digit, conf = read(model_input)
        inputs.append(model_input)
        digits.append(str(digit))
        confs.append(conf)
    return inputs, digits, confs


def single(path: Path, image, label: str, method: str, reader_name: str, read) -> None:
    height, width = image.shape[:2]
    crops, boxes = ALL_METHODS[method](image)

    print("=" * 78)
    print(f"Input image   : {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
    print(f"Image size    : {width} x {height} pixels   (origin 0,0 = top-left corner)")
    print(f"Ground truth  : {label}   (taken from the filename)")
    print(f"Segmentation  : {method}")
    print(f"Preprocessing : {SELECTED_CONFIG}")
    print(f"Classifier    : {reader_name}")
    print("=" * 78)
    print(f"Found {len(boxes)} digit region(s), ordered left to right:\n")
    print(f"  {'#':>2}  {'x':>5} {'y':>5} {'w':>5} {'h':>5}   {'crop size':>10}   "
          f"{'model input':>12}   {'read':>4}  {'confidence':>10}")
    print("  " + "-" * 74)

    inputs, digits, confs = read_number(image, boxes, read)
    for i, ((x, y, w, h), model_input, digit, conf) in enumerate(
            zip(boxes, inputs, digits, confs), 1):
        print(f"  {i:>2}  {x:>5} {y:>5} {w:>5} {h:>5}   {w:>4} x {h:<4}   "
              f"{str(model_input.shape):>12}   {digit:>4}  {conf:>9.0%}")

    result = "".join(digits)
    print("  " + "-" * 74)
    print(f"\n  Read as      : {result}")
    print(f"  Ground truth : {label}")
    print(f"  Digit count  : {'correct' if len(boxes) == len(label) else 'WRONG'} "
          f"({len(boxes)} found, {len(label)} expected)")
    print(f"  Whole number : {'CORRECT' if result == label else 'WRONG'}")
    print("\n  x, y = top-left corner of the box;  w, h = its width and height, in pixels")
    save_figure(image, boxes, crops, inputs, digits, label, result, method)


def compare(path: Path, image, label: str, reader_name: str, read) -> None:
    """Same image through contours and the four sliding-window models."""
    print("=" * 78)
    print(f"Input image  : {path.name}   ground truth {label}")
    print(f"Comparing    : {', '.join(COMPARE)}")
    print(f"Every crop read by {reader_name}, preprocessed with {SELECTED_CONFIG}")
    print("=" * 78)
    print(f"  {'method':16s} {'boxes':>5}  {'boxes (x, width)':40s} {'read':>8}  count  number")
    print("  " + "-" * 92)

    rows = []
    for method in COMPARE:
        try:
            _, boxes = ALL_METHODS[method](image)
        except (ImportError, FileNotFoundError):
            print(f"  {method:16s} skipped - needs TensorFlow")
            continue
        _, digits, _ = read_number(image, boxes, read)
        result = "".join(digits)
        spans = " ".join(f"({x},{w})" for x, _, w, _ in boxes)
        count_ok = len(boxes) == len(label)
        exact_ok = result == label
        print(f"  {method:16s} {len(boxes):>5}  {spans[:40]:40s} {result:>8}  "
              f"{'ok' if count_ok else 'WRONG':>5}  {'ok' if exact_ok else 'WRONG':>6}")
        rows.append((method, boxes, digits, result, count_ok, exact_ok))
    print("  " + "-" * 92)
    print("  count = right number of digits found;  number = whole number read exactly")
    save_compare(image, label, rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the pipeline on one image.")
    parser.add_argument("image", nargs="?", default=str(DEFAULT))
    parser.add_argument("--method", choices=list(ALL_METHODS), default=SELECTED_METHOD)
    parser.add_argument("--compare", action="store_true",
                        help="contours against the four sliding-window models")
    args = parser.parse_args()

    path = Path(args.image)
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Could not read {path}")
    label = path.stem.split("_")[0]
    reader_name, read = load_reader()

    if args.compare:
        compare(path, image, label, reader_name, read)
    else:
        single(path, image, label, args.method, reader_name, read)


def save_figure(image, boxes, crops, inputs, digits, label, result, method) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = max(1, len(boxes))
    fig = plt.figure(figsize=(max(8, 2.2 * n), 6.2), dpi=140)
    grid = fig.add_gridspec(3, n, height_ratios=[2.2, 1, 1])

    top = fig.add_subplot(grid[0, :])
    top.imshow(cv2.cvtColor(annotate(image, boxes, labels=digits), cv2.COLOR_BGR2RGB))
    top.set_title(f"1. segmentation ({method}) - ground truth {label}, "
                  f"read {result}", fontsize=10)
    top.axis("off")

    for i, (box, crop, model_input, digit) in enumerate(zip(boxes, crops, inputs, digits)):
        ax = fig.add_subplot(grid[1, i])
        x, y, w, h = box
        ax.imshow(cv2.cvtColor(image[y:y + h, x:x + w], cv2.COLOR_BGR2RGB))
        ax.set_title(f"crop {i + 1}\n(x={x}, y={y}, {w}x{h})", fontsize=7)
        ax.axis("off")

        ax = fig.add_subplot(grid[2, i])
        ax.imshow(model_input, cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"28x28 -> reads {digit}", fontsize=8)
        ax.axis("off")

    fig.text(0.01, 0.43, "2. crops", rotation=90, fontsize=9)
    fig.text(0.01, 0.10, f"3. {SELECTED_CONFIG}", rotation=90, fontsize=9)
    fig.tight_layout()
    out = ROOT / "reports" / "demo_one.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out.relative_to(ROOT)}")


def save_compare(image, label, rows) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(rows), 1, figsize=(6.5, 2.4 * len(rows)), dpi=140)
    axes = [axes] if len(rows) == 1 else list(axes)
    for ax, (method, boxes, digits, result, count_ok, exact_ok) in zip(axes, rows):
        ax.imshow(cv2.cvtColor(annotate(image, boxes, labels=digits), cv2.COLOR_BGR2RGB))
        ax.set_xticks([])
        ax.set_yticks([])
        verdict = "CORRECT" if exact_ok else ("count ok, misread" if count_ok else "wrong count")
        colour = "#1a7f37" if exact_ok else "#c00000"
        ax.set_title(f"{method}:  {len(boxes)} boxes, read {result or '-'}  ({verdict})",
                     fontsize=9, color=colour)
    fig.suptitle(f"Same image, five segmentation methods - ground truth {label}", fontsize=11)
    fig.tight_layout()
    out = ROOT / "reports" / "demo_compare.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
