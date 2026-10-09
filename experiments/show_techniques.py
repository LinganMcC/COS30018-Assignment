"""Every technique compared for Task 1 and Task 2, with its results, in one place.

Reads the result CSVs the experiments already wrote, so it runs in a couple of
seconds with no model and no TensorFlow. Also draws two figures on REAL
handwritten images, so the techniques can be seen, not just scored.

Run:
    python experiments/show_techniques.py

Outputs:
    tables in the terminal
    reports/preprocessing_real_visual.png   8 preprocessing techniques on real digits
    reports/segmentation_real_visual.png    4 segmentation techniques on real numbers
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import CONFIGS, SELECTED_CONFIG, preprocess
from segmentation import METHODS, SELECTED_METHOD, annotate

REPORTS = ROOT / "reports"
SAMPLES = ROOT / "data" / "custom_samples"

PRE_EXPLAIN = {
    "grayscale_only":         "grey, then resize straight to 28x28",
    "otsu":                   "grey + Otsu: one automatic ink/paper threshold",
    "otsu_denoised":          "median blur to remove specks, then Otsu",
    "otsu_denoised_centered": "blur + Otsu, then shift to the centre of mass",
    "adaptive":               "blur + a threshold computed per local area",
    "grayscale_mnist_box":    "grey, then fit into MNIST's 20x20 box",
    "adaptive_mnist_box":     "adaptive threshold, then the 20x20 box",
    "otsu_mnist_box":         "Otsu, then the 20x20 box",
}
SEG_EXPLAIN = {
    "contours":             "trace the outline of each connected ink region",
    "connected_components": "label every group of touching ink pixels",
    "projection":           "count ink per column, cut where a column is empty",
    "watershed":            "grow regions from the centre of each stroke",
}
MODEL_EXPLAIN = [
    ("sliding_mlp", "MLP (John)",           "fully connected network, has a 'not a digit' class"),
    ("sliding_cnn", "CNN LeNet (Thien)",    "convolutional network, 10 digit classes only"),
    ("sliding_svm", "SVM (John)",           "RBF kernel, boundary from support vectors"),
    ("sliding_rf",  "Random Forest (John)", "300 decision trees voting on pixel values"),
]


def read(name: str, key: str) -> dict:
    path = REPORTS / name
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as fh:
        return {row[key]: row for row in csv.DictReader(fh)}


def pct(row: dict | None, key: str) -> str:
    if not row or row.get(key) in (None, ""):
        return "-"
    return f"{float(row[key]):.0%}"


def section_preprocessing() -> None:
    mnist = read("preprocessing_comparison.csv", "config")
    real = read("custom_preprocessing.csv", "config")
    print("TASK 1 - 8 PREPROCESSING TECHNIQUES")
    print("  MNIST: team CNN retrained 5 times per technique.  Real: 70 handwritten digits.\n")
    print(f"  {'#':>2}  {'technique':24s} {'MNIST':>6s} {'real':>6s}  what it does")
    print("  " + "-" * 92)
    for i, name in enumerate(CONFIGS, 1):
        mark = "  <- selected" if name == SELECTED_CONFIG else ""
        mnist_val = pct(mnist.get(name), "cnn_mean") if "cnn_mean" in next(iter(mnist.values()), {}) \
            else pct(mnist.get(name), "cnn_accuracy")
        print(f"  {i:>2}  {name:24s} {mnist_val:>6s} {pct(real.get(name), 'accuracy'):>6s}  "
              f"{PRE_EXPLAIN.get(name, '')}{mark}")
    print()


def section_segmentation() -> None:
    generated = read("segmentation_comparison.csv", "method")
    real = read("custom_segmentation.csv", "method")
    print("TASK 2 - 4 SEGMENTATION TECHNIQUES (no model)")
    print("  Generated: 30 numbers.  Real: 27 handwritten numbers, 8 with touching digits.\n")
    print(f"  {'#':>2}  {'technique':22s} {'generated':>9s} {'real':>6s} {'read':>6s}  what it does")
    print("  " + "-" * 92)
    for i, name in enumerate(SEG_EXPLAIN, 1):
        mark = "  <- selected" if name == SELECTED_METHOD else ""
        print(f"  {i:>2}  {name:22s} {pct(generated.get(name), 'accuracy'):>9s} "
              f"{pct(real.get(name), 'count_accuracy'):>6s} "
              f"{pct(real.get(name), 'exact_number_accuracy'):>6s}  {SEG_EXPLAIN[name]}{mark}")
    print("  generated/real = right number of digits found;  read = whole number read exactly\n")


def section_models() -> None:
    generated = read("segmentation_comparison.csv", "method")
    print("TASK 2 - 4 MODELS behind the same sliding-window search")
    print("  Right number of digits found on 30 generated numbers.\n")
    print(f"  {'#':>2}  {'model':22s} {'overall':>8s} {'touching':>9s}  what it is")
    print("  " + "-" * 92)
    for i, (key, label, text) in enumerate(MODEL_EXPLAIN, 1):
        row = generated.get(key)
        print(f"  {i:>2}  {label:22s} {pct(row, 'accuracy'):>8s} {pct(row, 'acc_touching'):>9s}  {text}")
    print(f"      {'contours, for reference':22s} {pct(generated.get('contours'), 'accuracy'):>8s}"
          f" {pct(generated.get('contours'), 'acc_touching'):>9s}  no model at all\n")


def pick(folder: Path, names: list[str]) -> list[Path]:
    found = [folder / n for n in names if (folder / n).exists()]
    if found:
        return found
    return sorted(folder.glob("*.jpg"))[:len(names)]


def figure_preprocessing(plt) -> Path | None:
    files = pick(SAMPLES / "digits", ["2_03.jpg", "5_05.jpg", "7_04.jpg"])
    if not files:
        return None
    real = read("custom_preprocessing.csv", "config")
    configs = list(CONFIGS)
    fig, axes = plt.subplots(len(files), len(configs) + 1,
                             figsize=(1.7 * (len(configs) + 1), 2.1 * len(files)), dpi=140)
    for r, path in enumerate(files):
        image = cv2.imread(str(path))
        axes[r, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[r, 0].set_ylabel(f"label {path.stem.split('_')[0]}", fontsize=8)
        for c, name in enumerate(configs, 1):
            axes[r, c].imshow(preprocess(image, config=name), cmap="gray", vmin=0, vmax=1)
            if r == 0:
                acc = pct(real.get(name), "accuracy")
                star = " *" if name == SELECTED_CONFIG else ""
                axes[r, c].set_title(f"{name.replace('_', chr(10))}\nreal {acc}{star}", fontsize=7)
        if r == 0:
            axes[r, 0].set_title("photo", fontsize=8)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Task 1 - the 8 preprocessing techniques on real handwritten digits "
                 "(* = selected)", fontsize=10)
    fig.tight_layout()
    out = REPORTS / "preprocessing_real_visual.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def figure_segmentation(plt) -> Path | None:
    files = pick(SAMPLES / "numbers", ["5189_01.jpg", "13579_01.jpg", "896_t1.jpg"])
    if not files:
        return None
    fig, axes = plt.subplots(len(files), len(METHODS),
                             figsize=(3.2 * len(METHODS), 1.9 * len(files)), dpi=140)
    for r, path in enumerate(files):
        image = cv2.imread(str(path))
        label = path.stem.split("_")[0]
        for c, (name, segment) in enumerate(METHODS.items()):
            crops, boxes = segment(image)
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(annotate(image, boxes), cv2.COLOR_BGR2RGB))
            ax.set_xticks([])
            ax.set_yticks([])
            ok = len(crops) == len(label)
            ax.set_xlabel(f"found {len(crops)}/{len(label)}", fontsize=8,
                          color="#1a7f37" if ok else "#c00000")
            if r == 0:
                star = "  (selected)" if name == SELECTED_METHOD else ""
                ax.set_title(name + star, fontsize=9)
            if c == 0:
                kind = "touching" if "_t" in path.stem else "separate"
                ax.set_ylabel(f"{label}\n({kind})", fontsize=8)
    fig.suptitle("Task 2 - the 4 segmentation techniques on real handwritten numbers",
                 fontsize=10)
    fig.tight_layout()
    out = REPORTS / "segmentation_real_visual.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("=" * 96)
    section_preprocessing()
    section_segmentation()
    section_models()
    print("=" * 96)
    for out in (figure_preprocessing(plt), figure_segmentation(plt)):
        if out:
            print(f"Saved: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
