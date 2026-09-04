"""
Task 1 evidence - quantitative comparison of preprocessing techniques.
COS30018 Option B - Handwritten Number Recognition System
Owner: John (Person A)

The 5 marks for Task 1 are not for writing a resize function; they are for
"research and experiment with different preprocessing techniques and select the
appropriate technique for your project". This script produces that evidence.

For each preprocessing configuration it:
  1. trains the same simple classifier on identically-sized MNIST subsets,
  2. records test accuracy and processing time,
  3. saves a side-by-side visual of what each configuration does to one image.

The classifier is deliberately simple and identical across runs - the point is
to isolate the effect of preprocessing, not to find the best model (that is
Thien's and Liam's job in Task 3).

Run:
    python experiments/compare_preprocessing.py

Outputs:
    reports/preprocessing_comparison.csv
    reports/preprocessing_visual.png
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import CONFIGS, preprocess          # noqa: E402

REPORTS = ROOT / "reports"
TRAIN_SIZE = 4000       # kept small so the experiment runs in about a minute
TEST_SIZE = 1000


def load_mnist_subset():
    """Load a fixed subset of MNIST for the comparison."""
    cache = ROOT / "data" / "mnist.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as data:
            return (data["x_train"][:TRAIN_SIZE], data["y_train"][:TRAIN_SIZE],
                    data["x_test"][:TEST_SIZE], data["y_test"][:TEST_SIZE])

    from tensorflow.keras.datasets import mnist
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    return x_tr[:TRAIN_SIZE], y_tr[:TRAIN_SIZE], x_te[:TEST_SIZE], y_te[:TEST_SIZE]


def evaluate_config(name: str, x_train, y_train, x_test, y_test) -> dict:
    """Preprocess with one configuration, train, and score."""
    from sklearn.neighbors import KNeighborsClassifier

    start = time.time()
    # MNIST arrives light-on-dark; invert so it looks like a scanned page and the
    # preprocessing pipeline is exercised the same way it will be in production.
    train_proc = np.array([preprocess(cv2.bitwise_not(img), config=name, flatten=True)
                           for img in x_train])
    test_proc = np.array([preprocess(cv2.bitwise_not(img), config=name, flatten=True)
                          for img in x_test])
    prep_time = time.time() - start

    clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    clf.fit(train_proc, y_train)
    accuracy = clf.score(test_proc, y_test)

    return {
        "config": name,
        "description": CONFIGS[name]["description"],
        "test_accuracy": round(float(accuracy), 4),
        "preprocessing_seconds": round(prep_time, 2),
        "ms_per_image": round(1000 * prep_time / (len(x_train) + len(x_test)), 3),
    }


def save_visual_comparison(sample: np.ndarray) -> Path:
    """Save one image processed by every configuration, side by side."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(CONFIGS.keys())
    fig, axes = plt.subplots(1, len(names) + 1, figsize=(2.1 * (len(names) + 1), 2.8))

    axes[0].imshow(sample, cmap="gray")
    axes[0].set_title("original", fontsize=9)
    axes[0].axis("off")

    for ax, name in zip(axes[1:], names):
        ax.imshow(preprocess(sample, config=name), cmap="gray")
        ax.set_title(name.replace("_", "\n"), fontsize=8)
        ax.axis("off")

    fig.suptitle("Task 1 - effect of each preprocessing configuration", fontsize=11)
    fig.tight_layout()

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "preprocessing_visual.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    import csv

    print("Task 1 - preprocessing comparison experiment")
    print("=" * 62)
    x_train, y_train, x_test, y_test = load_mnist_subset()
    print(f"Training on {len(x_train)} images, testing on {len(x_test)}.")
    print("Same classifier (k-NN, k=3) for every configuration, so any difference")
    print("in accuracy is caused by preprocessing alone.\n")

    results = []
    for name in CONFIGS:
        print(f"  running {name:26s} ...", end=" ", flush=True)
        row = evaluate_config(name, x_train, y_train, x_test, y_test)
        results.append(row)
        print(f"accuracy {row['test_accuracy']:.4f}   ({row['ms_per_image']} ms/image)")

    results.sort(key=lambda r: r["test_accuracy"], reverse=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "preprocessing_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    visual = save_visual_comparison(cv2.bitwise_not(x_test[0]))

    best = results[0]
    print("\n" + "=" * 62)
    print(f"Best configuration: {best['config']}  ({best['test_accuracy']:.4f})")
    print(f"  {best['description']}")
    print(f"\nSaved: {csv_path.relative_to(ROOT)}")
    print(f"Saved: {visual.relative_to(ROOT)}")
    print("\nBoth files go straight into the report's 'Data preprocessing' section.")


if __name__ == "__main__":
    main()
