"""Task 1 - compare the preprocessing configurations using the team's own CNN.

Each configuration is scored by retraining one of the team's CNN
architectures from scratch on data preprocessed that way, then measuring
test accuracy. Training settings are identical for every configuration, so
the difference in accuracy comes from preprocessing alone.

k-NN was used for this earlier. It is kept behind --with-knn because the
ranking it gives is worth reporting next to the CNN: the best preprocessing
depends on the model that consumes it.

Run:
    python experiments/compare_preprocessing.py
    python experiments/compare_preprocessing.py --model vgg_small --with-knn

Outputs:
    reports/preprocessing_comparison.csv
    reports/preprocessing_visual.png
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "models" / "cnn"))

from preprocessing import CONFIGS, preprocess

REPORTS = ROOT / "reports"
TRAIN_SIZE = 12000
TEST_SIZE = 3000
VAL_SPLIT = 0.15
BATCH_SIZE = 128
SEED = 42

# The team's architectures, weakest first. Thien owns these.
TEAM_MODELS = {
    "shallow": ("cnn_shallow", "build_shallow"),
    "lenet": ("cnn_lenet", "build_lenet5"),
    "vgg_small": ("cnn_vgg_small", "build_vgg_small"),
}


def get_model_spec(name: str):
    """Import a team model and read the training settings from its own script.

    Each architecture defines its own MAX_EPOCHS and PATIENCE. Reusing them
    matters: vgg_small needs 30 epochs and patience 5, and forcing the
    shallower models' 20/3 on it stops training before it leaves chance level.
    """
    module_name, func_name = TEAM_MODELS[name]
    module = __import__(module_name)
    return getattr(module, func_name), module.MAX_EPOCHS, module.PATIENCE


def load_mnist_subset():
    """Load a fixed subset of MNIST."""
    cache = ROOT / "data" / "mnist.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=True) as data:
            return (data["x_train"][:TRAIN_SIZE], data["y_train"][:TRAIN_SIZE],
                    data["x_test"][:TEST_SIZE], data["y_test"][:TEST_SIZE])

    from tensorflow.keras.datasets import mnist
    (x_tr, y_tr), (x_te, y_te) = mnist.load_data()
    return x_tr[:TRAIN_SIZE], y_tr[:TRAIN_SIZE], x_te[:TEST_SIZE], y_te[:TEST_SIZE]


def apply_config(images, config: str, flatten: bool = False):
    """Preprocess a stack of MNIST images with one configuration.

    MNIST is inverted first so it looks like dark ink on paper, which is what
    the pipeline sees in production.
    """
    out = np.array([preprocess(cv2.bitwise_not(img), config=config, flatten=flatten)
                    for img in images])
    return out if flatten else out.reshape(-1, 28, 28, 1)


def train_and_score(spec, x_train, y_train, x_test, y_test, seed: int = SEED):
    """Train the team's CNN from scratch and return (accuracy, epochs, seconds)."""
    import tensorflow as tf
    from tensorflow.keras import callbacks

    builder, max_epochs, patience = spec
    tf.keras.utils.set_random_seed(seed)
    model = builder()

    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=patience,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                    patience=2, min_lr=1e-5, verbose=0),
    ]
    start = time.time()
    history = model.fit(
        x_train, y_train,
        validation_split=VAL_SPLIT,
        epochs=max_epochs,
        batch_size=BATCH_SIZE,
        callbacks=cbs,
        verbose=0,
    )
    train_time = time.time() - start
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return float(accuracy), len(history.history["loss"]), train_time


def score_knn(x_train, y_train, x_test, y_test) -> float:
    """Score the same data with the original k-NN, for comparison."""
    from sklearn.neighbors import KNeighborsClassifier

    clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
    clf.fit(x_train, y_train)
    return float(clf.score(x_test, y_test))


def evaluate_config(name: str, spec, data, with_knn: bool, repeats: int) -> dict:
    """Preprocess with one configuration, then train and score the CNN `repeats` times.

    One run per configuration is not enough. Two single runs of this exact
    script ranked the configurations differently: grayscale_only went from
    0.966 to 0.960 and otsu from 0.965 to 0.973, because CPU training is not
    bit-for-bit repeatable and early stopping then fires at a different epoch.
    That run-to-run swing is as large as the gaps being measured, so each
    configuration is trained several times with different seeds and reported
    as a mean and standard deviation.
    """
    x_tr_raw, y_tr, x_te_raw, y_te = data

    start = time.time()
    x_train = apply_config(x_tr_raw, name)
    x_test = apply_config(x_te_raw, name)
    prep_time = time.time() - start

    runs = [train_and_score(spec, x_train, y_tr, x_test, y_te, seed=SEED + i)
            for i in range(repeats)]
    accs = np.array([acc for acc, _, _ in runs])

    row = {
        "config": name,
        "description": CONFIGS[name]["description"],
        "cnn_mean": round(float(accs.mean()), 4),
        "cnn_std": round(float(accs.std(ddof=1)) if repeats > 1 else 0.0, 4),
        "cnn_min": round(float(accs.min()), 4),
        "cnn_max": round(float(accs.max()), 4),
        "runs": repeats,
        "epochs_mean": round(float(np.mean([e for _, e, _ in runs])), 1),
        "train_seconds_mean": round(float(np.mean([t for _, _, t in runs])), 1),
        "ms_per_image": round(1000 * prep_time / (len(x_tr_raw) + len(x_te_raw)), 3),
    }
    if with_knn:
        row["knn_accuracy"] = round(
            score_knn(apply_config(x_tr_raw, name, flatten=True), y_tr,
                      apply_config(x_te_raw, name, flatten=True), y_te), 4)
    return row


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
    parser = argparse.ArgumentParser(description="Compare preprocessing configurations.")
    parser.add_argument("--model", choices=list(TEAM_MODELS), default="lenet",
                        help="which of the team's CNNs to score with")
    parser.add_argument("--with-knn", action="store_true",
                        help="also score with the original k-NN, for comparison")
    parser.add_argument("--repeats", type=int, default=5,
                        help="training runs per configuration, each with its own seed")
    args = parser.parse_args()

    print("Task 1 - preprocessing comparison experiment")
    print("=" * 66)
    spec = get_model_spec(args.model)
    data = load_mnist_subset()
    print(f"Scoring with the team's '{args.model}' CNN, retrained per configuration "
          f"({spec[1]} epochs max, patience {spec[2]} - its own settings).")
    print(f"Training on {len(data[0])} images, testing on {len(data[2])}.")
    print("Identical training settings every time, so any difference in accuracy")
    print("is caused by preprocessing alone.")
    print(f"Each configuration is trained {args.repeats} times with different seeds.\n")

    results = []
    for name in CONFIGS:
        print(f"  {name:26s} ...", end=" ", flush=True)
        row = evaluate_config(name, spec, data, args.with_knn, args.repeats)
        results.append(row)
        extra = f"   k-NN {row['knn_accuracy']:.4f}" if args.with_knn else ""
        print(f"CNN {row['cnn_mean']:.4f} +/- {row['cnn_std']:.4f}  "
              f"(range {row['cnn_min']:.4f}-{row['cnn_max']:.4f}){extra}")

    results.sort(key=lambda r: r["cnn_mean"], reverse=True)

    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "preprocessing_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    visual = save_visual_comparison(cv2.bitwise_not(data[2][0]))

    best = results[0]
    # Tied = the gap to the best mean is smaller than the two spreads combined.
    tied = [r["config"] for r in results[1:]
            if best["cnn_mean"] - r["cnn_mean"] <= best["cnn_std"] + r["cnn_std"]]
    print("\n" + "=" * 66)
    print(f"Highest mean: {best['config']}  ({best['cnn_mean']:.4f} +/- {best['cnn_std']:.4f})")
    print(f"  {best['description']}")
    if tied:
        print(f"Within run-to-run noise of it: {', '.join(tied)}")
    print(f"\nSaved: {csv_path.relative_to(ROOT)}")
    print(f"Saved: {visual.relative_to(ROOT)}")
    print("\nBoth files go into the 'Data preprocessing' section of the report.")


if __name__ == "__main__":
    main()
