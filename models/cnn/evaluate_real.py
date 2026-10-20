"""
ext_best is run twice: unrestricted (it may answer "/" for a "1") and with the

Usage (from models/cnn/):
    python evaluate_real.py
    python evaluate_real.py --runs s3_A1_aug_none s3_A2_aug_light s3_A3_aug_medium s3_A4_aug_strong
        -> also scores those tuning runs (checkpoints/tuning/, local only) and writes
           real_handwriting_runs.csv; used to ask whether stronger augmentation
           helps on real photos even though it did not help on MNIST
Outputs:
    models/experiments/eval/real_handwriting.csv
    models/experiments/eval/real_handwriting.md
    models/experiments/eval/real_handwriting_errors.csv   every wrong single digit
    models/experiments/eval/real_vs_mnist.png       MNIST test vs real digits, per model
    models/experiments/eval/real_errors_cnn_best.png  the photos cnn_best gets wrong
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from inference import CHECKPOINTS, REPO_ROOT, Recogniser

SAMPLES = REPO_ROOT / "data" / "custom_samples"
OUT_DIR = REPO_ROOT / "models" / "experiments" / "eval"
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}

# (display name, checkpoint, allowed classes). Per-model "best" runs live in
# checkpoints/tuning/ which is not committed; missing ones are skipped.
MODELS: List[Tuple[str, str, Optional[tuple]]] = [
    ("MLP (tuned)", "tuning/pm_mlp_best", None),
    ("LeNet-5 (tuned)", "tuning/pm_lenet_best", None),
    ("ResNet (tuned)", "tuning/pm_resnet_best", None),
    ("VGG-deep = cnn_best", "cnn_best", None),
    ("ext_best (v3, retrained), digits only", "ext_best", Recogniser.DIGITS),
    ("ext_best (v3, retrained), all 16 classes", "ext_best", None),
    ("ext_v2 (before retraining), digits only", "ext_v2", Recogniser.DIGITS),
    ("ext_v2 (before retraining), all 16 classes", "ext_v2", None),
    ("Sprint 2 LeNet (cnn_lenet)", "cnn_lenet", None),
]


def load_samples(folder: Path, single: bool) -> List[Tuple[str, np.ndarray, str]]:
    """(filename, image, label) - the label is the filename before '_'."""
    import cv2
    out = []
    if not folder.exists():
        return out
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue
        label = path.stem.split("_")[0]
        if not label.isdigit() or (single and len(label) != 1):
            continue
        img = cv2.imread(str(path))
        if img is not None:
            out.append((path.name, img, label))
    return out


# Tuning run behind each checkpoint, so MNIST accuracy is read from the log
# and the chart never drifts from the logged numbers.
SUMMARIES = {
    "tuning/pm_mlp_best": "pm_mlp_best", "tuning/pm_lenet_best": "pm_lenet_best",
    "tuning/pm_resnet_best": "pm_resnet_best", "cnn_best": "s3_B5_bs_64",
    "ext_best": "ext_v3", "ext_v2": "ext_v2",
}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """95 % Wilson score interval for k successes out of n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mnist_test_acc(ckpt: str) -> Optional[float]:
    """MNIST test accuracy of the run behind `ckpt`, from the committed tuning_log.csv."""
    return mnist_test_acc_run(SUMMARIES.get(ckpt))


def mnist_test_acc_run(run: Optional[str]) -> Optional[float]:
    log = REPO_ROOT / "models" / "experiments" / "tuning_log.csv"
    if run is None or not log.exists():
        return None
    with log.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["run_id"] == run:
                return float(row["test_accuracy"])
    return None


def save_real_vs_mnist(rows: List[dict], n_digits: int, out_path: Path) -> None:
    """Paired bars: MNIST test accuracy vs accuracy on real single digits."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in rows if r.get("mnist_acc") is not None and "digit_acc" in r]
    if not rows:
        return
    names = [r["model"] for r in rows]
    x = np.arange(len(rows))
    real = np.array([r["digit_acc"] for r in rows]) * 100
    lo = real - np.array([r["digit_ci"][0] for r in rows]) * 100
    hi = np.array([r["digit_ci"][1] for r in rows]) * 100 - real
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    ax.bar(x - 0.2, [r["mnist_acc"] * 100 for r in rows], 0.4, label="MNIST test (10,000)",
           color="#9ecae1")
    ax.bar(x + 0.2, real, 0.4, yerr=[lo, hi], capsize=3,
           label=f"Real handwriting ({n_digits} photos, 95 % CI)", color="#3182bd")
    for xi, v in zip(x, real):
        ax.text(xi + 0.2, 52, f"{v:.0f}%", ha="center", color="white", fontsize=8)
    ax.set_xticks(x, names, fontsize=8, rotation=12)
    ax.set_ylim(50, 101)
    ax.set_ylabel("Single-digit accuracy (%)")
    ax.set_title("Every model loses accuracy on real handwriting", fontsize=10)
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_error_gallery(rec: Recogniser, digits, out_path: Path, allowed=None) -> None:
    """Each photo the model gets wrong, above the 28x28 image the model saw."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import cv2
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from preprocessing import preprocess
    preds = rec.read_crops([img for _, img, _ in digits], allowed)
    wrong = [(d, p) for d, p in zip(digits, preds) if p["label"] != d[2]]
    if not wrong:
        return
    fig, axes = plt.subplots(2, len(wrong), figsize=(1.5 * len(wrong), 3.4), squeeze=False)
    for i, ((name, img, label), p) in enumerate(wrong):
        axes[0, i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, i].set_title(f"true {label}", fontsize=8)
        axes[1, i].imshow(preprocess(img), cmap="gray")
        axes[1, i].set_title(f"read {p['label']} ({p['confidence']:.2f})", fontsize=8,
                             color="#b2182b")
        for a in axes[:, i]:
            a.axis("off")
    fig.suptitle(f"{rec.name}: real digits it gets wrong (top: photo, bottom: model input)",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def score_numbers(rec: Recogniser, numbers, allowed) -> Dict[str, float]:
    exact = count_ok = digit_ok = digit_total = 0
    for _, img, label in numbers:
        text, _ = rec.read_image(img, allowed)
        exact += text == label
        if len(text) == len(label):
            count_ok += 1
            digit_total += len(label)
            digit_ok += sum(a == b for a, b in zip(text, label))
    n = max(1, len(numbers))
    return {"exact_number": exact / n, "count_right": count_ok / n,
            "digit_when_count_right": digit_ok / digit_total if digit_total else float("nan")}


def score_runs(run_ids: List[str], digits, numbers) -> None:
    """Score tuning runs on real photos - kept apart from the final-model table."""
    rows = []
    for run in run_ids:
        try:
            rec = Recogniser(f"tuning/{run}")
        except FileNotFoundError:
            print(f"  skip {run}: not in checkpoints/tuning/")
            continue
        preds = rec.read_crops([img for _, img, _ in digits])
        k = sum(p["label"] == lab for p, (_, _, lab) in zip(preds, digits))
        lo, hi = wilson(k, len(digits))
        row = {"run_id": run, "mnist_test": mnist_test_acc_run(run),
               "real_digits": k / len(digits), "ci_low": lo, "ci_high": hi}
        row.update(score_numbers(rec, numbers, None))
        rows.append(row)
        print(f"  {run:20s} real digits {row['real_digits']:6.1%} ({lo:.0%}-{hi:.0%}) | "
              f"digit acc (count right) {row['digit_when_count_right']:6.1%}")
    if rows:
        path = OUT_DIR / "real_handwriting_runs.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows([{k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()}
                         for r in rows])
        print(f"-> {path}")


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="Final models on real handwriting.")
    ap.add_argument("--runs", nargs="+", default=[],
                    help="Also score these tuning run_ids from checkpoints/tuning/.")
    args = ap.parse_args(argv)

    digits = load_samples(SAMPLES / "digits", single=True)
    numbers = load_samples(SAMPLES / "numbers", single=False)
    print(f"{len(digits)} digit photos, {len(numbers)} number photos")
    if not digits and not numbers:
        sys.exit(f"No photos in {SAMPLES}")

    rows, errors = [], []
    for display, ckpt, allowed in MODELS:
        try:
            rec = Recogniser(ckpt)
        except FileNotFoundError:
            print(f"  skip {display}: {ckpt}.keras not found")
            continue
        row = {"model": display, "checkpoint": ckpt, "mnist_acc": mnist_test_acc(ckpt)}
        if digits:
            preds = rec.read_crops([img for _, img, _ in digits], allowed)
            correct = 0
            for (name, _, label), p in zip(digits, preds):
                if p["label"] == label:
                    correct += 1
                else:
                    errors.append({"model": display, "file": name, "true": label,
                                   "predicted": p["label"],
                                   "confidence": f"{p['confidence']:.3f}"})
            row["digit_acc"] = correct / len(digits)
            row["digit_ci"] = wilson(correct, len(digits))
            row["mean_conf"] = float(np.mean([p["confidence"] for p in preds]))
        if numbers:
            row.update(score_numbers(rec, numbers, allowed))
        rows.append(row)
        print(f"  {display:28s} digits {row.get('digit_acc', 0):6.1%} | "
              f"exact numbers {row.get('exact_number', 0):6.1%} | "
              f"digit acc (count right) {row.get('digit_when_count_right', 0):6.1%}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    keys = ["model", "checkpoint", "digit_acc", "mean_conf", "exact_number",
            "count_right", "digit_when_count_right"]
    with (OUT_DIR / "real_handwriting.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k, ""))
                        for k in keys})
    with (OUT_DIR / "real_handwriting_errors.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["model", "file", "true", "predicted", "confidence"])
        w.writeheader()
        w.writerows(errors)

    md = ["# Final models on real handwriting", "",
          f"{len(digits)} single-digit photos, {len(numbers)} number photos "
          "(`data/custom_samples/`). Segmentation: contours; preprocessing: otsu_mnist_box.", "",
          "| Model | MNIST test | Real single digits (95 % CI) | Exact numbers | "
          "Digit count right | Digit acc. when count right |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        mn = f"{100 * r['mnist_acc']:.2f} %" if r.get("mnist_acc") else "–"
        ci = r.get("digit_ci", (0, 0))
        md.append(f"| {r['model']} | {mn} | {100 * r.get('digit_acc', 0):.1f} % "
                  f"({100 * ci[0]:.0f}–{100 * ci[1]:.0f}) | "
                  f"{100 * r.get('exact_number', 0):.1f} % | {100 * r.get('count_right', 0):.1f} % | "
                  f"{100 * r.get('digit_when_count_right', 0):.1f} % |")
    md += ["", "Digit count right is set by segmentation alone, so it is the same for every "
           "model; the last column isolates recognition errors."]
    (OUT_DIR / "real_handwriting.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    save_real_vs_mnist([r for r in rows if r["checkpoint"] != "cnn_lenet"
                        and "all 16 classes" not in r["model"]],
                       len(digits), OUT_DIR / "real_vs_mnist.png")
    if digits:
        try:
            save_error_gallery(Recogniser("cnn_best"), digits,
                               OUT_DIR / "real_errors_cnn_best.png")
        except FileNotFoundError:
            pass
    if args.runs:
        score_runs(args.runs, digits, numbers)
    print(f"-> {OUT_DIR}")


if __name__ == "__main__":
    main()
