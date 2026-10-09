"""Task 1 + Task 2 on real handwriting - the comparison that MNIST cannot give.

Every earlier result was measured on MNIST or on numbers generated from MNIST,
which is the case least favourable to thresholding and denoising. This script
repeats both comparisons on photographed handwriting.

Put the photos here, with the label in the filename:

    data/custom_samples/digits/3_01.jpg        one digit, label 3
    data/custom_samples/numbers/94026_01.jpg   a whole number, label 94026

Run:
    python experiments/evaluate_custom.py                  # team CNN (needs TensorFlow)
    python experiments/evaluate_custom.py --model mlp      # no TensorFlow needed
    python experiments/evaluate_custom.py --all-methods    # include the model-based segmenters
    python experiments/evaluate_custom.py --config grayscale_mnist_box   # crop preprocessing
    python experiments/evaluate_custom.py --all-configs    # every method x configuration

Outputs:
    reports/custom_preprocessing.csv    digit accuracy per preprocessing configuration
    reports/custom_segmentation.csv     count accuracy and exact-number accuracy per method
    reports/custom_pipeline.csv         with --all-configs: method x configuration
    reports/custom_demo/<method>/       annotated images, MISMATCH_ prefix on failures
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import CONFIGS, SELECTED_CONFIG, preprocess
from segmentation import METHODS as CLASSICAL, SELECTED_METHOD, annotate

SAMPLES = ROOT / "data" / "custom_samples"
REPORTS = ROOT / "reports"
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}
CNN_CHECKPOINT = ROOT / "models" / "checkpoints" / "cnn_lenet.keras"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_samples(folder: Path, single_digit: bool) -> list[tuple[str, np.ndarray, str]]:
    """Return (filename, image, label). The label is the filename before '_'."""
    if not folder.exists():
        return []
    heic = sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".heic")
    if heic:
        print(f"  ! {len(heic)} HEIC file(s) in {folder.name}/ cannot be read by OpenCV. "
              f"Export them as JPG. e.g. {heic[0]}")

    samples = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue
        label = path.stem.split("_")[0]
        if not label.isdigit() or (single_digit and len(label) != 1):
            print(f"  ! skipped {path.name}: filename must start with the label, e.g. 3_01.jpg")
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"  ! skipped {path.name}: could not be read")
            continue
        samples.append((path.name, image, label))
    return samples


# ---------------------------------------------------------------------------
# Classifier - the team CNN, or the MLP when TensorFlow is not available
# ---------------------------------------------------------------------------

class Classifier:
    def __init__(self, kind: str):
        self.kind = kind
        if kind == "cnn":
            import tensorflow as tf
            if not CNN_CHECKPOINT.exists():
                raise FileNotFoundError(f"{CNN_CHECKPOINT} not found")
            self.model = tf.keras.models.load_model(str(CNN_CHECKPOINT))
        else:
            from digit_classifier import get_classifier
            self.model = get_classifier("mlp")

    def predict(self, image: np.ndarray, config: str) -> int:
        if self.kind == "cnn":
            x = preprocess(image, config=config, add_channel=True)[np.newaxis]
            return int(self.model.predict(x, verbose=0).argmax())
        x = preprocess(image, config=config)[np.newaxis]
        return int(self.model.predict_proba(x)[0, :10].argmax())


def pick_classifier(requested: str) -> Classifier:
    if requested == "cnn":
        try:
            return Classifier("cnn")
        except (ImportError, FileNotFoundError) as exc:
            print(f"  ! CNN unavailable ({exc}); falling back to the MLP.")
    return Classifier("mlp")


# ---------------------------------------------------------------------------
# The pipeline: segment, then crop the ORIGINAL photo, then preprocess
# ---------------------------------------------------------------------------

def read_number(image: np.ndarray, segment, classifier: Classifier,
                config: str = SELECTED_CONFIG, margin: float = 0.0) -> tuple[str, list]:
    """Recognise a whole number.

    Segmentation returns binary white-on-black crops, but the digits are
    classified from crops of the original photo. The preprocessing
    configurations expect the photo as it was taken, dark ink on light paper;
    feeding them an already-binarised crop would invert it twice under Otsu.
    """
    _, boxes = segment(image)
    height, width = image.shape[:2]
    digits = []
    for (x, y, w, h) in boxes:
        # Optional margin of paper around each box. Off by default: on closely
        # written numbers a margin pulls in ink from the neighbouring digit.
        m = int(margin * h)
        crop = image[max(0, y - m):min(height, y + h + m),
                     max(0, x - m):min(width, x + w + m)]
        digits.append(str(classifier.predict(crop, config)))
    return "".join(digits), boxes


# ---------------------------------------------------------------------------
# The two comparisons
# ---------------------------------------------------------------------------

def compare_preprocessing(samples, classifier) -> list[dict]:
    results = []
    for config in CONFIGS:
        predictions = [classifier.predict(img, config) for _, img, _ in samples]
        correct = sum(str(p) == label for p, (_, _, label) in zip(predictions, samples))
        results.append({
            "config": config,
            "images": len(samples),
            "correct": correct,
            "accuracy": round(correct / len(samples), 4),
            "selected": "yes" if config == SELECTED_CONFIG else "",
        })
    return sorted(results, key=lambda r: r["accuracy"], reverse=True)


def compare_segmentation(samples, methods: dict, classifier,
                         config: str = SELECTED_CONFIG) -> list[dict]:
    results = []
    for name, segment in methods.items():
        try:
            segment(samples[0][1])
        except (ImportError, FileNotFoundError) as exc:
            print(f"  ! {name} skipped ({type(exc).__name__}: {exc})")   # sliding_cnn without TF
            continue
        out_dir = REPORTS / "custom_demo" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        count_ok = exact_ok = 0
        for filename, image, label in samples:
            predicted, boxes = read_number(image, segment, classifier, config)
            count_ok += len(boxes) == len(label)
            exact = predicted == label
            exact_ok += exact
            tag = "ok" if exact else "MISMATCH"
            cv2.imwrite(str(out_dir / f"{tag}_{Path(filename).stem}_read_{predicted or 'none'}.png"),
                        annotate(image, boxes, labels=list(predicted)))
        results.append({
            "method": name,
            "images": len(samples),
            "count_accuracy": round(count_ok / len(samples), 4),
            "exact_number_accuracy": round(exact_ok / len(samples), 4),
            "selected": "yes" if name == SELECTED_METHOD else "",
        })
    return sorted(results, key=lambda r: r["exact_number_accuracy"], reverse=True)


def compare_pipeline(samples, methods: dict, classifier) -> list[dict]:
    """Every segmentation method against every preprocessing configuration.

    The two choices interact. Segmentation hands over tight crops, which a
    plain resize stretches and the 20x20 box does not; and thresholding that
    wins on single digits may lose on crops. So the pair is chosen together,
    on whole numbers read exactly right.
    """
    boxes_by_method = {}
    for name, segment in methods.items():
        try:
            boxes_by_method[name] = [segment(img)[1] for _, img, _ in samples]
        except (ImportError, FileNotFoundError):
            print(f"  ! {name} skipped (dependency missing)")
    results = []
    for name, all_boxes in boxes_by_method.items():
        for config in CONFIGS:
            exact = digit_ok = digit_total = 0
            for (_, image, label), boxes in zip(samples, all_boxes):
                read = "".join(str(classifier.predict(image[y:y + h, x:x + w], config))
                               for (x, y, w, h) in boxes)
                exact += read == label
                if len(read) == len(label):          # digits only comparable if counts match
                    digit_total += len(label)
                    digit_ok += sum(a == b for a, b in zip(read, label))
            results.append({
                "method": name,
                "config": config,
                "exact_number_accuracy": round(exact / len(samples), 4),
                "digit_accuracy_when_count_right": round(digit_ok / digit_total, 4) if digit_total else "",
                "selected": "yes" if (name == SELECTED_METHOD and config == SELECTED_CONFIG) else "",
            })
    return sorted(results, key=lambda r: r["exact_number_accuracy"], reverse=True)


def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate on real handwritten photos.")
    parser.add_argument("--model", choices=["cnn", "mlp"], default="cnn")
    parser.add_argument("--all-methods", action="store_true",
                        help="also run the model-based segmentation methods")
    parser.add_argument("--config", choices=list(CONFIGS), default=SELECTED_CONFIG,
                        help="preprocessing used on each digit crop in the number test")
    parser.add_argument("--all-configs", action="store_true",
                        help="whole-number test for every method x every configuration")
    args = parser.parse_args()

    print("Evaluation on real handwriting")
    print("=" * 66)
    digits = load_samples(SAMPLES / "digits", single_digit=True)
    numbers = load_samples(SAMPLES / "numbers", single_digit=False)
    print(f"Found {len(digits)} digit photos and {len(numbers)} number photos.")
    if not digits and not numbers:
        print(f"\nNothing to evaluate. Add photos to {SAMPLES.relative_to(ROOT)}/digits "
              f"and /numbers, named like 3_01.jpg and 94026_01.jpg.")
        return

    classifier = pick_classifier(args.model)
    print(f"Classifier: {classifier.kind}\n")
    REPORTS.mkdir(parents=True, exist_ok=True)

    if digits:
        rows = compare_preprocessing(digits, classifier)
        save_csv(rows, REPORTS / "custom_preprocessing.csv")
        print("Task 1 - digit accuracy per preprocessing configuration")
        for r in rows:
            mark = "  <- selected" if r["selected"] else ""
            print(f"  {r['config']:24s} {r['correct']:>3}/{r['images']:<3} {r['accuracy']:.1%}{mark}")
        print()

    if numbers:
        methods = dict(CLASSICAL)
        if args.all_methods:
            from segmentation_ml import ML_METHODS
            methods.update(ML_METHODS)
        rows = compare_segmentation(numbers, methods, classifier, args.config)
        save_csv(rows, REPORTS / "custom_segmentation.csv")
        print(f"Task 2 - segmentation, then whole-number recognition "
              f"(preprocessing: {args.config})")
        print(f"  {'method':22s} {'right count':>12s} {'exact number':>13s}")
        for r in rows:
            mark = "  <- selected" if r["selected"] else ""
            print(f"  {r['method']:22s} {r['count_accuracy']:>12.0%} "
                  f"{r['exact_number_accuracy']:>13.0%}{mark}")
        print()

        if args.all_configs:
            rows = compare_pipeline(numbers, methods, classifier)
            save_csv(rows, REPORTS / "custom_pipeline.csv")
            print("End to end - exact numbers read, every method x configuration (top 10)")
            print(f"  {'method':22s} {'config':24s} {'exact':>6s} {'digits*':>8s}")
            for r in rows[:10]:
                mark = "  <- selected" if r["selected"] else ""
                digits_acc = r["digit_accuracy_when_count_right"]
                digits_txt = f"{digits_acc:.0%}" if digits_acc != "" else "-"
                print(f"  {r['method']:22s} {r['config']:24s} "
                      f"{r['exact_number_accuracy']:>6.0%} {digits_txt:>8s}{mark}")
            print("  * digit accuracy, over numbers where the digit count was right")
            print("  Full table: reports/custom_pipeline.csv\n")

    print("Saved: reports/custom_preprocessing.csv, reports/custom_segmentation.csv, "
          "reports/custom_demo/")


if __name__ == "__main__":
    main()
