"""Task 2 - compare both segmentation methods on the generated number images.
Metric: how often the correct number of digits is found.

Run:
    python experiments/demo_segmentation.py

Outputs:
    reports/segmentation_comparison.csv
    reports/segmentation_demo/*.png
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation import METHODS as CLASSICAL, annotate, segment_digits
from segmentation_ml import ML_METHODS

# Classical first, then the model-based ones, so the table reads as two families.
METHODS = {**CLASSICAL, **ML_METHODS}

GENERATED = ROOT / "data" / "generated"
REPORTS = ROOT / "reports"
DEMO_DIR = REPORTS / "segmentation_demo"


def load_ground_truth() -> list[dict]:
    labels_csv = GENERATED / "labels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(
            f"{labels_csv} not found. Run 'python src/generate_number.py' first."
        )
    with open(labels_csv, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def evaluate_method(method: str, samples: list[dict], save_images: bool = True) -> dict:
    """Count how often `method` finds the correct number of digits."""
    correct = 0
    failures = []
    failed_files = set()

    if save_images:
        (DEMO_DIR / method).mkdir(parents=True, exist_ok=True)

    for row in samples:
        image = cv2.imread(str(GENERATED / row["filename"]))
        crops, boxes = METHODS[method](image)

        expected = int(row["num_digits"])
        found = len(crops)
        ok = found == expected
        correct += ok

        if not ok:
            failures.append(f"{row['filename']} (expected {expected}, found {found})")
            failed_files.add(row["filename"])

        if save_images:
            out = annotate(image, boxes)
            status = "ok" if ok else "MISMATCH"
            cv2.imwrite(str(DEMO_DIR / method / f"{status}_{row['filename']}"), out)

    row = {
        "method": method,
        "images_tested": len(samples),
        "correct_digit_count": correct,
        "accuracy": round(correct / len(samples), 4),
    }

    # Per-difficulty accuracy. This is the column that actually separates the
    # methods: on well-spaced digits they all score the same.
    for level in sorted({s.get("difficulty", "all") for s in samples}):
        subset = [s for s in samples if s.get("difficulty", "all") == level]
        hits = sum(1 for s in subset if s["filename"] not in failed_files)
        row[f"acc_{level}"] = round(hits / len(subset), 4) if subset else ""

    row["failures"] = "; ".join(failures[:5]) if failures else "none"
    return row


def main() -> None:
    print("Task 2 - segmentation method comparison")
    print("=" * 72)

    samples = load_ground_truth()
    levels = sorted({s.get("difficulty", "all") for s in samples})
    print(f"Testing {len(METHODS)} methods on {len(samples)} generated number images.")
    print(f"Difficulty levels present: {', '.join(levels)}\n")

    results = []
    header = f"  {'method':22s} overall  " + "  ".join(f"{lv:>9s}" for lv in levels)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for method in METHODS:
        try:
            row = evaluate_method(method, samples)
        except (ImportError, FileNotFoundError) as exc:
            # sliding_cnn needs TensorFlow and Thien's checkpoint. Skip rather
            # than fail, so the rest of the comparison still runs.
            print(f"  {method:22s} skipped ({type(exc).__name__}: {exc})")
            continue
        results.append(row)
        cells = "  ".join(f"{row[f'acc_{lv}']:>9.0%}" for lv in levels)
        print(f"  {method:22s} {row['accuracy']:>6.0%}   {cells}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "segmentation_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    best = max(results, key=lambda r: r["accuracy"])
    print("\n" + "=" * 72)
    print(f"Best overall: {best['method']} ({best['accuracy']:.0%})")
    print(f"Saved: {csv_path.relative_to(ROOT)}")
    print(f"Saved: annotated images in {DEMO_DIR.relative_to(ROOT)}/")
    print("\nFiles prefixed MISMATCH_ are the failure cases to discuss in the report.")


if __name__ == "__main__":
    main()
