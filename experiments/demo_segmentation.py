"""
Task 2 evidence - segmentation demonstration and method comparison.
COS30018 Option B - Handwritten Number Recognition System
Owner: John (Person A)

Runs both segmentation techniques over the generated number images and reports,
for each method:
  - how often it found exactly the right number of digits
  - where it failed, with annotated images showing what it detected

"Correct digit count" is the right metric here because segmentation's job is to
find the digits; whether each one is then classified correctly is Task 3's
responsibility, not Task 2's.

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

from segmentation import METHODS, annotate, segment_digits    # noqa: E402

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
    """Count how often `method` detects the correct number of digits."""
    correct = 0
    failures = []

    if save_images:
        (DEMO_DIR / method).mkdir(parents=True, exist_ok=True)

    for row in samples:
        image = cv2.imread(str(GENERATED / row["filename"]))
        crops, boxes = segment_digits(image, method=method)

        expected = int(row["num_digits"])
        found = len(crops)
        ok = found == expected
        correct += ok

        if not ok:
            failures.append(f"{row['filename']} (expected {expected}, found {found})")

        if save_images:
            out = annotate(image, boxes)
            status = "ok" if ok else "MISMATCH"
            cv2.imwrite(str(DEMO_DIR / method / f"{status}_{row['filename']}"), out)

    return {
        "method": method,
        "images_tested": len(samples),
        "correct_digit_count": correct,
        "accuracy": round(correct / len(samples), 4),
        "failures": "; ".join(failures[:5]) if failures else "none",
    }


def main() -> None:
    print("Task 2 - segmentation method comparison")
    print("=" * 62)

    samples = load_ground_truth()
    print(f"Testing on {len(samples)} generated number images.\n")

    results = []
    for method in METHODS:
        row = evaluate_method(method, samples)
        results.append(row)
        print(f"  {method:22s} correct digit count on "
              f"{row['correct_digit_count']}/{row['images_tested']} images "
              f"({row['accuracy']:.1%})")
        if row["failures"] != "none":
            print(f"    failures: {row['failures']}")

    REPORTS.mkdir(parents=True, exist_ok=True)
    csv_path = REPORTS / "segmentation_comparison.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 62)
    print(f"Saved: {csv_path.relative_to(ROOT)}")
    print(f"Saved: annotated images in {DEMO_DIR.relative_to(ROOT)}/")
    print("\nFiles prefixed MISMATCH_ are the failure cases to discuss in the")
    print("report's critical analysis section.")


if __name__ == "__main__":
    main()
