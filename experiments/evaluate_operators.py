"""Task 2 on arithmetic expressions - does segmentation find the operators?

Every Task 2 figure so far was measured on digits only. The extension needs the
same methods to return one box per symbol for '+ - * / ( )' as well, and the
digit filter drops all of those: a minus sign is 5% of the image height against
a 15% gate, an asterisk 13%, and a division sign arrives as three pieces.

This script measures the fix (segmentation.keep_operators) two ways:

  1. on expression images, how many are split into exactly the right number of
     symbols, per method, with the flag off and on;
  2. on the 27 real photographed numbers, what turning the flag on costs - it
     admits short shapes, and a speck of dirt is also a short shape.

Both numbers are needed to justify the flag being opt-in rather than always on.

Run:
    python src/generate_number.py --expressions --count 40     # first, once
    python experiments/evaluate_operators.py

Outputs:
    reports/operator_segmentation.csv   symbol-count accuracy per method
    reports/operator_cost.csv           what the flag costs on digit photos
    reports/operator_segmentation.png   worked example, boxes drawn
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation import METHODS, SELECTED_METHOD, annotate

EXPRESSIONS = ROOT / "data" / "generated_expressions"
NUMBERS = ROOT / "data" / "custom_samples" / "numbers"
REPORTS = ROOT / "reports"
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}


def load_expressions() -> list[tuple[str, np.ndarray, str]]:
    """Read the generated expressions and their labels.csv ground truth."""
    labels_file = EXPRESSIONS / "labels.csv"
    if not labels_file.exists():
        return []
    samples = []
    with open(labels_file, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            image = cv2.imread(str(EXPRESSIONS / row["filename"]))
            if image is not None:
                samples.append((row["filename"], image, row["label"]))
    return samples


def load_numbers() -> list[tuple[str, np.ndarray, str]]:
    """Read the real photographed numbers; the label is the filename before '_'."""
    if not NUMBERS.exists():
        return []
    samples = []
    for path in sorted(NUMBERS.iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue
        label = path.stem.split("_")[0]
        image = cv2.imread(str(path))
        if label.isdigit() and image is not None:
            samples.append((path.name, image, label))
    return samples


def count_accuracy(samples, segment, **kwargs) -> float:
    """Share of images split into exactly as many boxes as the label has symbols."""
    if not samples:
        return 0.0
    right = sum(len(segment(image, **kwargs)[1]) == len(label)
                for _, image, label in samples)
    return right / len(samples)


def save_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_example(samples) -> None:
    """Draw one expression twice, flag off and flag on, as the report figure."""
    # The longest expression shows the most dropped operators.
    _, image, label = max(samples, key=lambda s: len(s[2]))
    segment = METHODS[SELECTED_METHOD]

    panels = []
    for caption, kwargs in [("keep_operators=False", {}),
                            ("keep_operators=True", {"keep_operators": True})]:
        _, boxes = segment(image, **kwargs)
        panel = annotate(image, boxes)
        cv2.putText(panel, f"{caption}  -  {len(boxes)}/{len(label)} symbols",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        panels.append(panel)

    cv2.imwrite(str(REPORTS / "operator_segmentation.png"), cv2.vconcat(panels))
    print(f"  example written for '{label}'")


def main() -> None:
    expressions = load_expressions()
    numbers = load_numbers()
    REPORTS.mkdir(parents=True, exist_ok=True)

    if not expressions:
        print(f"No expression images in {EXPRESSIONS.relative_to(ROOT)}/.\n"
              "Run: python src/generate_number.py --expressions --count 40")
        return

    print(f"Expressions: {len(expressions)}   real numbers: {len(numbers)}\n")

    print("Task 2 on expressions - images split into the right number of symbols")
    print(f"  {'method':24s} {'flag off':>9s} {'flag on':>8s}")
    rows = []
    for name, segment in METHODS.items():
        off = count_accuracy(expressions, segment)
        on = count_accuracy(expressions, segment, keep_operators=True)
        rows.append({"method": name,
                     "images": len(expressions),
                     "symbol_count_accuracy_off": round(off, 4),
                     "symbol_count_accuracy_on": round(on, 4),
                     "selected": "yes" if name == SELECTED_METHOD else ""})
        mark = "  <- selected" if name == SELECTED_METHOD else ""
        print(f"  {name:24s} {off:>9.0%} {on:>8.0%}{mark}")
    save_csv(rows, REPORTS / "operator_segmentation.csv")

    if numbers:
        print(f"\nWhat the flag costs on {len(numbers)} real digit photos")
        print(f"  {'method':24s} {'flag off':>9s} {'flag on':>8s}")
        rows = []
        for name, segment in METHODS.items():
            off = count_accuracy(numbers, segment)
            on = count_accuracy(numbers, segment, keep_operators=True)
            rows.append({"method": name,
                         "images": len(numbers),
                         "digit_count_accuracy_off": round(off, 4),
                         "digit_count_accuracy_on": round(on, 4),
                         "selected": "yes" if name == SELECTED_METHOD else ""})
            mark = "  <- selected" if name == SELECTED_METHOD else ""
            print(f"  {name:24s} {off:>9.0%} {on:>8.0%}{mark}")
        save_csv(rows, REPORTS / "operator_cost.csv")

    save_example(expressions)
    print("\nSaved: reports/operator_segmentation.csv, reports/operator_cost.csv, "
          "reports/operator_segmentation.png")


if __name__ == "__main__":
    main()
