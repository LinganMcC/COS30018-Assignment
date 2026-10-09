"""Task 2 - regenerate the two segmentation figures used in the report.

The figures used to be made by hand in a one-off snippet, and one of them went
out of date when the code changed underneath it. Making them from a script
means they always show what the current code actually does.

Run (after generate_number.py and digit_classifier.py):
    python experiments/make_segmentation_figures.py

Outputs:
    reports/segmentation_methods.png    four classical methods, one number per difficulty
    reports/segmentation_touching.png   every method on the same touching-digit number

The example images are picked deterministically (the first number with at
least three digits at each difficulty), not chosen because they look good.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from segmentation import METHODS as CLASSICAL, annotate
from segmentation_ml import ML_METHODS

GENERATED = ROOT / "data" / "generated"
REPORTS = ROOT / "reports"

OK = "#1a7f37"
MISS = "#c00000"


def load_labels() -> list[dict]:
    path = GENERATED / "labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run 'python src/generate_number.py' first.")
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def first_example(rows: list[dict], difficulty: str, min_digits: int = 3) -> dict:
    for row in rows:
        if row.get("difficulty") == difficulty and int(row["num_digits"]) >= min_digits:
            return row
    raise LookupError(f"No '{difficulty}' number with {min_digits}+ digits in labels.csv")


def draw(ax, image, method, fn, expected: int) -> None:
    crops, boxes = fn(image)
    ax.imshow(cv2.cvtColor(annotate(image, boxes), cv2.COLOR_BGR2RGB))
    ax.set_xticks([])
    ax.set_yticks([])
    found = len(crops)
    ax.set_xlabel(f"found {found}/{expected}", fontsize=8,
                  color=OK if found == expected else MISS)


def classical_figure(rows: list[dict], plt) -> Path:
    levels = ["wide", "tight", "touching"]
    fig, axes = plt.subplots(len(levels), len(CLASSICAL), figsize=(11, 6.2), dpi=140)
    for r, level in enumerate(levels):
        row = first_example(rows, level)
        image = cv2.imread(str(GENERATED / row["filename"]))
        for c, (name, fn) in enumerate(CLASSICAL.items()):
            draw(axes[r, c], image, name, fn, int(row["num_digits"]))
            if r == 0:
                axes[r, c].set_title(name.replace("_", "\n"), fontsize=9)
        axes[r, 0].set_ylabel(f"{level}\n(label {row['label']})", fontsize=8)
    fig.suptitle("Task 2 - four classical segmentation techniques", fontsize=11)
    fig.tight_layout()
    out = REPORTS / "segmentation_methods.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out


def touching_figure(rows: list[dict], plt) -> tuple[Path, list[str]]:
    row = first_example(rows, "touching")
    image = cv2.imread(str(GENERATED / row["filename"]))
    expected = int(row["num_digits"])

    methods = {**CLASSICAL, **ML_METHODS}
    usable, skipped = {}, []
    for name, fn in methods.items():
        try:
            fn(image)
            usable[name] = fn
        except (ImportError, FileNotFoundError):
            skipped.append(name)          # sliding_cnn without TensorFlow

    cols = 4
    rows_n = -(-len(usable) // cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 2.4 * rows_n), dpi=140)
    grid = list(axes.flat)
    for ax, (name, fn) in zip(grid, usable.items()):
        draw(ax, image, name, fn, expected)
        family = "classical" if name in CLASSICAL else "model-based"
        ax.set_title(f"{name}\n({family})", fontsize=8)
    for ax in grid[len(usable):]:         # blank cells at the end of the grid
        ax.axis("off")

    fig.suptitle(f"Task 2 - every technique on one touching-digit number "
                 f"(ground truth {row['label']})", fontsize=11)
    fig.tight_layout()
    out = REPORTS / "segmentation_touching.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return out, skipped


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    REPORTS.mkdir(parents=True, exist_ok=True)
    rows = load_labels()

    print(f"Saved: {classical_figure(rows, plt).relative_to(ROOT)}")
    out, skipped = touching_figure(rows, plt)
    print(f"Saved: {out.relative_to(ROOT)}")
    if skipped:
        print(f"Skipped (dependency missing): {', '.join(skipped)}")


if __name__ == "__main__":
    main()
