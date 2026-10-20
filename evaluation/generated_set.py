"""Task 4 test set of generated multi-digit numbers, built from digits no model was trained on.

src/generate_number.py builds numbers from data/raw/, which prepare_data.py
fills with digits from the MNIST *training* split - the images every model
learned from. Scoring a model on those would overstate it. This builds the
same kind of numbers (same layout code, same three spacings) from the MNIST
*test* split instead, which only the final test scores ever touched.

    wide      14 px between digits    - easy, digits clearly apart
    tight     -6 px (margins overlap) - digits close but strokes separate
    touching -14 px                    - strokes of neighbours actually join

Run from the repository root (once, ~10 s):
    python evaluation/generated_set.py               # 600 numbers, 200 per spacing
    python evaluation/generated_set.py --count 300

Output (not committed - rebuilt identically from the seed):
    data/generated_test/digits/<d>/*.png     the MNIST test digits used as tiles
    data/generated_test/numbers/*.png        the numbers
    data/generated_test/labels.csv           filename, label, num_digits, difficulty
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

from common import ROOT, load_mnist_splits

OUT = ROOT / "data" / "generated_test"
SEED = 2026
TILES_PER_DIGIT = 300


def export_test_digits(out: Path = OUT, per_digit: int = TILES_PER_DIGIT) -> dict:
    """Write MNIST test-split digits as PNG tiles (white on black, like data/raw/)."""
    _, _, (x_te, y_te) = load_mnist_splits()
    pool = {}
    for d in range(10):
        folder = out / "digits" / str(d)
        folder.mkdir(parents=True, exist_ok=True)
        paths = []
        for n, i in enumerate(np.where(y_te == d)[0][:per_digit]):
            p = folder / f"{d}_{n:03d}.png"
            if not p.exists():
                cv2.imwrite(str(p), (x_te[i, ..., 0] * 255).astype(np.uint8))
            paths.append(p)
        pool[d] = paths
    return pool


def build(count: int = 600, out: Path = OUT, seed: int = SEED) -> Path:
    from generate_number import DIFFICULTY, compose_number
    pool = export_test_digits(out)
    random.seed(seed)
    numbers = out / "numbers"
    numbers.mkdir(parents=True, exist_ok=True)
    levels = list(DIFFICULTY)
    rows = []
    for i in range(count):
        level = levels[i % len(levels)]
        digits = [random.randint(0, 9) for _ in range(random.randint(2, 5))]
        image = compose_number(digits, pool, spacing=DIFFICULTY[level])
        name = f"number_{i:03d}.png"
        cv2.imwrite(str(numbers / name), image)
        rows.append({"filename": name, "label": "".join(map(str, digits)),
                     "num_digits": len(digits), "difficulty": level})
    labels = out / "labels.csv"
    with open(labels, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{count} numbers from MNIST test digits -> {numbers.relative_to(ROOT)}/ "
          f"({', '.join(f'{lv} {count // len(levels)}' for lv in levels)})")
    return labels


def load(out: Path = OUT):
    """(filename, image BGR, label, difficulty) for every generated test number."""
    labels = out / "labels.csv"
    if not labels.exists():
        raise FileNotFoundError(f"{labels} not found - run: python evaluation/generated_set.py")
    rows = []
    for r in csv.DictReader(open(labels)):
        img = cv2.imread(str(out / "numbers" / r["filename"]))
        rows.append((r["filename"], img, r["label"], r["difficulty"]))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=600)
    build(ap.parse_args().count)
