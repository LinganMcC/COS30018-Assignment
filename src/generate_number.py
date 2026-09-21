"""Create multi-digit number images from single-digit images, with ground-truth labels.

Run:
    python src/generate_number.py                 # 20 numbers, 2-5 digits each
    python src/generate_number.py --count 50      # more samples
    python src/generate_number.py --digits 4      # fixed length
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "generated"


def load_digit_pool(raw_dir: Path = RAW_DIR) -> dict[int, list[Path]]:
    if not raw_dir.exists():
        raise FileNotFoundError(
            f"{raw_dir} does not exist. Run 'python src/prepare_data.py' first."
        )
    pool = {}
    for digit in range(10):
        files = sorted((raw_dir / str(digit)).glob("*.png"))
        if files:
            pool[digit] = files
    if not pool:
        raise FileNotFoundError(f"No digit images found under {raw_dir}.")
    return pool


def compose_number(digits: list[int], pool: dict[int, list[Path]],
                   spacing: int = 14, margin: int = 20,
                   jitter: int = 4, invert: bool = True) -> np.ndarray:
    """Place digit images side by side on one canvas (dark digits on white if invert=True)."""
    tiles = []
    for d in digits:
        img = cv2.imread(str(random.choice(pool[d])), cv2.IMREAD_GRAYSCALE)
        tiles.append(img)

    height = max(t.shape[0] for t in tiles) + 2 * margin + 2 * jitter
    width = sum(t.shape[1] for t in tiles) + spacing * (len(tiles) - 1) + 2 * margin
    canvas = np.zeros((height, width), dtype=np.uint8)

    x = margin
    for tile in tiles:
        h, w = tile.shape
        y = margin + jitter + random.randint(-jitter, jitter)
        # np.maximum so overlapping tiles merge instead of overwriting each other
        canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], tile)
        x += w + spacing

    return cv2.bitwise_not(canvas) if invert else canvas


def generate_dataset(count: int = 20, digits: int | None = None,
                     min_digits: int = 2, max_digits: int = 5,
                     out_dir: Path = OUT_DIR, seed: int | None = 42) -> Path:
    """Generate `count` number images and a labels.csv with the ground truth."""
    if seed is not None:
        random.seed(seed)

    pool = load_digit_pool()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(count):
        length = digits if digits else random.randint(min_digits, max_digits)
        number = [random.randint(0, 9) for _ in range(length)]
        image = compose_number(number, pool)

        filename = f"number_{i:03d}.png"
        cv2.imwrite(str(out_dir / filename), image)
        rows.append({"filename": filename,
                     "label": "".join(str(d) for d in number),
                     "num_digits": length})

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["filename", "label", "num_digits"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated {count} number images in {out_dir.relative_to(ROOT)}/")
    print(f"Ground truth written to {labels_path.relative_to(ROOT)}")
    print("Sample:", ", ".join(r["label"] for r in rows[:8]), "...")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-digit number images.")
    parser.add_argument("--count", type=int, default=20, help="how many images to create")
    parser.add_argument("--digits", type=int, default=None,
                        help="fixed number of digits (default: random 2-5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    generate_dataset(count=args.count, digits=args.digits, seed=args.seed)


if __name__ == "__main__":
    main()
