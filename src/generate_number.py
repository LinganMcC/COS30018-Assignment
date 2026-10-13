"""Create multi-digit number images from single-digit images, with ground-truth labels.

Run:
    python src/generate_number.py                 # 20 numbers, 2-5 digits each
    python src/generate_number.py --count 50      # more samples
    python src/generate_number.py --digits 4      # fixed length
    python src/generate_number.py --expressions   # arithmetic expressions instead
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
EXPRESSION_DIR = ROOT / "data" / "generated_expressions"


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


# Spacing between digit tiles, in pixels. Negative means the tiles overlap.
# Segmentation gets harder from left to right, which is the whole point: a test
# set where every method scores 100% tells us nothing about which to choose.
# MNIST tiles carry about 4 blank pixels of margin on each side, so spacing has
# to go past about -10 before the ink of two digits actually merges. These three
# values were picked by measuring that: 'tight' eats the margins but leaves the
# strokes separate, 'touching' genuinely joins them into one blob.
DIFFICULTY = {
    "wide": 14,
    "tight": -6,
    "touching": -14,
}


def generate_dataset(count: int = 20, digits: int | None = None,
                     min_digits: int = 2, max_digits: int = 5,
                     out_dir: Path = OUT_DIR, seed: int | None = 42,
                     difficulty: str | None = None) -> Path:
    """Generate `count` number images and a labels.csv with the ground truth.

    difficulty picks a fixed spacing from DIFFICULTY. Left as None, the set is
    split evenly across all three so one run produces easy and hard cases.
    """
    if seed is not None:
        random.seed(seed)
    if difficulty is not None and difficulty not in DIFFICULTY:
        raise KeyError(f"Unknown difficulty '{difficulty}'. Available: {list(DIFFICULTY)}")

    pool = load_digit_pool()
    out_dir.mkdir(parents=True, exist_ok=True)

    levels = [difficulty] if difficulty else list(DIFFICULTY)

    rows = []
    for i in range(count):
        level = levels[i % len(levels)]
        length = digits if digits else random.randint(min_digits, max_digits)
        number = [random.randint(0, 9) for _ in range(length)]
        image = compose_number(number, pool, spacing=DIFFICULTY[level])

        filename = f"number_{i:03d}.png"
        cv2.imwrite(str(out_dir / filename), image)
        rows.append({"filename": filename,
                     "label": "".join(str(d) for d in number),
                     "num_digits": length,
                     "difficulty": level})

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "label", "num_digits", "difficulty"])
        writer.writeheader()
        writer.writerows(rows)

    counts = {lv: sum(r["difficulty"] == lv for r in rows) for lv in levels}
    print(f"Generated {count} number images in {out_dir.relative_to(ROOT)}/")
    print("Difficulty split:", ", ".join(f"{k} {v}" for k, v in counts.items()))
    print(f"Ground truth written to {labels_path.relative_to(ROOT)}")
    print("Sample:", ", ".join(r["label"] for r in rows[:8]), "...")
    return labels_path


# ---------------------------------------------------------------------------
# Arithmetic expressions, for the extension
# ---------------------------------------------------------------------------

OPERATORS = ["+", "-", "*", "/", "(", ")"]


def draw_operator(symbol: str, size: int = 28, thickness: int = 2) -> np.ndarray:
    """Draw one operator as a tile the same size as an MNIST digit, white on black.

    The team's own handwritten operator samples are not collected yet, so these
    are drawn. They are a stand-in for measuring segmentation, which only needs
    the shape and size to be right; they are not training data.

    '/' is drawn as the division sign the spec shows, bar plus two dots, which
    is the only symbol here made of more than one piece of ink.
    """
    tile = np.zeros((size, size), dtype=np.uint8)
    mid = size // 2
    arm = size // 3

    if symbol == "+":
        cv2.line(tile, (mid - arm, mid), (mid + arm, mid), 255, thickness)
        cv2.line(tile, (mid, mid - arm), (mid, mid + arm), 255, thickness)
    elif symbol == "-":
        cv2.line(tile, (mid - arm, mid), (mid + arm, mid), 255, thickness)
    elif symbol == "*":
        for dx, dy in [(arm, 0), (arm // 2, arm), (-arm // 2, arm)]:
            cv2.line(tile, (mid - dx, mid - dy), (mid + dx, mid + dy), 255, thickness)
    elif symbol == "/":
        cv2.line(tile, (mid - arm, mid), (mid + arm, mid), 255, thickness)
        cv2.circle(tile, (mid, mid - arm), max(1, thickness), 255, -1)
        cv2.circle(tile, (mid, mid + arm), max(1, thickness), 255, -1)
    elif symbol in "()":
        start, end = (300, 420) if symbol == "(" else (120, 240)
        cv2.ellipse(tile, (mid, mid), (arm, size // 2 - 2), 0, start, end, 255, thickness)
    else:
        raise KeyError(f"Unknown operator '{symbol}'. Available: {OPERATORS}")
    return tile


def compose_expression(symbols: list[str], pool: dict[int, list[Path]],
                       spacing: int = 14, margin: int = 20,
                       jitter: int = 4, invert: bool = True) -> np.ndarray:
    """Place digit images and drawn operators side by side on one canvas."""
    tiles = []
    for symbol in symbols:
        if symbol.isdigit():
            tiles.append(cv2.imread(str(random.choice(pool[int(symbol)])),
                                    cv2.IMREAD_GRAYSCALE))
        else:
            tiles.append(draw_operator(symbol))

    height = max(t.shape[0] for t in tiles) + 2 * margin + 2 * jitter
    width = sum(t.shape[1] for t in tiles) + spacing * (len(tiles) - 1) + 2 * margin
    canvas = np.zeros((height, width), dtype=np.uint8)

    x = margin
    for tile in tiles:
        h, w = tile.shape
        y = margin + jitter + random.randint(-jitter, jitter)
        canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], tile)
        x += w + spacing

    return cv2.bitwise_not(canvas) if invert else canvas


# Expression shapes, in the order the extension has to cope with them: no
# brackets, then one pair, then nested. {n} is a 1-2 digit number, {o} a binary
# operator. Taken from the spec's own example, 26 - 3 * (17 - 4).
EXPRESSION_PATTERNS = [
    "{n}{o}{n}",
    "{n}{o}{n}{o}{n}",
    "{n}{o}({n}{o}{n})",
    "({n}{o}{n}){o}{n}",
    "{n}{o}({n}{o}({n}{o}{n}))",
]


def random_expression(rng: random.Random) -> list[str]:
    """Build one expression and return it as a list of single symbols."""
    pattern = rng.choice(EXPRESSION_PATTERNS)
    text = ""
    i = 0
    while i < len(pattern):
        if pattern[i] == "{":
            kind = pattern[i + 1]
            text += (str(rng.randint(1, 99)) if kind == "n"
                     else rng.choice(["+", "-", "*", "/"]))
            i += 3
        else:
            text += pattern[i]
            i += 1
    return list(text)


def generate_expressions(count: int = 20, out_dir: Path = EXPRESSION_DIR,
                         seed: int | None = 42, spacing: int = 14) -> Path:
    """Generate labelled expression images for the extension's segmentation test."""
    rng = random.Random(seed)
    random.seed(seed)          # compose_expression picks digit tiles with random
    pool = load_digit_pool()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(count):
        symbols = random_expression(rng)
        image = compose_expression(symbols, pool, spacing=spacing)

        filename = f"expr_{i:03d}.png"
        cv2.imwrite(str(out_dir / filename), image)
        rows.append({"filename": filename,
                     "label": "".join(symbols),
                     "num_symbols": len(symbols),
                     "has_brackets": "yes" if "(" in symbols else "no"})

    labels_path = out_dir / "labels.csv"
    with open(labels_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["filename", "label", "num_symbols", "has_brackets"])
        writer.writeheader()
        writer.writerows(rows)

    bracketed = sum(r["has_brackets"] == "yes" for r in rows)
    print(f"Generated {count} expression images in {out_dir.relative_to(ROOT)}/")
    print(f"With brackets: {bracketed}, without: {count - bracketed}")
    print(f"Ground truth written to {labels_path.relative_to(ROOT)}")
    print("Sample:", ", ".join(r["label"] for r in rows[:6]), "...")
    return labels_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-digit number images.")
    parser.add_argument("--count", type=int, default=20, help="how many images to create")
    parser.add_argument("--digits", type=int, default=None,
                        help="fixed number of digits (default: random 2-5)")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--difficulty", choices=list(DIFFICULTY), default=None,
                        help="fix the digit spacing (default: mix all three)")
    parser.add_argument("--expressions", action="store_true",
                        help="generate arithmetic expressions for the extension")
    args = parser.parse_args()

    if args.expressions:
        generate_expressions(count=args.count, seed=args.seed)
    else:
        generate_dataset(count=args.count, digits=args.digits, seed=args.seed,
                         difficulty=args.difficulty)


if __name__ == "__main__":
    main()
