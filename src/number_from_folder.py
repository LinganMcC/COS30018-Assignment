"""Image acquisition, mode 1: build the image of a number from a folder of single-digit images.

The specification asks for two ways to acquire the input image: loading an
image file, and "the automatic creation of the image of a number from a folder
of images of individual digits". This module is the second one; the GUI calls
it from *Input -> Generate from folder*.

Two folder layouts are accepted, and may be mixed:

    by subfolder   0/  1/ ... 9/  each holding images of that digit
                   (MNIST as written by src/prepare_data.py into data/raw/)
    by file name   the label first, then "_":  3_01.jpg, 7_photo2.png
                   (the team's photos in data/custom_samples/digits/)

Digit images may be white ink on black (MNIST) or dark ink on paper (photos);
each is cropped to its ink and scaled to the same height, so MNIST digits and
photographed digits can be combined. Operators for an expression are drawn
(generate_number.draw_operator), since the folder holds digits only.

    python src/number_from_folder.py data/custom_samples/digits --text 4096
    python src/number_from_folder.py data/raw --digits 5 --out number.png
    python src/number_from_folder.py data/custom_samples/digits --expression --text "(3+4)*5"
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from generate_number import draw_operator, random_expression

IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp")
DIGITS = "0123456789"
OPERATORS = "+-*/()"
# other ways of typing an operator -> the label the models use
ALIASES = {"x": "*", "X": "*", "×": "*", "÷": "/", "−": "-", "–": "-"}

DIGIT_HEIGHT = 56     # every digit is scaled so its ink is this tall
GAP = 18              # pixels of paper between neighbouring symbols
MARGIN = 30           # paper around the whole number
JITTER = 3            # random vertical offset of each symbol, like handwriting


# ---------------------------------------------------------------- the folder

def load_folder(folder: str | Path) -> Dict[str, List[Path]]:
    """Digit -> its image files, from subfolders 0/-9/ and/or label-first file names."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    pool: Dict[str, List[Path]] = {d: [] for d in DIGITS}
    for d in DIGITS:
        sub = folder / d
        if sub.is_dir():
            pool[d] += sorted(p for p in sub.iterdir() if p.suffix.lower() in IMAGE_TYPES)
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_TYPES:
            label = p.stem.split("_")[0]
            if label in pool:
                pool[label].append(p)
    pool = {d: files for d, files in pool.items() if files}
    if not pool:
        raise FileNotFoundError(
            f"No digit images in {folder}. Use subfolders 0/ to 9/, or name files "
            "with the digit first, e.g. 3_01.jpg.")
    return pool


# ---------------------------------------------------------------- one symbol

def ink_tile(gray: np.ndarray) -> np.ndarray:
    """A digit image -> white ink on black, cropped to the digit (any input polarity).

    Photos are cut from sheets, so besides the digit they can hold a ruled line
    or the edge of a neighbouring digit. Ruled lines are removed first
    (page_cleanup.remove_ruled_lines); then only the largest stroke and the
    strokes that belong with it are kept: a piece touching the edge of the
    image is kept only if it lies within the main stroke's columns.
    """
    if gray.mean() > 127:                       # dark ink on paper: a photo
        from page_cleanup import remove_ruled_lines
        gray = cv2.cvtColor(remove_ruled_lines(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)),
                            cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    t, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (blur > t).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise ValueError("no ink found")
    H, W = mask.shape
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mx, my, mw, mh, marea = stats[main]
    keep = [main]
    for i in range(1, n):
        if i == main:
            continue
        x, y, w, h, area = stats[i]
        if area < 0.05 * marea:                                  # dust
            continue
        at_edge = x == 0 or y == 0 or x + w >= W or y + h >= H
        overlap = min(x + w, mx + mw) - max(x, mx)               # shared columns
        if at_edge and overlap < 0.5 * min(w, mw):               # a cut-off neighbour
            continue
        if overlap < -0.25 * mw:                                 # far to one side
            continue
        keep.append(i)
    ink = np.isin(labels, keep)
    ys, xs = np.nonzero(ink)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    tile = np.where(ink, gray, 0)[y0:y1, x0:x1]
    # stretch to the full 0-255 range so photos and MNIST look alike
    return cv2.normalize(tile, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def scale_to_height(tile: np.ndarray, height: int) -> np.ndarray:
    h, w = tile.shape
    width = max(1, int(round(w * height / h)))
    return cv2.resize(tile, (width, height), interpolation=cv2.INTER_AREA
                      if h > height else cv2.INTER_CUBIC)


def stroke_width(mask: np.ndarray) -> float:
    """Typical stroke width in pixels: ink area / stroke length (skeleton pixels)."""
    from skimage.morphology import skeletonize
    length = skeletonize(mask > 0).sum()
    return float((mask > 0).sum() / max(1, length))


def normalise_digit(tile: np.ndarray, height: int = DIGIT_HEIGHT) -> np.ndarray:
    """Ink tile -> `height` px tall with an MNIST-like stroke (about height / 10).

    A photographed digit is several hundred pixels tall with a thin pen line;
    shrunk to `height` its strokes become faint hairlines that thresholding
    breaks apart. So the ink is scaled as a mask, thickened to the target
    width, and lightly smoothed - MNIST and photo digits end up alike.
    """
    small = scale_to_height(tile, height)
    mask = (small > 0.25 * small.max()).astype(np.uint8)
    target = height / 10
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(6):                       # each dilation adds about 2 px
        if stroke_width(mask) >= target - 1:
            break
        mask = cv2.dilate(mask, kernel)
    out = cv2.GaussianBlur(mask.astype(np.float32) * 255, (3, 3), 0)
    return out.astype(np.uint8)


def operator_tile(symbol: str, height: int = DIGIT_HEIGHT) -> np.ndarray:
    """A drawn operator at the right size next to digits of ink height `height`."""
    tile = draw_operator(symbol)            # 28x28, sized like an MNIST digit (ink ~20 px)
    if symbol in "()":                      # generate_number.draw_operator draws them mirrored
        tile = cv2.flip(tile, 1)
    size = int(round(28 * height / 20))
    tile = cv2.resize(tile, (size, size), interpolation=cv2.INTER_NEAREST)
    tile = cv2.dilate(tile, np.ones((3, 3), np.uint8))          # pen-like stroke width
    cols = np.flatnonzero(tile.max(axis=0))
    return tile[:, cols.min():cols.max() + 1]                    # keep height, trim width


# ---------------------------------------------------------------- the number

def parse(text: str, expression: bool) -> List[str]:
    """User text -> list of symbols; raises ValueError on anything that cannot be drawn."""
    symbols = [ALIASES.get(c, c) for c in text if not c.isspace()]
    allowed = DIGITS + (OPERATORS if expression else "")
    bad = sorted({c for c in symbols if c not in allowed})
    if bad:
        kind = "digits and + - * / ( )" if expression else "digits"
        raise ValueError(f"Cannot draw {' '.join(bad)} - type {kind} only.")
    if not symbols:
        raise ValueError("Nothing to draw.")
    return symbols


def compose(symbols: Sequence[str], pool: Dict[str, List[Path]],
            rng: Optional[random.Random] = None) -> np.ndarray:
    """Symbols -> one grayscale image, dark ink on white paper (like a photo)."""
    rng = rng or random.Random()
    missing = sorted({s for s in symbols if s in DIGITS and s not in pool})
    if missing:
        raise ValueError(f"The folder has no images of digit {', '.join(missing)}.")
    tiles = []
    for s in symbols:
        if s in DIGITS:
            path = rng.choice(pool[s])
            gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError(f"Cannot read {path.name}")
            tiles.append(normalise_digit(ink_tile(gray)))
        else:
            tiles.append(operator_tile(s))
    band = max(t.shape[0] for t in tiles) + 2 * JITTER
    width = sum(t.shape[1] for t in tiles) + GAP * (len(tiles) - 1) + 2 * MARGIN
    canvas = np.zeros((band + 2 * MARGIN, width), np.uint8)
    x = MARGIN
    for t in tiles:
        h, w = t.shape
        y = MARGIN + (band - h) // 2 + rng.randint(-JITTER, JITTER)   # centred on one line
        canvas[y:y + h, x:x + w] = np.maximum(canvas[y:y + h, x:x + w], t)
        x += w + GAP
    return 255 - canvas


def random_symbols(pool: Dict[str, List[Path]], n_digits: int, expression: bool,
                   rng: random.Random) -> List[str]:
    """A random number of n_digits (no leading zero), or a random expression, from available digits."""
    have = [d for d in DIGITS if d in pool]
    for _ in range(200):
        if expression:
            symbols = random_expression(rng)
        else:
            first = [d for d in have if d != "0"] or have
            symbols = [rng.choice(first)] + [rng.choice(have) for _ in range(n_digits - 1)]
        if all(s not in DIGITS or s in pool for s in symbols):
            return symbols
    raise ValueError("The folder does not have enough different digits for a random expression.")


def make_number(folder: str | Path, text: str = "", n_digits: int = 4,
                expression: bool = False, seed: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """The GUI entry point: (image, the text it shows). Empty `text` -> random."""
    rng = random.Random(seed)
    pool = load_folder(folder)
    symbols = parse(text, expression) if text.strip() else \
        random_symbols(pool, max(1, int(n_digits)), expression, rng)
    return compose(symbols, pool, rng), "".join(symbols)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Build a number image from a folder of digit images.")
    ap.add_argument("folder")
    ap.add_argument("--text", default="", help="the number (or expression) to build; empty = random")
    ap.add_argument("--digits", type=int, default=4, help="length of a random number")
    ap.add_argument("--expression", action="store_true", help="build an arithmetic expression")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default="generated_number.png")
    args = ap.parse_args(argv)
    image, text = make_number(args.folder, args.text, args.digits, args.expression, args.seed)
    cv2.imwrite(args.out, image)
    print(f"'{text}' -> {args.out}")


if __name__ == "__main__":
    main()
