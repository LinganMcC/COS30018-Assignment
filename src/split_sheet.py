"""Cut a photographed sheet of handwriting into one image per number.

Writing every test number on its own sheet and photographing each one is slow,
so the real handwritten set is written as rows on one sheet and split here.
The labels are given row by row, left to right, exactly as written:

    python src/split_sheet.py sheet.jpg \\
        --rows "0 1 2 3 4 5 6 7 8 9" "17 49 38 60 25" "104 573 896 241 730" \\
        --roi 60 1000 4100 3900

Single-digit labels go to data/custom_samples/digits/, longer ones to
data/custom_samples/numbers/, so a row like "0 1 2 ... 9 10" is fine. A number is marked as
touching (label_t1.jpg instead of label_01.jpg) when it has fewer separate
ink blobs than digits - measured, not taken from what was intended, because
digits written "to touch" do not always end up touching.

The crops come from the original photo with a margin of paper around them,
so the preprocessing is tested on what the camera actually captured.

Also writes data/custom_samples/split_preview.jpg: every crop with its label,
to check by eye before running any experiment on them.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "custom_samples"


def ink_mask(image: np.ndarray) -> np.ndarray:
    """Dark pen strokes, found against the local paper brightness.

    A black-hat transform subtracts the image from its own closing, which keeps
    thin dark strokes and removes slow changes such as shadows and the gradient
    across the sheet. A single global threshold cannot do that on a phone photo.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    strokes = cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)
    mask = ((strokes > 30) * 255).astype(np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def find_components(mask: np.ndarray, min_area: int = 120) -> list[dict]:
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    return [dict(x=int(stats[i, 0]), y=int(stats[i, 1]), w=int(stats[i, 2]),
                 h=int(stats[i, 3]), area=int(stats[i, 4]), cy=float(centres[i, 1]))
            for i in range(1, n) if stats[i, 4] >= min_area]


def split_rows(components: list[dict], row_gap: int) -> list[list[dict]]:
    components = sorted(components, key=lambda c: c["cy"])
    rows: list[list[dict]] = []
    for c in components:
        if rows and abs(c["cy"] - np.median([d["cy"] for d in rows[-1]])) < row_gap:
            rows[-1].append(c)
        else:
            rows.append([c])
    return rows


def split_groups(row: list[dict], gap: int) -> list[list[dict]]:
    """Merge blobs into one number while the horizontal gap stays under `gap`."""
    row = sorted(row, key=lambda c: c["x"])
    groups = [[row[0]]]
    for c in row[1:]:
        right = max(d["x"] + d["w"] for d in groups[-1])
        if c["x"] - right < gap:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def bounding(group: list[dict]) -> tuple[int, int, int, int]:
    x0 = min(c["x"] for c in group)
    y0 = min(c["y"] for c in group)
    x1 = max(c["x"] + c["w"] for c in group)
    y1 = max(c["y"] + c["h"] for c in group)
    return x0, y0, x1 - x0, y1 - y0


def split_box(mask: np.ndarray, box: tuple[int, int, int, int]):
    """Cut one box in two at its emptiest column, searching the middle 40%.

    Used when two neighbouring digits are joined by a long stroke - a 7 whose
    bar runs on into the next digit - and so arrive as one ink group.
    """
    x, y, w, h = box
    profile = (mask[y:y + h, x:x + w] > 0).sum(axis=0)
    lo, hi = int(0.3 * w), int(0.7 * w)
    cut = x + lo + int(np.argmin(profile[lo:hi]))
    halves = []
    for a, b in ((x, cut), (cut, x + w)):
        rows = np.where(mask[y:y + h, a:b].any(axis=1))[0]
        halves.append((a, y + int(rows[0]), b - a, int(rows[-1] - rows[0] + 1)))
    return halves


def next_name(folder: Path, label: str, touching: bool) -> Path:
    """label_01.jpg, label_02.jpg ... (label_t1.jpg ... when touching), never overwriting."""
    i = 1
    while True:
        stem = f"{label}_t{i}" if touching else f"{label}_{i:02d}"
        path = folder / f"{stem}.jpg"
        if not path.exists():
            return path
        i += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a sheet photo into labelled images.")
    parser.add_argument("photo")
    parser.add_argument("--rows", nargs="+", required=True,
                        help='one quoted string per written row, e.g. "17 49 38"')
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                        help="only look inside this rectangle (skips the table, sheet edges)")
    parser.add_argument("--number-gap", type=int, default=120,
                        help="gap in pixels that separates two numbers on the same row")
    parser.add_argument("--digit-gap", type=int, default=60,
                        help="the same, for rows of single digits written close together")
    parser.add_argument("--main-blob-only", nargs="*", default=[],
                        help="labels to crop on their largest blob only, to drop a stray mark")
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="preview only, write no crops")
    args = parser.parse_args()

    image = cv2.imread(args.photo)           # cv2 applies the EXIF rotation
    if image is None:
        raise SystemExit(f"Could not read {args.photo}. HEIC? Export it as JPG first.")
    height, width = image.shape[:2]

    mask = ink_mask(image)
    if args.roi:
        x0, y0, x1, y1 = args.roi
        keep = np.zeros_like(mask)
        keep[y0:y1, x0:x1] = 1
        mask *= keep

    labels = [r.split() for r in args.rows]
    rows = split_rows(find_components(mask), row_gap=int(0.04 * height))
    if len(rows) != len(labels):
        raise SystemExit(f"Found {len(rows)} rows of ink but {len(labels)} rows of labels. "
                         f"Adjust --roi so only the written area is inside it.")

    crops = []
    for row, row_labels in zip(rows, labels):
        mostly_single = sum(len(l) == 1 for l in row_labels) > len(row_labels) / 2
        groups = split_groups(row, args.digit_gap if mostly_single else args.number_gap)
        items = [(bounding(g), g) for g in groups]
        while len(items) < len(row_labels):
            # Fewer groups than labels: two digits are joined. Split the widest.
            i = max(range(len(items)), key=lambda k: items[k][0][2])
            left, right = split_box(mask, items[i][0])
            print(f"  row {row_labels}: split a joined group at x={right[0]}")
            items[i:i + 1] = [(left, None), (right, None)]
        if len(items) != len(row_labels):
            raise SystemExit(f"Row {row_labels}: found {len(items)} numbers. "
                             f"Adjust --number-gap / --digit-gap.")
        for (box, group), label in zip(items, row_labels):
            if label in args.main_blob_only and group:
                box = bounding([max(group, key=lambda c: c["area"])])
            x, y, w, h = box
            sub = mask[y:y + h, x:x + w]
            n, _, st, _ = cv2.connectedComponentsWithStats(sub, 8)
            total = max(1, int((sub > 0).sum()))
            blobs = sum(1 for i in range(1, n) if st[i, 4] > 0.12 * total / len(label))
            single = len(label) == 1
            crops.append(dict(label=label, single=single,
                              touching=(not single and blobs < len(label)),
                              crop=image[max(0, y - args.margin):min(height, y + h + args.margin),
                                         max(0, x - args.margin):min(width, x + w + args.margin)]))

    OUT.mkdir(parents=True, exist_ok=True)
    save_preview(crops, OUT / "split_preview.jpg")
    print(f"Preview: {(OUT / 'split_preview.jpg').relative_to(ROOT)}")
    if args.dry_run:
        return

    for c in crops:
        folder = OUT / ("digits" if c["single"] else "numbers")
        folder.mkdir(parents=True, exist_ok=True)
        path = next_name(folder, c["label"], c["touching"])
        cv2.imwrite(str(path), c["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        print(f"  {path.relative_to(ROOT)}")

    sheets = OUT / "sheets"
    sheets.mkdir(exist_ok=True)
    shutil.copy2(args.photo, sheets / Path(args.photo).name)
    n_touch = sum(c["touching"] for c in crops)
    print(f"{len(crops)} images written ({n_touch} touching). "
          f"Original photo kept in {sheets.relative_to(ROOT)}/")


def save_preview(crops: list[dict], path: Path, row_width: int = 1500) -> None:
    tiles = []
    for c in crops:
        im = c["crop"]
        scale = 180 / im.shape[0]
        im = cv2.resize(im, (max(1, int(im.shape[1] * scale)), 180))
        tile = np.full((215, max(im.shape[1], 120), 3), 255, np.uint8)
        tile[35:, :im.shape[1]] = im
        tag = c["label"] + (" (touching)" if c["touching"] else "")
        cv2.putText(tile, tag, (4, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        tiles.append(tile)
    lines, line, used = [], [], 0
    for t in tiles:
        if used + t.shape[1] > row_width and line:
            lines.append(line)
            line, used = [], 0
        line.append(t)
        used += t.shape[1] + 10
    lines.append(line)
    canvas = []
    for line in lines:
        strip = np.full((215, row_width, 3), 235, np.uint8)
        x = 0
        for t in line:
            strip[:, x:x + t.shape[1]] = t[:, :min(t.shape[1], row_width - x)]
            x += t.shape[1] + 10
        canvas.append(strip)
    cv2.imwrite(str(path), np.vstack(canvas))


if __name__ == "__main__":
    main()
