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
import csv
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


def paper_mask(image: np.ndarray, inset: int = 10) -> np.ndarray:
    """The sheet of paper itself, shrunk by `inset` pixels.

    The edge between white paper and a darker desk looks like a long dark
    stroke to the black-hat filter, so it came out as "ink" whenever the ROI
    reached the desk. Keeping only ink inside the sheet removes it, and makes
    the ROI a convenience rather than something that has to be exact.
    """
    # Paper is bright and colourless. Brightness alone is not enough: a light
    # wooden desk is nearly as bright, but it is clearly coloured.
    hsv = cv2.cvtColor(cv2.GaussianBlur(image, (51, 51), 0), cv2.COLOR_BGR2HSV)
    bright = (((hsv[..., 1] < 45) & (hsv[..., 2] > 110)) * 255).astype(np.uint8)
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sheet = np.zeros_like(bright)
    if contours:
        # the largest bright region, filled, so the writing on it is included
        cv2.drawContours(sheet, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inset + 1, 2 * inset + 1))
    return cv2.erode(sheet, kernel)


def ruled_lines(mask: np.ndarray, length: int = 301, drift: int = 25) -> np.ndarray:
    """Long, nearly horizontal strokes: the ruled lines on lined paper.

    A line runs under, and sometimes through, a whole row of writing, so it
    joins every digit on it into one blob and the row can no longer be cut.
    An opening with a `length`-pixel horizontal bar keeps only ink that runs
    straight for that far - far longer than any stroke of a digit at this
    scale. The mask is first stretched vertically by `drift` pixels so a line
    that is slightly tilted or curved, as a hand-drawn one is, still counts.

    Used only to find where the numbers are. The saved crops come from the
    photo and keep the line, because lined paper is the condition under test.
    """
    tall = cv2.dilate(mask, np.ones((drift, 1), np.uint8))
    lines = cv2.morphologyEx(tall, cv2.MORPH_OPEN, np.ones((1, length), np.uint8))
    lines = cv2.bitwise_and(cv2.dilate(lines, np.ones((5, 5), np.uint8)), mask)

    # The ends of a hand-drawn line curve more than `drift` allows and survive
    # as loose flat pieces. Anything left that is long and flat is line too.
    rest = cv2.subtract(mask, lines)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(rest, 8)
    for i in range(1, n):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if w >= length // 2 and h < w / 4:
            lines[labels == i] = 255
    return lines


def find_components(mask: np.ndarray, min_area: int = 120) -> list[dict]:
    n, _, stats, centres = cv2.connectedComponentsWithStats(mask, 8)
    return [dict(x=int(stats[i, 0]), y=int(stats[i, 1]), w=int(stats[i, 2]),
                 h=int(stats[i, 3]), area=int(stats[i, 4]), cy=float(centres[i, 1]))
            for i in range(1, n) if stats[i, 4] >= min_area]


def split_rows(components: list[dict], row_gap: int) -> list[list[dict]]:
    """Group blobs into rows of writing, top to bottom.

    A blob joins the row above it if its centre is close to the row's, or if
    at least half of its height overlaps the row's vertical extent. The second
    test is for slanted writing: the "1" of a 104 written sloping down to the
    left sat 260 px below its row's centre, became a row of its own, and was
    then dropped as a stray mark.
    """
    components = sorted(components, key=lambda c: c["cy"])
    rows: list[list[dict]] = []
    for c in components:
        if rows:
            last = rows[-1]
            near = abs(c["cy"] - np.median([d["cy"] for d in last])) < row_gap
            bottom = max(d["y"] + d["h"] for d in last)
            overlap = min(bottom, c["y"] + c["h"]) - max(min(d["y"] for d in last), c["y"])
            if near or overlap >= 0.5 * c["h"]:
                last.append(c)
                continue
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


def split_groups_by_count(row: list[dict], count: int,
                          forced: list[int] | None = None) -> list[list[dict]]:
    """Cut a row into `count` numbers at its widest horizontal gaps.

    The labels say how many numbers a row holds, so there is no need to guess
    a gap threshold. A fixed threshold failed on thin, spaced-out handwriting:
    gaps inside one number reached ~105 px while gaps between numbers were as
    small as ~160 px on the same sheet, and no single value fits every row.
    If the row has fewer blobs than `count`, every gap is a cut and the caller
    splits joined digits afterwards.

    `forced` lists x positions that must be cuts, for a row where the writer
    left a wider space inside a number than between two numbers. The gap
    closest to each position is always taken.
    """
    row = sorted(row, key=lambda c: c["x"])
    gaps = []                              # (gap width, index of first blob after it)
    right = row[0]["x"] + row[0]["w"]
    for i, c in enumerate(row[1:], 1):
        gaps.append((c["x"] - right, i))
        right = max(right, c["x"] + c["w"])
    for fx in forced or []:
        k = min(range(len(gaps)), key=lambda j: abs(row[gaps[j][1]]["x"] - fx))
        gaps[k] = (float("inf"), gaps[k][1])
    cuts = sorted(i for _, i in sorted(gaps, reverse=True)[:count - 1])
    bounds = [0] + cuts + [len(row)]
    return [row[a:b] for a, b in zip(bounds, bounds[1:])]


def drop_stray_rows(rows: list[list[dict]]) -> list[list[dict]]:
    """Drop 'rows' that are only a dot or a speck between two real rows."""
    ink = [sum(c["area"] for c in r) for r in rows]
    floor = 0.15 * float(np.median(ink))
    return [r for r, a in zip(rows, ink) if a >= floor]


def clip_to_paper(paper: np.ndarray, outer: tuple[int, int, int, int],
                  inner: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Pull each edge of the crop `outer` in until it lies on paper.

    The margin of paper kept around a number would otherwise run off the sheet
    for writing near its edge, and a strip of desk in the crop is something the
    preprocessing never meets in use. Edges never move inside the ink box
    `inner`, so no writing is lost.
    """
    x0, y0, x1, y1 = outer
    ix0, iy0, ix1, iy1 = inner
    while x0 < ix0 and not paper[y0:y1, x0].all():
        x0 += 1
    while x1 > ix1 and not paper[y0:y1, x1 - 1].all():
        x1 -= 1
    while y0 < iy0 and not paper[y0, x0:x1].all():
        y0 += 1
    while y1 > iy1 and not paper[y1 - 1, x0:x1].all():
        y1 -= 1
    return x0, y0, x1, y1


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


def valley_depth(mask: np.ndarray, box: tuple[int, int, int, int]) -> float:
    """How much a box looks like two shapes side by side; lower = more so.

    The emptiest column in the middle 40%, as a fraction of the average
    column. A wide 0 is not a candidate even though it is the widest box: its
    middle columns cross the stroke twice, the same as the rest of the loop,
    so the ratio stays near 1. Two digits joined by a thin touch drop well
    below that at the join.
    """
    x, y, w, h = box
    profile = (mask[y:y + h, x:x + w] > 0).sum(axis=0)
    lo, hi = int(0.3 * w), int(0.7 * w)
    if hi <= lo or profile.mean() == 0:
        return 1.0
    return float(profile[lo:hi].min() / profile.mean())


def next_name(folder: Path, label: str, touching: bool) -> Path:
    """label_01.jpg, label_02.jpg ... (label_t1.jpg ... when touching), never overwriting."""
    i = 1
    while True:
        stem = f"{label}_t{i}" if touching else f"{label}_{i:02d}"
        path = folder / f"{stem}.jpg"
        if not path.exists():
            return path
        i += 1


SOURCES = OUT / "sources.csv"     # one row per image: which sheet it was cut from
SHEETS = OUT / "sheets.csv"       # one row per sheet: who wrote it, and how


def record_sources(files: list[str], sheet: str, writer: str, conditions: str) -> None:
    """Note which sheet each new image came from, and who wrote that sheet.

    Kept beside the images rather than in the filename, so the label_NN.jpg
    convention every other script relies on stays as it is. With it the
    results can be split by writer, which is the point of collecting sheets
    from more than one person.
    """
    new_file = not SOURCES.exists()
    with open(SOURCES, "a", newline="", encoding="utf-8") as fh:
        out = csv.writer(fh)
        if new_file:
            out.writerow(["file", "sheet"])
        out.writerows([f, sheet] for f in files)

    rows = []
    if SHEETS.exists():
        with open(SHEETS, encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r["sheet"] != sheet]
    rows.append({"sheet": sheet, "writer": writer, "conditions": conditions})
    with open(SHEETS, "w", newline="", encoding="utf-8") as fh:
        out = csv.DictWriter(fh, fieldnames=["sheet", "writer", "conditions"])
        out.writeheader()
        out.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split a sheet photo into labelled images.")
    parser.add_argument("photo")
    parser.add_argument("--rows", nargs="+", required=True,
                        help='one quoted string per written row, e.g. "17 49 38"')
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                        help="only look inside this rectangle (skips the table, sheet edges)")
    # By default a row is cut at its widest gaps, as many as the labels need.
    # These fixed thresholds are the older behaviour, kept for re-running the
    # first two sheets exactly as they were cut.
    parser.add_argument("--number-gap", type=int, default=None,
                        help="fixed gap in pixels between two numbers (default: cut by label count)")
    parser.add_argument("--digit-gap", type=int, default=None,
                        help="the same, for rows of single digits")
    parser.add_argument("--force-cut", nargs="*", default=[], metavar="ROW:X",
                        help="always cut row ROW (1 = top) at pixel column X, "
                             "for a number written with a wide space inside it")
    parser.add_argument("--main-blob-only", nargs="*", default=[],
                        help="labels to crop on their largest blob only, to drop a stray mark")
    parser.add_argument("--skip", nargs="*", default=[], metavar="ROW:N",
                        help="do not save the Nth item of row ROW (both from 1), e.g. a "
                             "digit written on top of its neighbour so no crop is clean")
    parser.add_argument("--writer", default="",
                        help="who wrote the sheet, recorded in sheets.csv")
    parser.add_argument("--conditions", default="",
                        help="how it was written and photographed, recorded in sheets.csv")
    parser.add_argument("--lined", action="store_true",
                        help="the sheet has ruled lines; ignore them when locating numbers")
    parser.add_argument("--margin", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true", help="preview only, write no crops")
    args = parser.parse_args()

    image = cv2.imread(args.photo)           # cv2 applies the EXIF rotation
    if image is None:
        raise SystemExit(f"Could not read {args.photo}. HEIC? Export it as JPG first.")
    height, width = image.shape[:2]

    paper = paper_mask(image, inset=0)
    mask = cv2.bitwise_and(ink_mask(image), paper_mask(image))
    if args.lined:
        mask = cv2.subtract(mask, ruled_lines(mask))
    if args.roi:
        x0, y0, x1, y1 = args.roi
        keep = np.zeros_like(mask)
        keep[y0:y1, x0:x1] = 1
        mask *= keep

    labels = [r.split() for r in args.rows]
    rows = drop_stray_rows(split_rows(find_components(mask), row_gap=int(0.04 * height)))
    if len(rows) != len(labels):
        raise SystemExit(f"Found {len(rows)} rows of ink but {len(labels)} rows of labels. "
                         f"Adjust --roi so only the written area is inside it.")

    forced: dict[int, list[int]] = {}
    for item in args.force_cut:
        r, x = item.split(":")
        forced.setdefault(int(r) - 1, []).append(int(x))

    skip = {tuple(int(v) - 1 for v in item.split(":")) for item in args.skip}

    crops = []
    for row_index, (row, row_labels) in enumerate(zip(rows, labels)):
        mostly_single = sum(len(l) == 1 for l in row_labels) > len(row_labels) / 2
        gap = args.digit_gap if mostly_single else args.number_gap
        if gap is None:
            groups = split_groups_by_count(row, len(row_labels), forced.get(row_index))
        else:
            groups = split_groups(row, gap)
        items = [(bounding(g), g) for g in groups]
        while len(items) < len(row_labels):
            # Fewer groups than labels: two digits are joined. Split the box
            # that most looks like two shapes. Splitting simply the widest one
            # cut a large 0 in half while the joined pair next to it survived.
            widest = max(items[k][0][2] for k in range(len(items)))
            candidates = [k for k in range(len(items)) if items[k][0][2] >= 0.5 * widest]
            i = min(candidates, key=lambda k: valley_depth(mask, items[k][0]))
            left, right = split_box(mask, items[i][0])
            print(f"  row {row_labels}: split a joined group at x={right[0]}")
            items[i:i + 1] = [(left, None), (right, None)]
        if len(items) != len(row_labels):
            raise SystemExit(f"Row {row_labels}: found {len(items)} numbers. "
                             f"Adjust --number-gap / --digit-gap.")
        for position, ((box, group), label) in enumerate(zip(items, row_labels)):
            if (row_index, position) in skip:
                print(f"  row {row_index + 1}: skipped item {position + 1} ('{label}')")
                continue
            if label in args.main_blob_only and group:
                box = bounding([max(group, key=lambda c: c["area"])])
            x, y, w, h = box
            sub = mask[y:y + h, x:x + w]
            n, _, st, _ = cv2.connectedComponentsWithStats(sub, 8)
            total = max(1, int((sub > 0).sum()))
            blobs = sum(1 for i in range(1, n) if st[i, 4] > 0.12 * total / len(label))
            single = len(label) == 1
            cx0, cy0, cx1, cy1 = clip_to_paper(
                paper,
                (max(0, x - args.margin), max(0, y - args.margin),
                 min(width, x + w + args.margin), min(height, y + h + args.margin)),
                (x, y, x + w, y + h))
            crops.append(dict(label=label, single=single,
                              touching=(not single and blobs < len(label)),
                              crop=image[cy0:cy1, cx0:cx1]))

    OUT.mkdir(parents=True, exist_ok=True)
    save_preview(crops, OUT / "split_preview.jpg")
    print(f"Preview: {(OUT / 'split_preview.jpg').relative_to(ROOT)}")
    if args.dry_run:
        return

    sheet_name = Path(args.photo).name
    written = []
    for c in crops:
        folder = OUT / ("digits" if c["single"] else "numbers")
        folder.mkdir(parents=True, exist_ok=True)
        path = next_name(folder, c["label"], c["touching"])
        cv2.imwrite(str(path), c["crop"], [cv2.IMWRITE_JPEG_QUALITY, 95])
        written.append(f"{folder.name}/{path.name}")
        print(f"  {path.relative_to(ROOT)}")
    record_sources(written, sheet_name, args.writer, args.conditions)

    sheets = OUT / "sheets"
    sheets.mkdir(exist_ok=True)
    if not (sheets / sheet_name).exists():     # re-cutting a sheet keeps the first copy
        shutil.copy2(args.photo, sheets / sheet_name)
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
