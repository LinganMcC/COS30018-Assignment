"""The real handwritten test set: loading it, and describing what is in it.

This is the hand-over point for evaluation (Task 4). Everything that reads the
photographed samples goes through here, so the filename convention and the
"which images exist" question have one answer and not one per script.

    from test_set import load_digits, load_numbers
    for filename, image, label in load_numbers():
        ...                      # image is BGR, straight from cv2.imread

Naming convention - the label comes first, then an underscore:

    data/custom_samples/digits/3_01.jpg        one digit, label 3
    data/custom_samples/numbers/94026_01.jpg   a whole number, label 94026

Run it directly to write the manifest and print a summary of the set:

    python src/test_set.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "data" / "custom_samples"
MANIFEST = ROOT / "reports" / "custom_test_set.csv"
IMAGE_TYPES = {".png", ".jpg", ".jpeg"}

Sample = tuple[str, np.ndarray, str]


def _load(folder: Path, single_digit: bool) -> list[Sample]:
    if not folder.exists():
        return []

    heic = sorted(p.name for p in folder.iterdir() if p.suffix.lower() == ".heic")
    if heic:
        print(f"  ! {len(heic)} HEIC file(s) in {folder.name}/ cannot be read by "
              f"OpenCV. Export them as JPG, e.g. {heic[0]}")

    samples = []
    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in IMAGE_TYPES:
            continue
        label = path.stem.split("_")[0]
        if not label.isdigit() or (single_digit and len(label) != 1):
            print(f"  ! skipped {path.name}: the filename must start with the "
                  f"label, e.g. 3_01.jpg")
            continue
        image = cv2.imread(str(path))
        if image is None:
            print(f"  ! skipped {path.name}: could not be read")
            continue
        samples.append((path.name, image, label))
    return samples


def load_digits() -> list[Sample]:
    """Single-digit photos, for per-digit accuracy."""
    return _load(SAMPLES / "digits", single_digit=True)


def load_numbers() -> list[Sample]:
    """Multi-digit number photos, for whole-number exact-match accuracy."""
    return _load(SAMPLES / "numbers", single_digit=False)


# ---------------------------------------------------------------------------
# Describing the set
# ---------------------------------------------------------------------------

def stroke_width(image: np.ndarray) -> float:
    """Median pen width in pixels, as twice the distance transform on the ink.

    The distance transform gives each ink pixel its distance to the nearest
    paper pixel, so the ridge running down the middle of a stroke holds half
    the stroke width. The median over all ink is a stable stand-in for it, and
    it is what makes "varied pen thickness" a measured claim and not a hope.
    """
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    _, binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if not binary.any():
        return 0.0
    distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    return round(float(2 * np.median(distance[binary > 0])), 2)


def load_writers() -> dict[str, tuple[str, str]]:
    """'digits/3_01.jpg' -> (sheet, writer), from the files split_sheet.py keeps.

    Images cut before these files existed, or added by hand, map to nothing
    and show up with a blank writer rather than being dropped.
    """
    sources, sheets = SAMPLES / "sources.csv", SAMPLES / "sheets.csv"
    if not sources.exists():
        return {}
    writer_of = {}
    if sheets.exists():
        with open(sheets, encoding="utf-8") as fh:
            writer_of = {r["sheet"]: r["writer"] for r in csv.DictReader(fh)}
    with open(sources, encoding="utf-8") as fh:
        return {r["file"]: (r["sheet"], writer_of.get(r["sheet"], ""))
                for r in csv.DictReader(fh)}


def describe(samples: list[Sample], kind: str) -> list[dict]:
    folder = "digits" if kind == "digit" else "numbers"
    origin = load_writers()
    rows = []
    for filename, image, label in samples:
        height, width = image.shape[:2]
        sheet, writer = origin.get(f"{folder}/{filename}", ("", ""))
        rows.append({
            "kind": kind,
            "filename": filename,
            "label": label,
            "sheet": sheet,
            "writer": writer,
            "digits": len(label),
            "width": width,
            "height": height,
            "stroke_width_px": stroke_width(image),
            # Normalised so photos taken at different distances compare: a thick
            # pen on a small crop and a thin pen on a large one are not the same.
            "stroke_width_pct_of_height": round(100 * stroke_width(image) / height, 2),
        })
    return rows


def main() -> None:
    digits = load_digits()
    numbers = load_numbers()
    if not digits and not numbers:
        print(f"No samples found under {SAMPLES.relative_to(ROOT)}/.")
        return

    rows = describe(digits, "digit") + describe(numbers, "number")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Real handwritten test set - {len(digits)} digits, {len(numbers)} numbers")
    print("=" * 62)

    per_class = {str(d): sum(1 for _, _, lab in digits if lab == str(d))
                 for d in range(10)}
    print("Single digits per class: "
          + ", ".join(f"{d}:{n}" for d, n in per_class.items()))
    missing = [d for d, n in per_class.items() if n == 0]
    if missing:
        print(f"  ! no samples for: {', '.join(missing)}")

    lengths = sorted({len(lab) for _, _, lab in numbers})
    print(f"Number lengths present: {lengths}")

    sheets: dict[str, list[dict]] = {}
    for r in rows:
        sheets.setdefault(r["sheet"] or "(unknown)", []).append(r)
    print("By sheet:")
    for sheet, items in sorted(sheets.items()):
        writer = items[0]["writer"] or "writer not set"
        n_dig = sum(r["kind"] == "digit" for r in items)
        print(f"  {sheet:16s} {writer:16s} {n_dig:3d} digits, {len(items) - n_dig:3d} numbers")
    if any(not r["writer"] for r in rows):
        print("  ! fill in the writer column of data/custom_samples/sheets.csv")

    widths = [r["stroke_width_pct_of_height"] for r in rows]
    print(f"Pen thickness (% of image height): min {min(widths):.2f}, "
          f"median {float(np.median(widths)):.2f}, max {max(widths):.2f}")

    print(f"\nManifest written to {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
