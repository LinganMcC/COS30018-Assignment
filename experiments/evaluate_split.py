"""Task 2, second round - does repairing touching and broken digits help?

Compares connected-ink segmentation as it was (the Task 2 selection) against
the repairs in src/segmentation_split.py, on two separate sets:

    development   John's two original sheets + the generated numbers.
                  Every parameter in segmentation_split.py was chosen here.
    held-out      the four sheets added later (Thien, Liam, Russell, and the
                  lined sheet). Not looked at while choosing anything.

Keeping the two apart is the point: a repair tuned on the very images it is
scored on would look better than it is.

Run:
    python experiments/evaluate_split.py              # the comparison (needs TensorFlow)
    python experiments/evaluate_split.py --tune       # parameter search, development set only
    python experiments/evaluate_split.py --count-only # no TensorFlow: digit counts only
    python experiments/evaluate_split.py --figure     # before / after on the numbers it fixed

Outputs:
    reports/split_comparison.csv
    reports/split_examples.png      with --figure
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import segmentation_split as S
from preprocessing import SELECTED_CONFIG, preprocess
from segmentation import _prepare_binary, segment_digits
from test_set import SAMPLES, load_numbers

REPORTS = ROOT / "reports"
DEV_SHEETS = {"IMG_1406.JPG", "IMG_1407.JPG"}       # John's first two sheets
GENERATED = ROOT / "data" / "generated"
MODEL = ROOT / "models" / "checkpoints" / "cnn_best.keras"


# ---------------------------------------------------------------------------
# The two sets
# ---------------------------------------------------------------------------

def load_sets() -> tuple[list, list]:
    """(development, held_out), each a list of (name, image, label, touching)."""
    with open(SAMPLES / "sources.csv", encoding="utf-8") as fh:
        sheet_of = {r["file"]: r["sheet"] for r in csv.DictReader(fh)}
    dev, held = [], []
    for name, image, label in load_numbers():
        item = (name, image, label, "_t" in Path(name).stem)
        (dev if sheet_of.get(f"numbers/{name}") in DEV_SHEETS else held).append(item)

    labels = GENERATED / "labels.csv"
    if labels.exists():
        with open(labels, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                image = cv2.imread(str(GENERATED / r["filename"]))
                if image is not None:
                    dev.append((f"generated/{r['filename']}", image, r["label"],
                                r["difficulty"] == "touching"))
    else:
        print("  ! data/generated/ not found - development set is the two sheets only")
    return dev, held


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

class Reader:
    """cnn_best, reading a batch of photo crops at once."""

    def __init__(self):
        import tensorflow as tf
        self.model = tf.keras.models.load_model(str(MODEL))

    def probabilities(self, crops) -> np.ndarray:
        if len(crops) == 0:
            return np.zeros((0, 10))
        x = np.stack([preprocess(c, config=SELECTED_CONFIG, add_channel=True) for c in crops])
        return self.model.predict(x, verbose=0)[:, :10]


def run(method: str, image: np.ndarray, reader: Reader | None):
    """(count, read) for one image. read is None when there is no reader."""
    if method == "contours":
        # the Task 2 selection exactly as the pipeline uses it: plain box crops
        _, boxes = segment_digits(image)
        crops = [image[y:y + h, x:x + w] for (x, y, w, h) in boxes]
        count = len(boxes)
    else:
        merge, split = CONFIGS[method]
        if split == "recognition" and reader is None:
            return None, None
        binary, pieces = S.segment_pieces(
            image, merge=merge, split=split,
            scorer=reader.probabilities if reader else None)
        crops = [S.photo_crop(image, binary, p) for p in pieces]
        count = len(pieces)
    if reader is None:
        return count, None
    probs = reader.probabilities(crops)
    return count, "".join(str(int(p.argmax())) for p in probs)


# name -> (merge, split); "contours" is the baseline and handled separately
CONFIGS = {
    "merge": (True, None),
    "merge + projection": (True, "projection"),
    "merge + dropfall": (True, "dropfall"),
    "merge + recognition": (True, "recognition"),
}
METHODS = ["contours"] + list(CONFIGS)


def score(items, method, reader) -> dict:
    count_ok = exact = t_count = t_exact = s_count = s_exact = 0
    n_t = sum(1 for *_, t in items if t)
    for _, image, label, touching in items:
        count, read = run(method, image, reader)
        if count is None:
            return {}
        c = count == len(label)
        e = read == label if read is not None else False
        count_ok += c
        exact += e
        if touching:
            t_count += c
            t_exact += e
        else:
            s_count += c
            s_exact += e
    n, n_s = len(items), len(items) - n_t
    pct = lambda a, b: round(a / b, 4) if b else ""
    out = {"count": pct(count_ok, n), "count_touching": pct(t_count, n_t),
           "count_separate": pct(s_count, n_s)}
    if reader is not None:
        out.update({"exact": pct(exact, n), "exact_touching": pct(t_exact, n_t),
                    "exact_separate": pct(s_exact, n_s)})
    return out


# ---------------------------------------------------------------------------
# Parameter search - development set only
# ---------------------------------------------------------------------------

def tune(dev) -> None:
    """Grid over the model-free parameters, scored on digit count, development only."""
    base = sum(len(segment_digits(img)[1]) == len(lab) for _, img, lab, _ in dev)
    print(f"Development set: {len(dev)} numbers; contours counts {base} right\n")

    print("merge overlap   right count")
    for mo in (0.5, 0.6, 0.7, 0.8, 0.9):
        S.MERGE_OVERLAP = mo
        ok = sum(len(S.segment_pieces(img, merge=True, split=None)[1]) == len(lab)
                 for _, img, lab, _ in dev)
        print(f"  {mo:.1f}          {ok}")

    print("\nsplit ratio  digit ratio  window   projection  dropfall")
    rows = []
    for sr, dr, win in itertools.product((0.85, 0.95, 1.05, 1.15, 1.3),
                                         (0.55, 0.62, 0.7), (0.25, 0.35)):
        S.SPLIT_RATIO, S.DIGIT_RATIO, S.CUT_WINDOW = sr, dr, win
        res = []
        for split in ("projection", "dropfall"):
            res.append(sum(len(S.segment_pieces(img, merge=True, split=split)[1]) == len(lab)
                           for _, img, lab, _ in dev))
        rows.append((sr, dr, win, *res))
    for r in sorted(rows, key=lambda r: -max(r[3], r[4]))[:10]:
        print(f"  {r[0]:.2f}         {r[1]:.2f}        {r[2]:.2f}     {r[3]:>4}        {r[4]:>4}")


def figure(items, reader) -> None:
    """Every real number the repair fixes or breaks: contours above, repaired below."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def boxes_and_read(method, image):
        if method == "contours":
            _, boxes = segment_digits(image)
        else:
            _, pieces = S.segment_pieces(image, merge=True, split="recognition",
                                         scorer=reader.probabilities)
            boxes = [p.box for p in pieces]
        return boxes, run(method, image, reader)[1]

    changed = []
    for name, image, label, _ in items:
        before = boxes_and_read("contours", image)
        after = boxes_and_read("merge + recognition", image)
        if (before[1] == label) != (after[1] == label):
            changed.append((name, image, label, before, after))
    if not changed:
        print("No number changed between contours and the repair.")
        return

    fig, axes = plt.subplots(2, len(changed), figsize=(3.2 * len(changed), 4.6),
                             dpi=140, squeeze=False)
    for col, (name, image, label, before, after) in enumerate(changed):
        for row, (title, (boxes, read)) in enumerate((("contours", before),
                                                      ("merge + recognition", after))):
            vis = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).copy()
            ok = read == label
            colour = (26, 127, 55) if ok else (192, 0, 0)
            for (x, y, w, h) in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), colour, max(2, image.shape[0] // 150))
            ax = axes[row, col]
            ax.imshow(vis)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{title}: read {read}", fontsize=8, color=np.array(colour) / 255)
            if row == 0:
                ax.set_ylabel(f"true {label}", fontsize=8)
    fig.suptitle("Real numbers whose result changed when the repair was added", fontsize=10)
    fig.tight_layout()
    out = REPORTS / "split_examples.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out.relative_to(ROOT)} ({len(changed)} numbers)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Segmentation repairs, development vs held-out.")
    parser.add_argument("--tune", action="store_true", help="parameter search on the development set")
    parser.add_argument("--count-only", action="store_true", help="skip reading (no TensorFlow)")
    parser.add_argument("--figure", action="store_true", help="before / after figure only")
    args = parser.parse_args()

    dev, held = load_sets()
    if args.tune:
        tune(dev)
        return
    if args.figure:
        real = [d for d in dev if not d[0].startswith("generated/")] + held
        figure(real, Reader())
        return

    reader = None if args.count_only else Reader()
    print(f"Development: {len(dev)} numbers ({sum(t for *_, t in dev)} touching).  "
          f"Held-out: {len(held)} numbers ({sum(t for *_, t in held)} touching).\n")
    rows = []
    for method in METHODS:
        for set_name, items in (("development", dev), ("held-out", held)):
            result = score(items, method, reader)
            if result:
                rows.append({"method": method, "set": set_name, "images": len(items), **result})

    for set_name in ("development", "held-out"):
        print(f"{set_name}")
        head = f"  {'method':22s} {'count':>6s} {'touch':>6s} {'separ':>6s}"
        if reader:
            head += f" {'exact':>6s} {'touch':>6s} {'separ':>6s}"
        print(head)
        for r in (r for r in rows if r["set"] == set_name):
            line = (f"  {r['method']:22s} {r['count']:>6.0%} {r['count_touching']:>6.0%} "
                    f"{r['count_separate']:>6.0%}")
            if reader:
                line += (f" {r['exact']:>6.0%} {r['exact_touching']:>6.0%} "
                         f"{r['exact_separate']:>6.0%}")
            print(line)
        print()
    print("count = numbers split into the right number of digits; exact = read exactly;")
    print("touch / separ = numbers whose digits touch / are apart")

    REPORTS.mkdir(exist_ok=True)
    with open(REPORTS / "split_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved: reports/split_comparison.csv")


if __name__ == "__main__":
    main()
