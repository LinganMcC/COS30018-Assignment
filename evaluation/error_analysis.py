"""Task 4 critical analysis: where and why the final system fails.
Outputs:
    failure_analysis.md     the tables used in the report
    failures.csv            every failed number, its category, truth and reading
    failure_gallery.png     examples of each category, boxes drawn
    confidence.png          confidence of right vs wrong digits
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from typing import Dict, List

import cv2
import numpy as np

from common import RESULTS, ROOT
from evaluate_all import GRID, HIGHLIGHT, INK, MUTED, NEUTRAL, SURFACE, pct

CATEGORIES = ["merged", "dropped", "split", "misread"]
# A digit counts as dropped when the ink segmentation threw away, with nearby pieces
# grouped together, forms a blob at least half as tall as a kept digit. Specks of dirt
# and stray fragments are far smaller than that.
DROPPED_HEIGHT = 0.5
EXPRESSIONS = ROOT / "models" / "experiments" / "eval" / "expressions_results.csv"


def dropped_digit(image: np.ndarray, boxes) -> bool:
    """Did segmentation throw away a digit-sized amount of ink?
    
    Looks only at the ink outside every kept box. Ruled lines and underlines
    (blobs more than 8x wider than tall) are ignored: they are not digits and
    rightly get no box. The remaining discarded pieces are grouped with a small
    dilation, because a faint digit is lost as several fragments, each one too
    small for the noise filter on its own.
    """
    from segmentation import _prepare_binary
    if not boxes:
        return True
    binary = _prepare_binary(image)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    lines = [i for i in range(1, n) if stats[i, cv2.CC_STAT_WIDTH] > 8 * stats[i, cv2.CC_STAT_HEIGHT]]
    outside = (binary > 0) & ~np.isin(labels, lines)
    for (x, y, w, h) in boxes:
        outside[y:y + h, x:x + w] = False
    if not outside.any():
        return False
    grouped = cv2.dilate(outside.astype(np.uint8), np.ones((5, 5), np.uint8))
    _, _, gstats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)
    digit_h = float(np.median([h for (_, _, _, h) in boxes]))
    return bool((gstats[1:, cv2.CC_STAT_HEIGHT] >= DROPPED_HEIGHT * digit_h).any())


def category(truth: str, read: str, boxes, dropped: bool = False) -> str:
    """boxes: list of (x, y, w, h) from segmentation; dropped: dropped_digit() of the image."""
    if read == truth:
        return "correct"
    if len(boxes) < len(truth):
        return "dropped" if dropped else "merged"
    if len(boxes) > len(truth):
        return "split"
    return "misread"


def read_numbers(rec, items):
    """Run the final pipeline on (name, image, label, *group); keep everything needed for analysis."""
    out = []
    for name, img, label, *rest in items:
        text, details = rec.read_image(img)
        boxes = [d["box"] for d in details]
        dropped = dropped_digit(img, boxes)
        out.append({"file": name, "truth": label, "read": text, "boxes": len(details),
                    "group": rest[0] if rest else "", "image": img,
                    "box_list": [d["box"] for d in details],
                    "confidence": [d["confidence"] for d in details],
                    "category": category(label, text, boxes, dropped)})
    return out


def breakdown(rows: List[Dict], key) -> List[Dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    out = []
    for g, rs in sorted(groups.items()):
        c = Counter(r["category"] for r in rs)
        out.append({"group": g, "n": len(rs), "correct": c["correct"] / len(rs),
                    **{k: c[k] for k in CATEGORIES}})
    return out


def confusion_pairs(rows: List[Dict]) -> Counter:
    pairs = Counter()
    for r in rows:
        if r["category"] == "misread":
            pairs.update((t, p) for t, p in zip(r["truth"], r["read"]) if t != p)
    return pairs


def confidence_split(rows: List[Dict]):
    """Confidence of each digit on numbers with the right count: right digits vs wrong ones."""
    right, wrong = [], []
    for r in rows:
        if r["boxes"] == len(r["truth"]):
            for t, p, c in zip(r["truth"], r["read"], r["confidence"]):
                (right if t == p else wrong).append(c)
    return np.array(right), np.array(wrong)


def gallery(rows: List[Dict], path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    picks = []
    for cat in CATEGORIES:
        picks += [r for r in rows if r["category"] == cat][:4]
    if not picks:
        return
    cols = 4
    nrows = int(np.ceil(len(picks) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(4 * cols, 2.6 * nrows), facecolor=SURFACE)
    for ax, r in zip(np.ravel(axes), picks):
        img = r["image"].copy()
        for (x, y, w, h) in r["box_list"]:
            cv2.rectangle(img, (x, y), (x + w, y + h), (214, 120, 42), max(2, img.shape[1] // 250))
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{r['category']}: true {r['truth']}, read {r['read'] or '–'}",
                     fontsize=9, color=INK, loc="left")
        ax.axis("off")
    for ax in np.ravel(axes)[len(picks):]:
        ax.axis("off")
    fig.suptitle("Failures of the final pipeline on real photos (boxes = segmentation output)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def confidence_chart(right, wrong, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True, facecolor=SURFACE)
    for ax, data, colour, title in ((axes[0], right, NEUTRAL, f"Digits read correctly (n = {len(right)})"),
                                    (axes[1], wrong, HIGHLIGHT, f"Digits read wrongly (n = {len(wrong)})")):
        ax.hist(data, bins=bins, color=colour, edgecolor=SURFACE, linewidth=2)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("model confidence", color=MUTED, fontsize=9)
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.spines["left"].set_color(GRID)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0, labelsize=8)
    axes[0].set_ylabel("digits", color=MUTED, fontsize=9)
    fig.suptitle("Confidence of the selected model on real photographed numbers",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def expression_breakdown():
    if not EXPRESSIONS.exists():
        return None
    rows = list(csv.DictReader(open(EXPRESSIONS, encoding="utf-8")))
    out = defaultdict(Counter)
    for r in rows:
        if r["exact"] == "True":
            cat = "correct"
        elif r["length_ok"] != "True":
            cat = "wrong symbol count (segmentation)"
        else:
            cat = "right count, symbol misread (recognition)"
        out["all"][cat] += 1
        out[f"level {r['level']}"][cat] += 1
    return out


def main():
    from generated_set import load as load_generated
    from inference import Recogniser
    from test_set import load_numbers, load_writers
    rec = Recogniser("cnn_best")
    writers = load_writers()
    real = read_numbers(rec, [(n, i, l, "touching" if "_t" in n else "separate")
                              for n, i, l in load_numbers()])
    for r in real:
        sheet, writer = writers.get(f"numbers/{r['file']}", ("", ""))
        r["writer"], r["sheet"] = writer or "unknown", sheet
    generated = read_numbers(rec, load_generated())

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "failures.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["set", "file", "group", "truth", "read", "boxes", "category"])
        w.writeheader()
        for tag, rows in (("real", real), ("generated", generated)):
            for r in rows:
                if r["category"] != "correct":
                    w.writerow({"set": tag, **{k: r[k] for k in ("file", "group", "truth", "read",
                                                                  "boxes", "category")}})

    md = ["# Where and why the system fails – Task 4 critical analysis", "",
          "Generated by `python evaluation/error_analysis.py`. Selected pipeline: `contours` segmentation, "
          "`otsu_mnist_box` preprocessing, VGG-deep (`cnn_best`); extension: `ext_best`.", "",
          "Every failure is put in one category by comparing the boxes with the true digits: **merged** "
          "(fewer boxes than digits, no digit-sized ink thrown away – digits cut together), "
          "**dropped** (fewer boxes, and the discarded ink adds up to a digit), **split** (more boxes than "
          "digits) – all three segmentation – or **misread** (right number of boxes, a digit wrong) – "
          "recognition.", ""]

    for title, rows in (("Real photographed numbers (99)", real), ("Generated numbers from MNIST test digits (600)", generated)):
        c = Counter(r["category"] for r in rows)
        fails = len(rows) - c["correct"]
        md += [f"## {title}", "",
               f"{c['correct']} of {len(rows)} read exactly ({pct(c['correct'] / len(rows))}); {fails} failures:", "",
               "| Category | Failures | Share of failures | Cause |", "|---|---|---|---|"]
        causes = {"merged": "segmentation – touching or overlapping digits cut as one",
                  "dropped": "segmentation – a faint digit broken into fragments too small to keep",
                  "split": "segmentation – a digit broken into pieces, or a speck kept",
                  "misread": "recognition – right cut, wrong digit"}
        for cat in CATEGORIES:
            md.append(f"| {cat} | {c[cat]} | {pct(c[cat] / fails) if fails else '–'} | {causes[cat]} |")
        seg = c["merged"] + c["dropped"] + c["split"]
        md += ["", f"**Segmentation causes {pct(seg / fails) if fails else '–'} of these failures, "
               f"recognition {pct(c['misread'] / fails) if fails else '–'}.**", ""]
        key = lambda r: r["group"]
        label = "Spacing" if rows is generated else "Digits"
        md += [f"| {label} | Numbers | Read exactly | Merged | Dropped | Split | Misread |", "|---|---|---|---|---|---|---|"]
        for b in breakdown(rows, key):
            md.append(f"| {b['group']} | {b['n']} | {pct(b['correct'])} | {b['merged']} | {b['dropped']} | {b['split']} | {b['misread']} |")
        if rows is real:
            md += ["", "| Writer | Numbers | Read exactly | Merged | Dropped | Split | Misread |", "|---|---|---|---|---|---|---|"]
            for b in breakdown(rows, lambda r: r["writer"]):
                md.append(f"| {b['group']} | {b['n']} | {pct(b['correct'])} | {b['merged']} | {b['dropped']} | {b['split']} | {b['misread']} |")
            md += ["", "| Sheet | Writer | Numbers | Read exactly |", "|---|---|---|---|"]
            sheet_writer = {r["sheet"]: r["writer"] for r in rows}
            for b in breakdown(rows, lambda r: r["sheet"]):
                md.append(f"| {b['group']} | {sheet_writer[b['group']]} | {b['n']} | {pct(b['correct'])} |")
        pairs = confusion_pairs(rows)
        if pairs:
            md += ["", "Most frequent misreadings (true → read): " +
                   ", ".join(f"{t}→{p} ({n})" for (t, p), n in pairs.most_common(8)), ""]
        else:
            md.append("")

    right, wrong = confidence_split(real)
    if len(wrong):
        md += ["## Could the system warn the user?", "",
               "Confidence of every digit on real numbers that were cut into the right number of pieces:", "",
               "| | Digits | Median confidence | Below 0.9 |", "|---|---|---|---|",
               f"| read correctly | {len(right)} | {np.median(right):.3f} | {pct((right < 0.9).mean())} |",
               f"| read wrongly | {len(wrong)} | {np.median(wrong):.3f} | {pct((wrong < 0.9).mean())} |", ""]
        for t in (0.9, 0.99):
            flagged = (right < t).sum() + (wrong < t).sum()
            md.append(f"- Flagging digits below **{t}** would catch {pct((wrong < t).mean())} of the wrong digits "
                      f"while flagging {flagged} of {len(right) + len(wrong)} digits in total.")
        md.append("")
        confidence_chart(right, wrong, RESULTS / "confidence.png")

    ex = expression_breakdown()
    if ex:
        cats = ["correct", "wrong symbol count (segmentation)", "right count, symbol misread (recognition)"]
        md += ["## Extension: handwritten expressions (ext_best, 408 expressions by 14 unseen writers)", "",
               "| Set | Expressions | " + " | ".join(cats) + " |", "|---|---|" + "---|" * len(cats)]
        for key in ["all"] + sorted(k for k in ex if k != "all"):
            n = sum(ex[key].values())
            md.append(f"| {key} | {n} | " + " | ".join(f"{ex[key][c]} ({pct(ex[key][c] / n)})" for c in cats) + " |")
        md.append("")

    gallery(real, RESULTS / "failure_gallery.png")
    md += ["Figures: `failure_gallery.png` (examples of each category), `confidence.png`. "
           "Every failure: `failures.csv`."]
    (RESULTS / "failure_analysis.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"-> {(RESULTS / 'failure_analysis.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
