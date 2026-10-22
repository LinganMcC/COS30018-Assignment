"""Tasks 3 and 4: every model, every test set, one harness.

The six models compared (common.MODEL_SPECS), simplest to most complex:
    HOG + SVM, HOG + Random Forest      classical machine learning
    MLP                                  simple neural network, no convolutions
    LeNet-5, ResNet (small), VGG-deep    convolutional networks (VGG-deep = cnn_best)

The test sets, each scored the same way for every model:
    A. MNIST test (10,000) and a shifted copy of it (rotated / shifted / zoomed):
       accuracy, precision / recall / F1 per digit, confusion matrices
    B. 149 photographed single digits (data/custom_samples/digits)
    C. 99 photographed multi-digit numbers (data/custom_samples/numbers)
    D. 600 generated numbers from MNIST *test* digits, three spacings
       (evaluation/generated_set.py)

For C and D the spec asks for two numbers, kept separate:
    per-digit accuracy  - how many individual digits are read correctly
    whole-number accuracy - a number counts only if every digit is right
Segmentation (repaired: contours, then touching digits cut where cnn_best reads
the pieces best - inference.segment_number) and preprocessing (otsu_mnist_box)
are the system's own and identical for every model, so differences between rows
are the model.

McNemar's exact test compares every model with the selected one (VGG-deep) on
the same images: are the two really different, or is the gap luck?

Run from the repository root (5–10 min on a laptop CPU):
    python evaluation/baselines.py          # once, trains the two classical models
    python evaluation/generated_set.py      # once, builds test set D
    python evaluation/evaluate_all.py

Outputs (evaluation/results/):
    model_comparison.md       the tables for the report
    model_comparison.csv      one row per model, every metric
    mnist_per_class.csv       precision / recall / F1 per digit per model
    significance.csv          McNemar tests against VGG-deep
    numbers_by_image.csv      every real and generated number, read by every model
    comparison_chart.png      one panel per test set
    confusion_mnist.png       confusion matrix of every model on MNIST test
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from typing import Dict, List

import numpy as np

from common import FINAL_MODEL, RESULTS, ROOT, load_mnist_splits, load_models, wilson

INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
HIGHLIGHT, NEUTRAL = "#2a78d6", "#b8b7b1"
BLUES = ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"]


# ---------------------------------------------------------------- metrics

def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def mcnemar(ref_ok: np.ndarray, other_ok: np.ndarray) -> Dict:
    """Exact McNemar test on paired right/wrong outcomes (two-sided binomial on the discordant pairs)."""
    from scipy.stats import binomtest
    b = int(np.sum(ref_ok & ~other_ok))      # reference right, other wrong
    c = int(np.sum(~ref_ok & other_ok))      # other right, reference wrong
    p = 1.0 if b + c == 0 else binomtest(min(b, c), b + c, 0.5).pvalue
    return {"ref_only_right": b, "other_only_right": c, "p_value": float(p)}


def number_metrics(rows: List[Dict]) -> Dict:
    """Per-digit and whole-number accuracy for one model on a set of numbers.

    rows: dicts with 'truth', 'read', 'count_ok'.
    digit_acc_count_ok  digits right / digits, on numbers whose digit count was right
    digit_acc           1 - edit distance / true length over all numbers (missing,
                        extra and wrong digits all count as errors)
    exact               whole number right
    """
    n = len(rows)
    ok = [r for r in rows if r["count_ok"]]
    pairs = [(t, p) for r in ok for t, p in zip(r["truth"], r["read"])]
    exact = sum(r["read"] == r["truth"] for r in rows)
    chars = sum(len(r["truth"]) for r in rows)
    return {
        "n": n, "exact": exact / n if n else 0.0, "exact_ci": wilson(exact, n),
        "count_ok": len(ok) / n if n else 0.0,
        "digit_acc_count_ok": sum(t == p for t, p in pairs) / len(pairs) if pairs else 0.0,
        "exact_count_ok": sum(r["read"] == r["truth"] for r in ok) / len(ok) if ok else 0.0,
        "digit_acc": 1 - sum(levenshtein(r["read"], r["truth"]) for r in rows) / chars if chars else 0.0,
    }


# ---------------------------------------------------------------- test sets

def mnist_block(models, results: Dict, per_class: List[Dict], quick: bool):
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    from augmentation import make_shifted_test_set
    _, _, (x, y) = load_mnist_splits()
    if quick:
        x, y = x[:1000], y[:1000]
    xs = make_shifted_test_set(x, seed=0)
    correct, confusions = {}, {}
    for m in models:
        for split, data in (("mnist", x), ("mnist_shifted", xs)):
            pred = m.predict_proba(data).argmax(1)
            results[m.key][f"{split}_acc"] = float((pred == y).mean())
            if split == "mnist":
                correct[m.key] = pred == y
                confusions[m.key] = confusion_matrix(y, pred, labels=range(10))
                p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(10), zero_division=0)
                results[m.key]["mnist_macro_f1"] = float(f.mean())
                per_class += [{"model": m.name, "digit": d, "precision": p[d], "recall": r[d],
                               "f1": f[d], "support": int(s[d])} for d in range(10)]
        print(f"  {m.name:22s} MNIST {results[m.key]['mnist_acc']:.4f} | "
              f"shifted {results[m.key]['mnist_shifted_acc']:.4f}")
    return correct, confusions


def real_digits_block(models, results: Dict):
    from preprocessing import preprocess
    from test_set import load_digits
    items = load_digits()
    x = np.stack([preprocess(img, add_channel=True) for _, img, _ in items])
    y = np.array([int(lbl) for _, _, lbl in items])
    correct = {}
    for m in models:
        ok = m.predict_proba(x).argmax(1) == y
        correct[m.key] = ok
        results[m.key]["real_digit_acc"] = float(ok.mean())
        results[m.key]["real_digit_ci"] = wilson(int(ok.sum()), len(ok))
    print(f"  {len(items)} real digit photos")
    return correct


def numbers_block(models, items, tag: str, results: Dict, by_image: List[Dict], extra=None):
    """Segment each number once (shared), then let every model read the same crops.

    Segmentation is the system's: inference.segment_number() (John's repaired
    segmentation, cuts chosen by cnn_best), exactly what the GUI does.
    """
    from preprocessing import preprocess
    from inference import segment_number
    segs = []
    for name, img, label, *rest in items:
        photo_crops, _ = segment_number(img)
        crops = [preprocess(c, add_channel=True) for c in photo_crops]
        segs.append((name, label, rest[0] if rest else "", crops))
    all_crops = np.stack([c for *_, crops in segs for c in crops])
    for m in models:
        labels = m.predict_proba(all_crops).argmax(1)
        rows, k = [], 0
        for name, label, group, crops in segs:
            read = "".join(str(d) for d in labels[k:k + len(crops)])
            k += len(crops)
            row = {"set": tag, "file": name, "group": group, "model": m.name, "truth": label,
                   "read": read, "boxes": len(crops), "count_ok": len(crops) == len(label)}
            rows.append(row)
        by_image += rows
        results[m.key][tag] = number_metrics(rows)
        for g in sorted({r["group"] for r in rows if r["group"]}):
            results[m.key][f"{tag}_{g}"] = number_metrics([r for r in rows if r["group"] == g])
    r = results[models[-1].key][tag]
    print(f"  {len(items)} {tag}: digit count right {r['count_ok']:.1%}")
    return {m.key: np.array([r["read"] == r["truth"] for r in by_image
                             if r["set"] == tag and r["model"] == m.name]) for m in models}


# ---------------------------------------------------------------- outputs

def pct(x: float) -> str:
    return f"{100 * x:.1f} %"


def write_tables(models, results, per_class, sig, quick: bool):
    RESULTS.mkdir(parents=True, exist_ok=True)
    fields = ["model", "family", "size", "mnist_acc", "mnist_shifted_acc", "mnist_macro_f1",
              "real_digit_acc", "real_numbers_digit_acc", "real_numbers_exact",
              "real_numbers_digit_acc_count_ok", "generated_digit_acc", "generated_exact",
              "generated_digit_acc_count_ok", "digit_count_ok_real", "digit_count_ok_generated"]
    with open(RESULTS / "model_comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for m in models:
            r = results[m.key]
            w.writerow({"model": m.name, "family": m.family, "size": m.params,
                        "mnist_acc": r["mnist_acc"], "mnist_shifted_acc": r["mnist_shifted_acc"],
                        "mnist_macro_f1": r["mnist_macro_f1"], "real_digit_acc": r["real_digit_acc"],
                        "real_numbers_digit_acc": r["real_numbers"]["digit_acc"],
                        "real_numbers_exact": r["real_numbers"]["exact"],
                        "real_numbers_digit_acc_count_ok": r["real_numbers"]["digit_acc_count_ok"],
                        "generated_digit_acc": r["generated"]["digit_acc"],
                        "generated_exact": r["generated"]["exact"],
                        "generated_digit_acc_count_ok": r["generated"]["digit_acc_count_ok"],
                        "digit_count_ok_real": r["real_numbers"]["count_ok"],
                        "digit_count_ok_generated": r["generated"]["count_ok"]})
    with open(RESULTS / "mnist_per_class.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_class[0]))
        w.writeheader()
        w.writerows(per_class)
    with open(RESULTS / "significance.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sig[0]))
        w.writeheader()
        w.writerows(sig)

    def size(m):
        if m.family == "classical":
            return f"{m.params:,} {'support vectors' if 'svm' in m.key else 'tree nodes'}"
        return f"{m.params / 1e6:.2f} M parameters"

    md = ["# Model comparison – Tasks 3 and 4", "",
          "Generated by `python evaluation/evaluate_all.py`." + (" **QUICK RUN – not for the report.**" if quick else ""),
          "Every model reads the same 28×28 inputs: the system's own segmentation (repaired contours: touching digits cut where `cnn_best` reads the pieces best) and "
          "preprocessing (`otsu_mnist_box`).", "",
          "## 1. Summary", "",
          "| Model | Family | Size | MNIST test | MNIST shifted | Real digits (149) | Real numbers: per digit | Real numbers: whole number | Generated: per digit | Generated: whole number |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for m in models:
        r = results[m.key]
        bold = (lambda s: f"**{s}**") if m.key == FINAL_MODEL else (lambda s: s)
        md.append(f"| {bold(m.name)} | {m.family} | {size(m)} | {pct(r['mnist_acc'])} | "
                  f"{pct(r['mnist_shifted_acc'])} | {pct(r['real_digit_acc'])} | "
                  f"{pct(r['real_numbers']['digit_acc'])} | {pct(r['real_numbers']['exact'])} | "
                  f"{pct(r['generated']['digit_acc'])} | {pct(r['generated']['exact'])} |")
    md += ["", "*Per digit* = 1 − edit distance ÷ number of true digits (a missing, extra or wrong digit "
           "is one error). *Whole number* = every digit right. These are the two Task 4 metrics and are "
           "kept separate.", "",
           "## 2. 95 % confidence intervals on the real photos", "",
           "| Model | Real digits | Real numbers, whole number |", "|---|---|---|"]
    for m in models:
        r = results[m.key]
        lo, hi = r["real_digit_ci"]
        elo, ehi = r["real_numbers"]["exact_ci"]
        md.append(f"| {m.name} | {pct(r['real_digit_acc'])} ({100*lo:.0f}–{100*hi:.0f}) | "
                  f"{pct(r['real_numbers']['exact'])} ({100*elo:.0f}–{100*ehi:.0f}) |")
    md += ["", "## 3. Recognition only: numbers that segmentation cut correctly", "",
           "Per-digit accuracy counted only on numbers split into the right number of boxes, so "
           "segmentation errors are taken out and the models are compared on recognition alone.", "",
           "| Model | Real numbers: per digit | Real numbers: whole number | Generated: per digit | Generated: whole number |",
           "|---|---|---|---|---|"]
    for m in models:
        r = results[m.key]
        md.append(f"| {m.name} | {pct(r['real_numbers']['digit_acc_count_ok'])} | "
                  f"{pct(r['real_numbers']['exact_count_ok'])} | {pct(r['generated']['digit_acc_count_ok'])} | "
                  f"{pct(r['generated']['exact_count_ok'])} |")
    md += ["", "## 4. Is the selected model really better? (McNemar's exact test vs VGG-deep)", "",
           "Same images, paired: *VGG only* = images VGG-deep gets right and the other model gets wrong.", "",
           "| Model | Test set | VGG only right | Other only right | p-value | Verdict (α = 0.05) |",
           "|---|---|---|---|---|---|"]
    for s in sig:
        verdict = ("VGG-deep better" if s["p_value"] < 0.05 and s["ref_only_right"] > s["other_only_right"]
                   else "other better" if s["p_value"] < 0.05 else "no significant difference")
        md.append(f"| {s['model']} | {s['test_set']} | {s['ref_only_right']} | {s['other_only_right']} | "
                  f"{s['p_value']:.2g} | {verdict} |")
    seg = results[models[0].key]
    md += ["", "## 5. Segmentation sets the ceiling (same for every model)", "",
           "| Test set | Numbers | Digit count right | Best model's whole-number accuracy |", "|---|---|---|---|"]
    for tag, label in (("real_numbers", "Real photos, all"), ("real_numbers_separate", "Real, digits apart"),
                       ("real_numbers_touching", "Real, digits touching"),
                       ("generated", "Generated, all spacings"),
                       ("generated_wide", "Generated, wide"), ("generated_tight", "Generated, tight"),
                       ("generated_touching", "Generated, touching")):
        if tag in seg:
            best = max(results[m.key][tag]["exact"] for m in models)
            md.append(f"| {label} | {seg[tag]['n']} | {pct(seg[tag]['count_ok'])} | {pct(best)} |")
    md += ["", "## 6. Generated numbers by spacing (whole-number accuracy)", "",
           "| Model | Wide | Tight | Touching |", "|---|---|---|---|"]
    for m in models:
        r = results[m.key]
        md.append(f"| {m.name} | " + " | ".join(pct(r[f"generated_{g}"]["exact"])
                                                for g in ("wide", "tight", "touching")) + " |")
    md += ["", "## 7. Precision, recall and F1 per digit (MNIST test, selected model)", "",
           "| Digit | Precision | Recall | F1 |", "|---|---|---|---|"]
    final = next(m.name for m in models if m.key == FINAL_MODEL)
    for row in per_class:
        if row["model"] == final:
            md.append(f"| {row['digit']} | {pct(row['precision'])} | {pct(row['recall'])} | {pct(row['f1'])} |")
    md += ["", "Every model's per-digit scores: `mnist_per_class.csv`. Confusion matrices: `confusion_mnist.png`."]
    (RESULTS / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def chart(models, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("MNIST test", lambda r: r["mnist_acc"]),
              ("MNIST shifted", lambda r: r["mnist_shifted_acc"]),
              ("Real single digits", lambda r: r["real_digit_acc"]),
              ("Real numbers, whole number", lambda r: r["real_numbers"]["exact"]),
              ("Generated numbers, whole number", lambda r: r["generated"]["exact"])]
    fig, axes = plt.subplots(1, len(panels), figsize=(17, 3.6), sharey=True, facecolor=SURFACE)
    names = [m.name for m in models]
    for ax, (title, get) in zip(axes, panels):
        vals = [100 * get(results[m.key]) for m in models]
        colors = [HIGHLIGHT if m.key == FINAL_MODEL else NEUTRAL for m in models]
        ax.barh(range(len(models)), vals, color=colors, height=0.62)
        for i, v in enumerate(vals):
            ax.text(v + 1, i, f"{v:.1f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 112)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        _style(ax)
    axes[0].set_yticks(range(len(models)))
    axes[0].set_yticklabels(names, fontsize=9, color=INK)
    axes[0].invert_yaxis()
    fig.suptitle("Accuracy (%) of every model on every test set – blue = selected model (VGG-deep)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "comparison_chart.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


def confusion_chart(models, confusions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    cols = 3
    rows = int(np.ceil(len(models) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), facecolor=SURFACE)
    for ax, m in zip(np.ravel(axes), models):
        cm = confusions[m.key]
        errors = cm.copy()
        np.fill_diagonal(errors, 0)                 # colour = errors, so the mistakes stand out
        ax.imshow(errors, cmap=cmap, vmin=0, vmax=max(1, errors.max()))
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                            color="white" if cm[i, j] > errors.max() * 0.55 else INK)
        acc = np.trace(cm) / cm.sum()
        ax.set_title(f"{m.name}  ({100 * acc:.2f} %)", fontsize=10, color=INK, loc="left")
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xlabel("predicted", color=MUTED, fontsize=8)
        ax.set_ylabel("true", color=MUTED, fontsize=8)
        ax.tick_params(labelsize=8, colors=MUTED, length=0)
    for ax in np.ravel(axes)[len(models):]:
        ax.axis("off")
    fig.suptitle("MNIST test errors per model (true digit → predicted digit; diagonal hidden; "
                 "colour scale per panel - compare the numbers, not the shades)",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(RESULTS / "confusion_mnist.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description="Score every model on every test set.")
    ap.add_argument("--models", nargs="+", help="only these keys (default: all six)")
    ap.add_argument("--quick", action="store_true", help="1,000 MNIST test images; for checking the code")
    args = ap.parse_args(argv)

    from generated_set import load as load_generated
    from test_set import load_numbers
    models = load_models(args.models)
    if FINAL_MODEL not in [m.key for m in models]:
        raise SystemExit("The selected model (cnn_best) must be included - it is the reference.")
    results: Dict = defaultdict(dict)
    per_class: List[Dict] = []
    by_image: List[Dict] = []

    print("A. MNIST test set")
    mnist_ok, confusions = mnist_block(models, results, per_class, args.quick)
    print("B. Real single digits")
    digit_ok = real_digits_block(models, results)
    print("C. Real numbers")
    real_ok = numbers_block(models, [(n, i, l, "touching" if "_t" in n else "separate")
                                     for n, i, l in load_numbers()],
                            "real_numbers", results, by_image)
    print("D. Generated numbers (MNIST test digits)")
    gen_ok = numbers_block(models, load_generated(), "generated", results, by_image)

    sig = []
    ref = FINAL_MODEL
    for m in models:
        if m.key == ref:
            continue
        for name, ok in (("MNIST test", mnist_ok), ("real digits", digit_ok),
                         ("real numbers", real_ok), ("generated numbers", gen_ok)):
            sig.append({"model": m.name, "test_set": name, **mcnemar(ok[ref], ok[m.key])})

    write_tables(models, results, per_class, sig, args.quick)
    with open(RESULTS / "numbers_by_image.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(by_image[0]))
        w.writeheader()
        w.writerows(by_image)
    chart(models, results)
    confusion_chart(models, confusions)
    print(f"-> {(RESULTS / 'model_comparison.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
