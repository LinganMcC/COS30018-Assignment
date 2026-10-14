from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # save figures without a display
import matplotlib.pyplot as plt
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent
CHECKPOINTS = MODELS_DIR / "checkpoints"
EVAL_DIR = MODELS_DIR / "experiments" / "eval"

# Readable names for the operator classes in figures and tables.
DISPLAY = {"+": "+", "-": "−", "*": "×", "/": "÷", "(": "(", ")": ")"}


# ---------------------------------------------------------------------------
# Metrics (pure NumPy so they can be unit-tested without TensorFlow)
# ---------------------------------------------------------------------------

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     num_classes: int) -> np.ndarray:
    """cm[i, j] = number of class-i images predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true, int), np.asarray(y_pred, int)), 1)
    return cm


def per_class_scores(cm: np.ndarray) -> Dict[str, np.ndarray]:
    """Precision, recall, F1 and support for every class from a confusion matrix.

    A class that is never predicted gets precision 0 (not NaN), so it shows up
    as a problem in the table instead of disappearing from averages.
    """
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    support = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(tp), where=denom > 0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "support": support.astype(int)}


def top_confusions(cm: np.ndarray, labels: List[str],
                   k: int = 10) -> List[Tuple[str, str, int, float]]:
    """The k largest off-diagonal cells as (true, predicted, count, % of true class)."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    order = np.argsort(off, axis=None)[::-1]
    out = []
    for flat in order[:k]:
        i, j = np.unravel_index(flat, off.shape)
        if off[i, j] == 0:
            break
        out.append((labels[i], labels[j], int(off[i, j]),
                    100.0 * off[i, j] / max(1, cm[i].sum())))
    return out


def group_accuracy(cm: np.ndarray, idx: List[int]) -> float:
    """Accuracy over a subset of TRUE classes (e.g. digits only, symbols only)."""
    correct = np.diag(cm)[idx].sum()
    total = cm[idx].sum()           # every image whose true class is in idx
    return float(correct) / max(1, total)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_frozen(name: str):
    """Load checkpoints/<name>.keras with its label file."""
    import tensorflow as tf
    path = CHECKPOINTS / f"{name}.keras"
    meta = json.loads((CHECKPOINTS / f"{name}.labels.json").read_text())
    return tf.keras.models.load_model(str(path)), meta["label_names"], meta["dataset"]


def predict(model, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    return model.predict(x, batch_size=batch_size, verbose=0).argmax(axis=1)


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def save_confusion_png(cm: np.ndarray, labels: List[str], title: str,
                       out_path: Path) -> None:
    """Row-normalised confusion matrix: each row shows where class i went (%).

    Normalising by row matters here because the classes are unbalanced
    (about 1,000 test images per digit, 120 for ÷). Raw counts would make
    every symbol row look empty.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = cm.sum(axis=1, keepdims=True)
    pct = np.divide(100.0 * cm, rows, out=np.zeros(cm.shape), where=rows > 0)
    names = [DISPLAY.get(l, l) for l in labels]
    n = len(labels)

    fig, ax = plt.subplots(figsize=(0.55 * n + 2.5, 0.55 * n + 2))
    ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
    for i in range(n):
        for j in range(n):
            if cm[i, j] == 0:
                continue
            on_diag = i == j
            text = f"{pct[i, j]:.1f}" if on_diag else str(cm[i, j])
            ax.text(j, i, text, ha="center", va="center", fontsize=7,
                    color="white" if pct[i, j] > 60 else ("black" if on_diag else "#b2182b"))
    ax.set_xticks(range(n), names)
    ax.set_yticks(range(n), names)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title(title, fontsize=10)
    if n > 10:  # separate digits from symbols
        for pos in (9.5,):
            ax.axhline(pos, color="grey", lw=0.8)
            ax.axvline(pos, color="grey", lw=0.8)
    fig.text(0.5, 0.005, "Diagonal: recall %   Off-diagonal: image count (red)",
             ha="center", fontsize=7, color="grey")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_per_class_csv(path: Path, labels: List[str], clean: dict,
                        shifted: Optional[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        head = ["class", "support", "precision", "recall", "f1"]
        if shifted:
            head += ["shifted_precision", "shifted_recall", "shifted_f1"]
        w.writerow(head)
        for i, label in enumerate(labels):
            row = [label, clean["support"][i]] + [f"{clean[k][i]:.4f}"
                                                  for k in ("precision", "recall", "f1")]
            if shifted:
                row += [f"{shifted[k][i]:.4f}" for k in ("precision", "recall", "f1")]
            w.writerow(row)


def write_report_md(path: Path, name: str, labels: List[str], results: dict) -> None:
    """A short Markdown summary that can be pasted straight into the report."""
    clean = results["clean"]
    shifted = results.get("shifted")
    lines = [f"# Per-class evaluation – `{name}`", ""]
    lines.append("| Split | Overall | Digits 0–9 | Symbols | Macro F1 |")
    lines.append("|---|---|---|---|---|")
    for split_name, r in (("Clean test", clean), ("Shifted test", shifted)):
        if r is None:
            continue
        sym = f"{100 * r['symbol_acc']:.2f} %" if r["symbol_acc"] is not None else "–"
        lines.append(f"| {split_name} | {100 * r['overall_acc']:.2f} % | "
                     f"{100 * r['digit_acc']:.2f} % | {sym} | {r['macro_f1']:.4f} |")
    lines += ["", "| Class | Support | Precision | Recall | F1 |"
              + (" Shifted recall |" if shifted else ""),
              "|---|---|---|---|---|" + ("---|" if shifted else "")]
    for i, label in enumerate(labels):
        s = clean["scores"]
        row = (f"| {DISPLAY.get(label, label)} | {s['support'][i]} | "
               f"{100 * s['precision'][i]:.2f} % | {100 * s['recall'][i]:.2f} % | "
               f"{s['f1'][i]:.4f} |")
        if shifted:
            row += f" {100 * shifted['scores']['recall'][i]:.2f} % |"
        lines.append(row)
    for split_name, r in (("clean", clean), ("shifted", shifted)):
        if r is None:
            continue
        lines += ["", f"**Most frequent confusions ({split_name} test)**", "",
                  "| True | Predicted as | Count | % of true class |", "|---|---|---|---|"]
        for t, p, c, pc in r["confusions"]:
            lines.append(f"| {DISPLAY.get(t, t)} | {DISPLAY.get(p, p)} | {c} | {pc:.2f} % |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def summarise(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str]) -> dict:
    n = len(labels)
    cm = confusion_matrix(y_true, y_pred, n)
    scores = per_class_scores(cm)
    digit_idx = [i for i, l in enumerate(labels) if l.isdigit()]
    symbol_idx = [i for i, l in enumerate(labels) if not l.isdigit()]
    return {
        "cm": cm,
        "scores": scores,
        "overall_acc": float(np.trace(cm)) / max(1, cm.sum()),
        "digit_acc": group_accuracy(cm, digit_idx),
        "symbol_acc": group_accuracy(cm, symbol_idx) if symbol_idx else None,
        "macro_f1": float(scores["f1"].mean()),
        "confusions": top_confusions(cm, labels),
    }


def evaluate(name: str, shifted: bool = True, out_dir: Path = EVAL_DIR) -> dict:
    from augmentation import make_shifted_test_set
    from datasets import load_dataset

    model, labels, dataset = load_frozen(name)
    data = load_dataset(dataset)
    if list(data.label_names) != list(labels):
        raise ValueError(f"{name}: label file {labels} does not match dataset "
                         f"'{dataset}' classes {data.label_names}")

    results = {"clean": summarise(data.y_test, predict(model, data.x_test), labels)}
    if shifted:
        x_shift = make_shifted_test_set(data.x_test, seed=0)
        results["shifted"] = summarise(data.y_test, predict(model, x_shift), labels)

    out_dir.mkdir(parents=True, exist_ok=True)
    write_per_class_csv(out_dir / f"{name}_per_class.csv", labels,
                        results["clean"]["scores"],
                        results["shifted"]["scores"] if shifted else None)
    save_confusion_png(results["clean"]["cm"], labels,
                       f"{name} – clean test ({results['clean']['cm'].sum():,} images)",
                       out_dir / f"{name}_confusion.png")
    if shifted:
        save_confusion_png(results["shifted"]["cm"], labels,
                           f"{name} – shifted test (rotated / shifted / zoomed)",
                           out_dir / f"{name}_confusion_shifted.png")
    write_report_md(out_dir / f"{name}_report.md", name, labels, results)

    for split in ("clean", "shifted"):
        if split in results:
            r = results[split]
            sym = f"{100 * r['symbol_acc']:.2f}%" if r["symbol_acc"] is not None else "-"
            print(f"{name:9s} {split:8s} overall {100 * r['overall_acc']:.2f}% | "
                  f"digits {100 * r['digit_acc']:.2f}% | symbols {sym} | "
                  f"macro-F1 {r['macro_f1']:.4f}")
    print(f"  -> {out_dir}")
    return results


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Per-class evaluation of frozen models.")
    p.add_argument("--model", nargs="+", default=["ext_best", "cnn_best"],
                   help="Checkpoint names in models/checkpoints/ (without .keras).")
    p.add_argument("--no-shifted", action="store_true",
                   help="Skip the shifted test split (faster).")
    args = p.parse_args(argv)
    for name in args.model:
        evaluate(name, shifted=not args.no_shifted)


if __name__ == "__main__":
    main()
