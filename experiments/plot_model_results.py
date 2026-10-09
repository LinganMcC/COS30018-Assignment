"""Task 2 - one chart comparing the four segmentation models.

Reads reports/segmentation_comparison.csv (written by demo_segmentation.py),
so it takes a second and needs no model or TensorFlow.

Run:
    python experiments/plot_model_results.py

Output:
    a results table in the terminal
    reports/segmentation_models_chart.png
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "reports" / "segmentation_comparison.csv"
OUT = ROOT / "reports" / "segmentation_models_chart.png"

MODELS = [
    ("sliding_mlp", "MLP\n(John)"),
    ("sliding_cnn", "CNN LeNet\n(Thien)"),
    ("sliding_svm", "SVM\n(John)"),
    ("sliding_rf", "Random Forest\n(John)"),
]
LEVELS = [("acc_wide", "wide spacing", "#9ecae1"),
          ("acc_tight", "tight spacing", "#4292c6"),
          ("acc_touching", "touching", "#08519c")]


def bar(fraction: float, width: int = 20) -> str:
    filled = round(fraction * width)
    return "#" * filled + "." * (width - filled)


def print_table(rows: dict) -> None:
    n = rows[MODELS[0][0]]["images_tested"]
    print("Task 2 - four models behind the same sliding-window search")
    print(f"Numbers split into the right count of digits, {n} generated numbers\n")
    print(f"  {'model':24s} {'overall':>8s}   {'wide':>5s} {'tight':>6s} {'touching':>9s}")
    print("  " + "-" * 62)
    for method, label in MODELS:
        r = rows[method]
        name = label.replace("\n", " ")
        print(f"  {name:24s} {float(r['accuracy']):>7.0%}   "
              f"{float(r['acc_wide']):>5.0%} {float(r['acc_tight']):>6.0%} "
              f"{float(r['acc_touching']):>9.0%}   {bar(float(r['accuracy']))}")
    if "contours" in rows:
        r = rows["contours"]
        print("  " + "-" * 62)
        print(f"  {'contours (no model)':24s} {float(r['accuracy']):>7.0%}   "
              f"{float(r['acc_wide']):>5.0%} {float(r['acc_tight']):>6.0%} "
              f"{float(r['acc_touching']):>9.0%}   {bar(float(r['accuracy']))}  <- selected")
    print()


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(CSV, encoding="utf-8") as fh:
        rows = {r["method"]: r for r in csv.DictReader(fh)}
    missing = [m for m, _ in MODELS if m not in rows]
    if missing:
        raise SystemExit(f"{', '.join(missing)} not in {CSV.name}. "
                         f"Run 'python experiments/demo_segmentation.py' first.")

    print_table(rows)

    fig, ax = plt.subplots(figsize=(10, 5.6), dpi=150)
    width = 0.22
    for i, (key, name, colour) in enumerate(LEVELS):
        xs = [m + (i - 1) * width for m in range(len(MODELS))]
        vals = [float(rows[m][key]) * 100 for m, _ in MODELS]
        bars = ax.bar(xs, vals, width, label=name, color=colour)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8)

    for m, (method, _) in enumerate(MODELS):
        overall = float(rows[method]["accuracy"]) * 100
        ax.text(m, 104, f"overall {overall:.0f}%", ha="center", fontsize=10, weight="bold")

    if "contours" in rows:
        ref = float(rows["contours"]["accuracy"]) * 100
        ax.axhline(ref, color="#d62728", ls="--", lw=1.2)
        ax.text(len(MODELS) - 0.55, ref + 1.5, f"contours, no model: {ref:.0f}%",
                color="#d62728", fontsize=9, ha="right")

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([label for _, label in MODELS], fontsize=10)
    ax.set_ylabel("numbers split into the right count of digits (%)")
    ax.set_ylim(0, 112)
    ax.set_title("Task 2 - same sliding-window search, four different models\n"
                 f"({rows[MODELS[0][0]]['images_tested']} generated numbers, "
                 "10 per spacing level)", fontsize=11)
    ax.legend(loc="upper right", fontsize=9, frameon=False, bbox_to_anchor=(1.0, 0.93))
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"Saved: {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
