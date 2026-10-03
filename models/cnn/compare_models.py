"""Compare the four final models on one loss chart and one accuracy chart.

Each model script saves its per-epoch Keras history to
models/experiments/<name>_history.json. This script overlays them so the
models can be compared on the same axes (validation = solid, train = dashed).

Run from models/cnn/:  python compare_models.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

EXP_DIR = Path(__file__).parent.parent / "experiments"

# Fixed colour per model (colour-blind-checked categorical order), so a model
# keeps the same colour in every figure of the report.
MODELS = [
    ("mlp_simple",   "MLP (simple)",   "#2a78d6"),
    ("cnn_lenet",    "LeNet-5",        "#eb6834"),
    ("cnn_resnet",   "ResNet (small)", "#1baf7a"),
    ("cnn_vgg_deep", "VGG (deep)",     "#eda100"),
]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_histories():
    hists = {}
    for key, label, colour in MODELS:
        path = EXP_DIR / f"{key}_history.json"
        if not path.exists():
            print(f"[skip] {path.name} not found - run {key}.py first")
            continue
        hists[key] = (label, colour, json.loads(path.read_text()))
    return hists


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=10)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def place_end_labels(fig, ax, ends, label_x, min_gap_px=13):
    """Label each validation line in a column right of the plot, nudging labels
    apart vertically so they never overlap; a thin leader line joins label to line."""
    fig.canvas.draw()
    to_px, to_data = ax.transData, ax.transData.inverted()
    items = sorted(((to_px.transform((x, y))[1], x, y, text) for x, y, text in ends))
    placed = []
    for py, x, y, text in items:                       # bottom -> top in pixels
        if placed and py - placed[-1][0] < min_gap_px:
            py = placed[-1][0] + min_gap_px
        placed.append((py, x, y, text))
    for py, x, y, text in placed:
        ly = to_data.transform((to_px.transform((label_x, y))[0], py))[1]
        ax.annotate(text, xy=(x, y), xytext=(label_x, ly), textcoords="data",
                    va="center", fontsize=8, color=INK,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6,
                                    shrinkA=0, shrinkB=3))


def plot_metric(hists, train_key, val_key, title, ylabel, out_name,
                best=min, log_y=False, ylim=None, fmt="{:.4f}"):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ends = []
    for key, (label, colour, h) in hists.items():
        val = h[val_key]
        epochs = range(1, len(val) + 1)
        ax.plot(epochs, h[train_key], color=colour, linewidth=1.2,
                linestyle="--", alpha=0.55)
        ax.plot(epochs, val, color=colour, linewidth=2, label=label)
        # EarlyStopping restores the best epoch's weights, so mark THAT epoch -
        # it is the model that was actually saved, not the last epoch run.
        b = val.index(best(val))
        ax.plot([b + 1], [val[b]], marker="o", markersize=6, color=colour,
                markeredgecolor="white", markeredgewidth=1.5, zorder=5)
        ends.append((len(val), val[-1], f"{label}  best {fmt.format(val[b])} (ep {b + 1})"))

    if log_y:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)

    style(ax, title, ylabel)
    last_epoch = max(e[0] for e in ends)
    ax.set_xlim(0.5, last_epoch + 14)   # room for the label column
    place_end_labels(fig, ax, ends, label_x=last_epoch + 1.5)
    leg = ax.legend(frameon=False, fontsize=9, loc="best",
                    title="solid = validation, dashed = train, dot = best epoch",
                    title_fontsize=8)
    leg.get_title().set_color(MUTED)
    fig.tight_layout()
    out = EXP_DIR / out_name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    hists = load_histories()
    if not hists:
        return
    plot_metric(hists, "loss", "val_loss",
                "Loss per epoch - 4 final models", "Cross-entropy loss (log scale)",
                "compare_loss.png", best=min, log_y=True)
    plot_metric(hists, "accuracy", "val_accuracy",
                "Accuracy per epoch - 4 final models", "Accuracy",
                "compare_accuracy.png", best=max, ylim=(0.95, 1.0005), fmt="{:.2%}")

    print("\nBest validation accuracy / min validation loss:")
    for key, (label, _, h) in hists.items():
        print(f"  {label:15s} val_acc={max(h['val_accuracy']):.4f}  "
              f"val_loss={min(h['val_loss']):.4f}  epochs={len(h['val_loss'])}")


if __name__ == "__main__":
    main()
