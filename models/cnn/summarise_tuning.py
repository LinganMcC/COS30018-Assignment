
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).resolve().parent.parent


def load_summaries(root: Path = MODELS_DIR) -> List[dict]:
    """Load all run summaries, sorted by run_id."""
    files = sorted((root / "experiments" / "tuning").glob("*_summary.json"))
    return [json.loads(f.read_text()) for f in files]


def pick_best(summaries: List[dict]) -> Optional[dict]:
    """Best by val accuracy, then shifted-test accuracy, then speed."""
    if not summaries:
        return None
    return max(summaries, key=lambda s: (round(s["val_accuracy"], 4),
                                         s["robust_test_accuracy"],
                                         -s["training_time_s"]))


def markdown_table(summaries: List[dict], best: Optional[dict]) -> str:
    header = ("| Run | Arch | Aug | LR | Batch | Dropout | Epochs | Val acc | "
              "Test acc | Shifted-test acc | Time (s) |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|\n")
    rows = []
    for s in summaries:
        c = s["config"]
        mark = " **(best)**" if best is s else ""
        rows.append(
            f"| {c['run_id']}{mark} | {c['arch']} | {c['augmentation']} | "
            f"{c['learning_rate']:g} | {c['batch_size']} | {c['dropout']} | "
            f"{s['epochs_run']} | {s['val_accuracy']*100:.2f}% | "
            f"{s['test_accuracy']*100:.2f}% | {s['robust_test_accuracy']*100:.2f}% | "
            f"{s['training_time_s']:.0f} |")
    return header + "\n".join(rows) + "\n"


def comparison_chart(summaries: List[dict], out_path: Path) -> None:
    """Grouped bars: val / test / shifted-test accuracy for every run."""
    names = [s["config"]["run_id"] for s in summaries]
    series = [("Val", "val_accuracy"), ("Test", "test_accuracy"),
              ("Shifted test", "robust_test_accuracy")]
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.9), 4.5))
    for i, (label, key) in enumerate(series):
        xs = [j + (i - 1) * width for j in range(len(names))]
        ax.bar(xs, [s[key] * 100 for s in summaries], width, label=label)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)")
    lowest = min(min(s[k] for _, k in series) for s in summaries) * 100
    ax.set_ylim(max(0, lowest - 2), 100)
    ax.set_title("Sprint 3 tuning runs")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def freeze(best: dict, root: Path = MODELS_DIR) -> Path:
    """Copy the winning checkpoint + label map to checkpoints/cnn_best.*"""
    src = Path(best["checkpoint"])
    dst = root / "checkpoints" / "cnn_best.keras"
    shutil.copy2(src, dst)
    shutil.copy2(src.with_suffix(".labels.json"),
                 root / "checkpoints" / "cnn_best.labels.json")
    (root / "checkpoints" / "cnn_best.source.json").write_text(json.dumps(
        {"source_run": best["config"]["run_id"], "summary": best}, indent=2))
    return dst


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Summarise Sprint 3 tuning runs.")
    p.add_argument("--freeze", action="store_true",
                   help="Copy the best run to checkpoints/cnn_best.keras")
    args = p.parse_args(argv)

    summaries = load_summaries()
    if not summaries:
        print("No runs found in experiments/tuning/ - run trainer.py first.")
        return
    best = pick_best(summaries)

    table = markdown_table(summaries, best)
    (MODELS_DIR / "experiments" / "tuning_table.md").write_text(table)
    comparison_chart(summaries, MODELS_DIR / "experiments" / "tuning_comparison.png")
    print(table)
    print(f"Best by val accuracy: {best['config']['run_id']}")

    if args.freeze:
        print(f"Frozen to {freeze(best)}")


if __name__ == "__main__":
    main()
