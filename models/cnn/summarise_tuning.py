
import argparse
import json
import shutil
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODELS_DIR = Path(__file__).resolve().parent.parent

# Runs are only ever compared with runs on the SAME dataset: a 16-class
# extension run and a 10-class digit run are different tasks with different
# test sets. Each dataset gets its own table, chart and frozen model.
#   dataset -> (prefix for table/chart, name of the frozen model)
OUTPUT_NAMES = {
    "mnist": ("tuning", "cnn_best"),
    "mnist+symbols": ("extension", "ext_best"),
}


def output_names(dataset: str) -> tuple:
    slug = dataset.replace("+", "_")
    return OUTPUT_NAMES.get(dataset, (f"tuning_{slug}", f"{slug}_best"))


def load_summaries(root: Path = MODELS_DIR, dataset: Optional[str] = None) -> List[dict]:
    """Load run summaries sorted by run_id; only one dataset if given."""
    files = sorted((root / "experiments" / "tuning").glob("*_summary.json"))
    summaries = [json.loads(f.read_text()) for f in files]
    if dataset is not None:
        summaries = [s for s in summaries
                     if s["config"].get("dataset", "mnist") == dataset]
    return summaries


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


def comparison_chart(summaries: List[dict], out_path: Path,
                     title: str = "Sprint 3 tuning runs") -> None:
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
    ax.set_title(title)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def freeze(best: dict, root: Path = MODELS_DIR, name: str = "cnn_best") -> Path:
    """Copy the winning checkpoint + label map to checkpoints/<name>.*"""
    src = Path(best["checkpoint"])
    dst = root / "checkpoints" / f"{name}.keras"
    shutil.copy2(src, dst)
    shutil.copy2(src.with_suffix(".labels.json"),
                 root / "checkpoints" / f"{name}.labels.json")
    (root / "checkpoints" / f"{name}.source.json").write_text(json.dumps(
        {"source_run": best["config"]["run_id"], "summary": best}, indent=2))
    return dst


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Summarise Sprint 3 tuning runs.")
    p.add_argument("--dataset", default="mnist",
                   help="which runs to compare: mnist (digits) or mnist+symbols (extension)")
    p.add_argument("--freeze", action="store_true",
                   help="copy the best run to checkpoints/cnn_best.keras "
                        "(ext_best.keras for mnist+symbols)")
    args = p.parse_args(argv)

    summaries = load_summaries(dataset=args.dataset)
    if not summaries:
        print(f"No '{args.dataset}' runs found in experiments/tuning/ - run trainer.py first.")
        return
    best = pick_best(summaries)
    prefix, model_name = output_names(args.dataset)

    table = markdown_table(summaries, best)
    (MODELS_DIR / "experiments" / f"{prefix}_table.md").write_text(table)
    comparison_chart(summaries, MODELS_DIR / "experiments" / f"{prefix}_comparison.png",
                     title=f"Runs on '{args.dataset}'")
    print(table)
    print(f"Best by val accuracy: {best['config']['run_id']}")

    if args.freeze:
        print(f"Frozen to {freeze(best, name=model_name)}")


if __name__ == "__main__":
    main()
