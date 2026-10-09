
import argparse
import json
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # Save figures without needing a display.
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import callbacks

from augmentation import build_augmenter, get_preset, make_shifted_test_set
from datasets import DatasetSplits, load_dataset
from experiment_logger import log_run
from model_zoo import DEFAULT_DROPOUT, build_model


MODELS_DIR = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    run_id: str
    arch: str = "vgg_deep"
    dataset: str = "mnist"
    augmentation: str = "none"         # preset name, see augmentation.PRESETS
    learning_rate: float = 1e-3
    batch_size: int = 128
    dropout: Optional[float] = None    # None -> architecture default
    max_epochs: int = 30
    patience: int = 5                  # EarlyStopping patience
    seed: int = 42
    notes: str = ""
    # not hyperparameters; control where/how the run is executed ---
    quick: bool = False                # small subset + 2 epochs, no CSV log
    output_root: Optional[str] = None  # override MODELS_DIR (used by tests)

    def resolved_dropout(self) -> float:
        return DEFAULT_DROPOUT[self.arch] if self.dropout is None else self.dropout

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        """Build from a dict, rejecting unknown keys (catches typos in JSON)."""
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)


@dataclass
class RunPaths:
    checkpoint: Path
    labels: Path
    history_json: Path
    history_png: Path
    summary: Path
    log_csv: Optional[Path]

    @classmethod
    def for_config(cls, cfg: TrainConfig) -> "RunPaths":
        root = Path(cfg.output_root) if cfg.output_root else MODELS_DIR
        sub = "tuning_quick" if cfg.quick else "tuning"
        ckpt_dir = root / "checkpoints" / sub
        exp_dir = root / "experiments" / sub
        return cls(
            checkpoint=ckpt_dir / f"{cfg.run_id}.keras",
            labels=ckpt_dir / f"{cfg.run_id}.labels.json",
            history_json=exp_dir / f"{cfg.run_id}_history.json",
            history_png=exp_dir / f"{cfg.run_id}_history.png",
            summary=exp_dir / f"{cfg.run_id}_summary.json",
            # Quick runs are smoke tests - keep them out of the real log.
            log_csv=None if cfg.quick else root / "experiments" / "tuning_log.csv",
        )


def make_train_dataset(x, y, batch_size: int, augmentation: str,
                       seed: int) -> tf.data.Dataset:
    """Shuffled, batched tf.data pipeline with augmentation applied per batch.

    Augmentation runs in the input pipeline (training data only), so the
    saved model contains no augmentation layers.
    """
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.shuffle(buffer_size=min(len(x), 20_000), seed=seed,
                    reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    augmenter = build_augmenter(get_preset(augmentation), seed=seed)
    if augmenter is not None:
        ds = ds.map(lambda xb, yb: (augmenter(xb, training=True), yb),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


def save_history_plot(history: dict, title: str, out_path: Path) -> None:
    """Loss and accuracy curves side by side (same style as Sprint 2)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, name in [(ax_loss, "loss", "Loss"), (ax_acc, "accuracy", "Accuracy")]:
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# Core training function


def train(cfg: TrainConfig, data: Optional[DatasetSplits] = None) -> dict:
    """Train one model according to `cfg` and return its summary dict.

    Args:
        cfg:  the run configuration.
        data: optional pre-loaded dataset (lets a grid load MNIST once and
              lets tests inject a tiny synthetic dataset).
    """
    tf.keras.utils.set_random_seed(cfg.seed)  # Python, NumPy and TF seeds.
    paths = RunPaths.for_config(cfg)
    dropout = cfg.resolved_dropout()

    # Data.
    if data is None:
        data = load_dataset(cfg.dataset)
    if cfg.quick:
        data = data.subsample(n_train=5_000, n_val=1_000, n_test=2_000, seed=cfg.seed)
    max_epochs = 2 if cfg.quick else cfg.max_epochs

    print(f"\n=== {cfg.run_id} | arch={cfg.arch} aug={cfg.augmentation} "
          f"lr={cfg.learning_rate} bs={cfg.batch_size} dropout={dropout} "
          f"classes={data.num_classes} ===")

    train_ds = make_train_dataset(data.x_train, data.y_train, cfg.batch_size,
                                  cfg.augmentation, cfg.seed)

    # Model.
    model = build_model(cfg.arch, num_classes=data.num_classes, dropout=dropout)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    # Callbacks same policy as the Sprint 2 models so runs are comparable.
    cbs = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=cfg.patience,
                                restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_accuracy", factor=0.5,
                                    patience=2, min_lr=1e-5, verbose=1),
    ]

    # Train.
    start = time.time()
    history = model.fit(train_ds, validation_data=(data.x_val, data.y_val),
                        epochs=max_epochs, callbacks=cbs, verbose=2)
    training_time_s = time.time() - start
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}

    # Evaluate. Test set is touched once, after training. Model SELECTION
    #    must use val_accuracy (see summarise_tuning.py), never test accuracy.
    _, test_acc = model.evaluate(data.x_test, data.y_test, verbose=0)
    x_shifted = make_shifted_test_set(data.x_test, seed=0)
    _, robust_acc = model.evaluate(x_shifted, data.y_test, verbose=0)
    val_acc = max(hist["val_accuracy"])
    epochs_run = len(hist["loss"])

    # 6. Save model, label map, curves, summary.
    paths.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    paths.summary.parent.mkdir(parents=True, exist_ok=True)
    model.save(paths.checkpoint)
    paths.labels.write_text(json.dumps(
        {"label_names": data.label_names, "dataset": cfg.dataset}, indent=2))
    paths.history_json.write_text(json.dumps(hist, indent=2))
    save_history_plot(hist, f"{cfg.arch} - {cfg.run_id}", paths.history_png)

    summary = {
        "config": {**asdict(cfg), "dropout": dropout, "max_epochs": max_epochs},
        "val_accuracy": val_acc,
        "test_accuracy": float(test_acc),
        "robust_test_accuracy": float(robust_acc),
        "epochs_run": epochs_run,
        "training_time_s": training_time_s,
        "num_params": int(model.count_params()),
        "checkpoint": str(paths.checkpoint),
    }
    paths.summary.write_text(json.dumps(summary, indent=2))

    # Append to the shared-schema CSV. Extra info (augmentation, robust
    #    accuracy) goes in `notes` so the agreed column set is not changed.
    if paths.log_csv is not None:
        log_run(
            run_id=cfg.run_id,
            architecture=cfg.arch,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            epochs=epochs_run,
            dropout=dropout,
            val_accuracy=val_acc,
            test_accuracy=float(test_acc),
            training_time_s=training_time_s,
            notes=(f"dataset={cfg.dataset}; aug={cfg.augmentation}; "
                   f"robust_acc={robust_acc:.4f}; {cfg.notes}").strip("; "),
            log_path=paths.log_csv,
        )

    print(f"--> val {val_acc:.4f} | test {test_acc:.4f} | shifted-test "
          f"{robust_acc:.4f} | {epochs_run} epochs | {training_time_s:.0f}s")
    return summary



# Grid support


def load_grid(path: Path, stage: Optional[str] = None,
              only: Optional[List[str]] = None) -> List[TrainConfig]:
    spec = json.loads(Path(path).read_text())
    base = spec.get("defaults", {})
    configs = []
    for stage_name, stage_spec in spec["stages"].items():
        if stage and stage_name != stage:
            continue
        stage_defaults = stage_spec.get("defaults", {})
        for run in stage_spec["runs"]:
            if only and run["run_id"] not in only:
                continue
            merged = {**base, **stage_defaults, **run}
            merged.pop("comment", None)  # comments are allowed in the JSON
            configs.append(TrainConfig.from_dict(merged))
    if not configs:
        raise ValueError("No runs selected - check --stage / --only.")
    return configs


def run_many(configs: List[TrainConfig], skip_existing: bool = False) -> List[dict]:
    """Train several configs, loading each dataset only once."""
    cache = {}
    results = []
    for i, cfg in enumerate(configs, 1):
        if skip_existing and RunPaths.for_config(cfg).summary.exists():
            print(f"[{i}/{len(configs)}] skip {cfg.run_id} (already done)")
            continue
        print(f"[{i}/{len(configs)}] {cfg.run_id}")
        if cfg.dataset not in cache:
            cache[cfg.dataset] = load_dataset(cfg.dataset)
        results.append(train(cfg, data=cache[cfg.dataset]))
    return results


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Config-driven CNN trainer.")
    p.add_argument("--grid", type=Path, help="Grid JSON file (runs many configs).")
    p.add_argument("--stage", help="Only run this stage of the grid (e.g. A).")
    p.add_argument("--only", nargs="+", help="Only run these run_ids from the grid.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip runs whose summary JSON already exists.")
    p.add_argument("--quick", action="store_true",
                   help="Smoke test: small subset, 2 epochs, no CSV logging.")
    # Single-run options (ignored when --grid is used).
    p.add_argument("--run-id", default="single_run")
    p.add_argument("--arch", default="vgg_deep")
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--aug", default="none", dest="augmentation")
    p.add_argument("--lr", type=float, default=1e-3, dest="learning_rate")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--epochs", type=int, default=30, dest="max_epochs")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    if args.grid:
        configs = load_grid(args.grid, stage=args.stage, only=args.only)
    else:
        configs = [TrainConfig(
            run_id=args.run_id, arch=args.arch, dataset=args.dataset,
            augmentation=args.augmentation, learning_rate=args.learning_rate,
            batch_size=args.batch_size, dropout=args.dropout,
            max_epochs=args.max_epochs, seed=args.seed)]
    for cfg in configs:
        cfg.quick = args.quick
    run_many(configs, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
