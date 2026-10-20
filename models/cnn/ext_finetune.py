"""Sprint 4: retrain the extension model for real handwritten expressions.

Starts from the frozen ext_best (VGG-deep, 16 classes) and fine-tunes it on:
  * MNIST + operator symbols (what it was trained on, so nothing is forgotten)
  * synthetic continental 7s (crossed) and 1s (long flag)    - augmentation.py
  * real symbols cut from the team's expression photos, training writers only
    (sheets 1-2, writers 1-6; see real_symbols.py), repeated `real_repeat` times
with geometric augmentation ("light") plus style augmentation (stroke width,
hard / sharpened / blurred edges, contrast) on every batch.

Model selection uses only the validation writers (sheets 1-2, writer 7) and the
MNIST+symbols validation split. Sheets 3-4 are never touched here.

Usage (from models/cnn/):
    python ext_finetune.py --run-id ext_v3                      # full recipe
    python ext_finetune.py --run-id ext_v3_noreal --real-repeat 0   # ablation
    python ext_finetune.py --run-id ext_v3_nostyle --no-style       # ablation

Outputs: models/checkpoints/tuning/<run>.keras + .labels.json,
         models/experiments/tuning/<run>_summary.json / _history.json / _history.png,
         a row in models/experiments/tuning_log.csv
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from augmentation import build_augmenter, cross_sevens, flag_ones, get_preset, style_augment
from datasets import load_dataset
from experiment_logger import log_run
from inference import CHECKPOINTS, resolve_checkpoint
from real_symbols import LABELS, load_real_symbols
from trainer import MODELS_DIR, save_history_plot


def build_training_set(data, real_repeat: int, n_synthetic: int, seed: int):
    rng = np.random.default_rng(seed)
    xs, ys = [data.x_train], [data.y_train]
    for digit, make in ((7, cross_sevens), (1, flag_ones)):
        pool = data.x_train[data.y_train == digit]
        pick = pool[rng.permutation(len(pool))[:n_synthetic]]
        xs.append(make(pick, rng))
        ys.append(np.full(len(pick), digit, data.y_train.dtype))
    if real_repeat:
        xr, yr, _ = load_real_symbols("train")
        xs += [xr] * real_repeat
        ys += [yr.astype(data.y_train.dtype)] * real_repeat
    return np.concatenate(xs).astype("float32"), np.concatenate(ys)


def make_dataset(x, y, batch_size: int, style: bool, seed: int) -> tf.data.Dataset:
    geo = build_augmenter(get_preset("light"), seed=seed)

    def aug(xb, yb):
        xb = geo(xb, training=True)
        if style:
            xb = style_augment(xb)
        return xb, yb

    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(len(x), seed=seed,
                                                            reshuffle_each_iteration=True)
    return ds.batch(batch_size).map(aug, num_parallel_calls=tf.data.AUTOTUNE).prefetch(2)


class RealValidation(tf.keras.callbacks.Callback):
    """Accuracy on the validation writers' real symbols; keeps the best weights."""

    def __init__(self, x, y):
        super().__init__()
        self.x, self.y = x, y
        self.best, self.best_weights, self.best_epoch = -1.0, None, -1

    def on_epoch_end(self, epoch, logs=None):
        pred = self.model.predict(self.x, verbose=0).argmax(1)
        acc = float((pred == self.y).mean())
        logs["val_real_accuracy"] = acc
        # real handwriting first; MNIST+symbols val breaks ties
        score = acc + 1e-3 * logs.get("val_accuracy", 0.0)
        if score > self.best:
            self.best, self.best_weights, self.best_epoch = score, self.model.get_weights(), epoch
        print(f"  val_real_accuracy {acc:.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-id", default="ext_v3")
    ap.add_argument("--start", default="ext_best", help="checkpoint to fine-tune")
    ap.add_argument("--real-repeat", type=int, default=6)
    ap.add_argument("--synthetic", type=int, default=1500, help="crossed 7s and flagged 1s each")
    ap.add_argument("--no-style", action="store_true", help="geometric augmentation only")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="use N training images (smoke test)")
    args = ap.parse_args(argv)

    tf.keras.utils.set_random_seed(args.seed)
    data = load_dataset("mnist+symbols")
    assert data.label_names == LABELS
    x, y = build_training_set(data, args.real_repeat, args.synthetic, args.seed)
    if args.limit:
        idx = np.random.default_rng(0).permutation(len(x))[:args.limit]
        x, y = x[idx], y[idx]
    xv_real, yv_real, _ = load_real_symbols("val")
    print(f"{args.run_id}: {len(x)} training images (real x{args.real_repeat}), "
          f"style={'off' if args.no_style else 'on'}, start={args.start}")

    model = tf.keras.models.load_model(str(resolve_checkpoint(args.start)))
    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    real_val = RealValidation(xv_real, yv_real)
    cbs = [real_val,
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_real_accuracy", mode="max",
                                                factor=0.5, patience=1, min_lr=2e-5, verbose=1)]
    start = time.time()
    hist = model.fit(make_dataset(x, y, args.batch_size, not args.no_style, args.seed),
                     validation_data=(data.x_val, data.y_val), epochs=args.epochs,
                     callbacks=cbs, verbose=2)
    seconds = time.time() - start
    model.set_weights(real_val.best_weights)

    _, test_acc = model.evaluate(data.x_test, data.y_test, verbose=0)
    h = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
    ckpt_dir, exp_dir = CHECKPOINTS / "tuning", MODELS_DIR / "experiments" / "tuning"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    exp_dir.mkdir(parents=True, exist_ok=True)
    ckpt = ckpt_dir / f"{args.run_id}.keras"
    model.save(ckpt)
    ckpt.with_suffix(".labels.json").write_text(json.dumps(
        {"label_names": LABELS, "dataset": "mnist+symbols",   # class set / test data
         "trained_on": "mnist+symbols+real (team handwriting, sheets 1-2)", "input_blur": False}, indent=2))
    (exp_dir / f"{args.run_id}_history.json").write_text(json.dumps(h, indent=2))
    save_history_plot(h, f"vgg_deep fine-tune - {args.run_id}", exp_dir / f"{args.run_id}_history.png")
    summary = {"config": vars(args), "best_epoch": real_val.best_epoch + 1,
               "val_real_accuracy": h["val_real_accuracy"][real_val.best_epoch],
               "val_accuracy": h["val_accuracy"][real_val.best_epoch],
               "test_accuracy": float(test_acc), "training_time_s": seconds,
               "training_images": int(len(x)), "checkpoint": str(ckpt)}
    (exp_dir / f"{args.run_id}_summary.json").write_text(json.dumps(summary, indent=2))
    if not args.limit:
        log_run(args.run_id, "vgg_deep", args.lr, args.batch_size, len(h["loss"]), 0.5,
                float(test_acc), seconds, val_accuracy=summary["val_accuracy"],
                notes=(f"fine-tune of {args.start}; real x{args.real_repeat}; "
                       f"style={'off' if args.no_style else 'on'}; "
                       f"val_real={summary['val_real_accuracy']:.4f}; dataset=mnist+symbols+real"),
                log_path=MODELS_DIR / "experiments" / "tuning_log.csv")
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
