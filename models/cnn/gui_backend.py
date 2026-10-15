"""CNN backend for gui.py (pretrained, train new, recognise)."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

from inference import CHECKPOINTS, REPO_ROOT, Recogniser, resolve_checkpoint

# GUI option -> frozen model used for "Pretrained"
CNN_OPTIONS: Dict[str, str] = {
    "CNN": "cnn_best",               # digits 0-9
    "CNN + operators": "ext_best",   # digits + - * / ( )
}
CNN_ARCHS = ["lenet", "vgg_deep", "resnet"]
AUGMENTATIONS = ["none", "light", "medium", "strong"]
GUI_RUNS = REPO_ROOT / "models" / "gui_runs"   # GUI-trained models (git-ignored)


def is_cnn(model_name: str) -> bool:
    return model_name in CNN_OPTIONS


def default_checkpoint(model_name: str) -> Path:
    return CHECKPOINTS / f"{CNN_OPTIONS[model_name]}.keras"


def load_cnn(model_name: str, path: Optional[str] = None) -> Recogniser:
    """Load `path`, or the frozen model for this option."""
    rec = Recogniser(resolve_checkpoint(path) if path else default_checkpoint(model_name))
    if model_name == "CNN + operators" and not rec.reads_operators():
        raise ValueError(f"{Path(path).name} has no operator classes - "
                         "pick a 16-class model or use the CNN option.")
    return rec


def train_cnn(model_name: str, params: dict,
              on_epoch: Callable[[int, float, float], None],
              stop_event, data=None, output_root: Optional[Path] = None) -> Recogniser:
    """Train with the GUI settings; reports each epoch and honours Stop."""
    import tensorflow as tf
    from trainer import TrainConfig, train

    class _GuiHooks(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            if stop_event.is_set():          # Stop pressed mid-epoch
                self.model.stop_training = True

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            on_epoch(epoch + 1, float(logs.get("loss", float("nan"))),
                     float(logs.get("val_accuracy", float("nan"))))
            if stop_event.is_set():          # Stop pressed between epochs
                self.model.stop_training = True

    arch = params.get("arch", "lenet")
    cfg = TrainConfig(
        run_id=f"gui_{arch}_{time.strftime('%Y%m%d_%H%M%S')}",
        arch=arch,
        dataset="mnist+symbols" if model_name == "CNN + operators" else "mnist",
        augmentation=params.get("augmentation", "light"),
        learning_rate=float(params["learning_rate"]),
        batch_size=int(params["batch_size"]),
        max_epochs=int(params["epochs"]),
        output_root=str(output_root or GUI_RUNS),
    )
    summary = train(cfg, data=data, extra_callbacks=[_GuiHooks()])   # data/output_root: tests only
    return Recogniser(Path(summary["checkpoint"]))


def _to_bgr(pil_image) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def recognise_cnn(model: Recogniser, pil_image) -> dict:
    """Read an image; returns text, results, boxes, thumbs (28x28) and binary."""
    from PIL import Image
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from segmentation import _prepare_binary

    bgr = _to_bgr(pil_image)
    text, details = model.read_image(bgr)    # project preprocessing + segmentation
    return {
        "text": text,
        "results": [(d["label"], d["confidence"]) for d in details],
        "boxes": [(x, y, x + w, y + h) for (x, y, w, h) in (d["box"] for d in details)],
        "thumbs": [Image.fromarray((d["input"] * 255).astype(np.uint8)) for d in details],
        "binary": Image.fromarray(_prepare_binary(bgr)),
    }
