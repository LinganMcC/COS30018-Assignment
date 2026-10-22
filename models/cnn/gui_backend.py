"""Model backend for gui.py: every model option, pretrained or trained in the GUI.

All five options read a photo through the same project pipeline
(contours segmentation -> otsu_mnist_box preprocessing -> classifier), so the
GUI shows the same boxes, 28x28 inputs and per-digit confidences whichever
model is chosen, and the numbers match evaluation/results/model_comparison.md.

    GUI option        Pretrained model (models/checkpoints/)   "Train new"
    CNN               cnn_best.keras   (VGG-deep)              chosen CNN arch, GUI settings
    CNN + operators   ext_best.keras   (16 classes)            chosen CNN arch, MNIST + symbols
    MLP               mlp_best.keras                           MLP, GUI settings
    SVM               hog_svm.joblib   (HOG + RBF SVM)         tuned settings, 10,000 images
    Random Forest     hog_rf.joblib    (HOG + 200 trees)       tuned settings, 10,000 images
"""
from __future__ import annotations

import json
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
NEURAL_OPTIONS: Dict[str, str] = {**CNN_OPTIONS, "MLP": "mlp_best"}
CLASSICAL_OPTIONS: Dict[str, str] = {   # trained by evaluation/baselines.py
    "SVM": "hog_svm",
    "Random Forest": "hog_rf",
}
CNN_ARCHS = ["lenet", "vgg_deep", "resnet"]
AUGMENTATIONS = ["none", "light", "medium", "strong"]
GUI_RUNS = REPO_ROOT / "models" / "gui_runs"   # GUI-trained models (git-ignored)
EVALUATION = REPO_ROOT / "evaluation"
CLASSICAL_TRAIN_IMAGES = 10_000   # GUI training subset: about a minute instead of 25


def is_cnn(model_name: str) -> bool:
    return model_name in CNN_OPTIONS


def is_classical(model_name: str) -> bool:
    return model_name in CLASSICAL_OPTIONS


def is_supported(model_name: str) -> bool:
    return model_name in NEURAL_OPTIONS or model_name in CLASSICAL_OPTIONS


def default_checkpoint(model_name: str) -> Path:
    if is_classical(model_name):
        return CHECKPOINTS / f"{CLASSICAL_OPTIONS[model_name]}.joblib"
    return CHECKPOINTS / f"{NEURAL_OPTIONS[model_name]}.keras"


def training_steps(model_name: str, params: dict) -> int:
    """Length of the GUI progress bar: epochs, or one fit for SVM / Random Forest."""
    return 1 if is_classical(model_name) else int(params["epochs"])


# ------------------------------------------------------------------ classical models

def _baselines():
    """evaluation/baselines.py (needed to build and to unpickle the HOG models)."""
    if str(EVALUATION) not in sys.path:
        sys.path.insert(0, str(EVALUATION))
    import baselines
    return baselines


class ClassicalModel:
    """Makes a HOG + SVM / Random Forest model look like a Keras model to Recogniser.

    Recogniser only calls model.predict(x) on (N, 28, 28, 1) patches in 0-1 and
    reads model.output_shape, so the classical models go through exactly the
    same segmentation and preprocessing as the CNN.
    """
    output_shape = (None, 10)

    def __init__(self, baseline):
        self.baseline = baseline

    def predict(self, x: np.ndarray, verbose: int = 0) -> np.ndarray:
        return self.baseline.predict_proba(np.asarray(x)[..., 0])


def classical_recogniser(baseline, name: str) -> Recogniser:
    return Recogniser(name, model=ClassicalModel(baseline), soften=False)


def load_classical(model_name: str, path: Optional[str] = None) -> Recogniser:
    import joblib
    _baselines()
    path = Path(path) if path else default_checkpoint(model_name)
    baseline = joblib.load(path)
    if not hasattr(baseline, "predict_proba") or not hasattr(baseline, "hog"):
        raise ValueError(f"{path.name} is not a HOG baseline saved by evaluation/baselines.py")
    expected = CLASSICAL_OPTIONS[model_name]
    if getattr(baseline, "key", expected) != expected:
        raise ValueError(f"{path.name} holds '{baseline.key}', not a {model_name} model.")
    return classical_recogniser(baseline, path.stem)


def _tuned_settings(key: str) -> dict:
    """HOG cell size and classifier settings chosen on validation by baselines.py."""
    source = CHECKPOINTS / f"{key}.source.json"
    if source.exists():
        s = json.loads(source.read_text())
        return {"hog": s["hog"], "params": s["params"]}
    if key == "hog_svm":
        return {"hog": "hog4", "params": {"C": 5.0, "gamma": "scale*2"}}
    return {"hog": "hog4", "params": {"n_estimators": 200, "max_features": "sqrt",
                                      "min_samples_leaf": 1}}


def train_classical(model_name: str, params: dict,
                    on_epoch: Callable[[int, float, float], None], stop_event,
                    data=None, output_root: Optional[Path] = None,
                    n_images: int = CLASSICAL_TRAIN_IMAGES) -> Recogniser:
    """Fit HOG + SVM / Random Forest with the tuned settings on a training subset.

    Epochs, batch size, learning rate and augmentation do not apply to these
    models. A fit cannot be interrupted, so Stop only works before it starts.
    Reports (1, validation log-loss, validation accuracy) when done.
    """
    import joblib
    from sklearn.metrics import log_loss
    b = _baselines()
    if data is None:
        from mnist_loader import load_mnist
        (x_tr, y_tr), (x_va, y_va), _ = load_mnist()
    else:
        x_tr, y_tr, x_va, y_va = data.x_train, data.y_train, data.x_val, data.y_val
    idx = np.random.default_rng(b.SEED).permutation(len(x_tr))[:n_images]
    x_tr, y_tr = x_tr[idx], y_tr[idx]
    if stop_event.is_set():
        raise RuntimeError("Training stopped before it started.")

    key = CLASSICAL_OPTIONS[model_name]
    tuned = _tuned_settings(key)
    f_tr = b.hog_features(x_tr, tuned["hog"])
    clf = b.make_classifier("svm" if key == "hog_svm" else "rf", tuned["params"], f_tr)
    clf.fit(f_tr, y_tr)
    baseline = b.Baseline(key, tuned["hog"], clf)

    probs = baseline.predict_proba(x_va)
    acc = float((probs.argmax(1) == y_va).mean())
    clipped = np.clip(probs, 1e-7, 1)
    loss = float(log_loss(y_va, clipped / clipped.sum(1, keepdims=True), labels=list(range(10))))
    on_epoch(1, loss, acc)

    out = Path(output_root or GUI_RUNS) / "checkpoints" / \
        f"gui_{key}_{time.strftime('%Y%m%d_%H%M%S')}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(baseline, out, compress=3)
    return classical_recogniser(baseline, out.stem)


# ------------------------------------------------------------------ neural models

def load_cnn(model_name: str, path: Optional[str] = None) -> Recogniser:
    """Load `path`, or the frozen model for this option (CNN, CNN + operators, MLP)."""
    rec = Recogniser(resolve_checkpoint(path) if path else default_checkpoint(model_name))
    if model_name == "CNN + operators" and not rec.reads_operators():
        raise ValueError(f"{Path(path).name} has no operator classes - "
                         "pick a 16-class model or use the CNN option.")
    if model_name != "CNN + operators" and rec.reads_operators():
        raise ValueError(f"{Path(path).name} has operator classes - "
                         "use the CNN + operators option.")
    return rec


def train_cnn(model_name: str, params: dict,
              on_epoch: Callable[[int, float, float], None],
              stop_event, data=None, output_root: Optional[Path] = None) -> Recogniser:
    """Train with the GUI settings; reports each epoch and honours Stop.

    The MLP option always trains the 'mlp' architecture; the CNN options use
    the architecture chosen in the GUI.
    """
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

    arch = "mlp" if model_name == "MLP" else params.get("arch", "lenet")
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


# ------------------------------------------------------------------ one entry point per action

def load_model(model_name: str, path: Optional[str] = None) -> Recogniser:
    """Pretrained: `path`, or the frozen model for this option."""
    if is_classical(model_name):
        return load_classical(model_name, path)
    if model_name in NEURAL_OPTIONS:
        return load_cnn(model_name, path)
    raise ValueError(f"Unknown model option '{model_name}'")


def train_model(model_name: str, params: dict,
                on_epoch: Callable[[int, float, float], None], stop_event,
                data=None, output_root: Optional[Path] = None) -> Recogniser:
    """Train new: SVM / Random Forest with their tuned settings, neural models with the GUI's."""
    if is_classical(model_name):
        return train_classical(model_name, params, on_epoch, stop_event, data, output_root)
    if model_name in NEURAL_OPTIONS:
        return train_cnn(model_name, params, on_epoch, stop_event, data, output_root)
    raise ValueError(f"Unknown model option '{model_name}'")


def _to_bgr(pil_image) -> np.ndarray:
    import cv2
    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def recognise_cnn(model: Recogniser, pil_image) -> dict:
    """Read an image with any loaded model; returns text, results, boxes, thumbs (28x28) and binary."""
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


recognise = recognise_cnn   # the same pipeline serves every option
