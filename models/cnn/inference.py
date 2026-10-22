"""
Each detail dict has: label, confidence, box (x, y, w, h) and top3, so the GUI
can show per-symbol confidence and the segmented preview.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINTS = REPO_ROOT / "models" / "checkpoints"
SRC = REPO_ROOT / "src"

DEFAULT_LABELS = [str(d) for d in range(10)]   # models trained before label files existed

# Frozen checkpoints the team should load (see README section 6).
FROZEN = {
    "cnn_best": "digits 0-9 - deeper VGG, tuned (run s3_B5_bs_64)  <- the digit model the system uses",
    "lenet_best": "digits 0-9 - LeNet-5, tuned (run pm_lenet_best), for the model comparison",
    "resnet_best": "digits 0-9 - small ResNet, tuned (run pm_resnet_best), for the model comparison",
    "mlp_best": "digits 0-9 - MLP, no convolutions (run pm_mlp_best), for the model comparison",
    "ext_best": "digits + operators (16 classes) - deeper VGG, retrained on team handwriting (run ext_v3)",
    "ext_v2": "previous ext_best, MNIST + symbol datasets only (needs the input blur)",
}


def resolve_checkpoint(name_or_path: str | Path) -> Path:
    """Accept 'cnn_best', 'tuning/pm_lenet_best' or a full path to a .keras file."""
    p = Path(name_or_path)
    if p.suffix == ".keras" and p.exists():
        return p
    candidate = CHECKPOINTS / f"{name_or_path}.keras"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"No checkpoint '{name_or_path}'. Expected {candidate}. Frozen models: "
        f"{', '.join(FROZEN)}")


def load_label_names(checkpoint: Path) -> List[str]:
    """Class names saved by the trainer, or 0-9 for older checkpoints."""
    label_file = checkpoint.with_suffix(".labels.json")
    if label_file.exists():
        return list(json.loads(label_file.read_text())["label_names"])
    return list(DEFAULT_LABELS)


def load_input_blur(checkpoint: Path) -> Optional[bool]:
    """The 'input_blur' flag saved with a checkpoint, or None if it has none."""
    label_file = checkpoint.with_suffix(".labels.json")
    if label_file.exists():
        return json.loads(label_file.read_text()).get("input_blur")
    return None


def _src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


# How numbers are segmented (expressions always use segment_expression):
#   "repaired"  John's second round (src/segmentation_split.py): ruled lines
#               removed, contours, stacked pieces merged, boxes too wide for one
#               digit cut where cnn_best reads the pieces most confidently.
#               With cnn_best on 99 real numbers: 63.6 % -> 67.7 % read exactly,
#               touching digits 8 % -> 20 %; no number contours read was lost.
#   "contours"  the Task 2 selection, one box per connected region of ink.
NUMBER_SEGMENTATION = "repaired"
SEGMENTATION_SCORER = "cnn_best"   # the model that chooses the cuts, for every reader
_scorer_model: Optional["Recogniser"] = None


def _digit_scorer(reader: Optional["Recogniser"] = None):
    """crops -> (n, 10) digit probabilities from cnn_best, used only to choose cuts.

    The same model chooses the cuts whichever model then reads the number, so
    every model is given the same crops (fair comparison, GUI = evaluation).
    """
    global _scorer_model
    if reader is not None and reader.name == SEGMENTATION_SCORER and len(reader.labels) == 10:
        model = reader
    else:
        if _scorer_model is None:
            _scorer_model = Recogniser(SEGMENTATION_SCORER)
        model = _scorer_model
    _src_on_path()
    from preprocessing import preprocess

    def score(crops):
        if len(crops) == 0:
            return np.zeros((0, 10))
        return model.probabilities(np.stack([preprocess(c, add_channel=True) for c in crops]))
    return score


def segment_number(image: np.ndarray, method: Optional[str] = None,
                   reader: Optional["Recogniser"] = None):
    """Photo of a number -> (crops cut from the photo, boxes (x, y, w, h)), left to right.

    method: "repaired" (default, NUMBER_SEGMENTATION) or any method of
    segmentation.METHODS, e.g. "contours".
    """
    _src_on_path()
    method = method or NUMBER_SEGMENTATION
    if method == "repaired":
        from segmentation_split import segment_digits_repaired
        crops, boxes = segment_digits_repaired(image, _digit_scorer(reader))
        return list(crops), [tuple(int(v) for v in b) for b in boxes]
    from segmentation import segment_digits
    _, boxes = segment_digits(image, method=method)
    return [image[y:y + h, x:x + w] for (x, y, w, h) in boxes], [tuple(int(v) for v in b) for b in boxes]


def soften(patches: np.ndarray) -> np.ndarray:
    """3x3 Gaussian blur on (N, 28, 28, 1) patches.

    Photo crops come out of preprocess() with hard binary edges; the training
    images (MNIST, and especially the low-res symbol set) have soft edges.
    ext_best learned "soft = bracket, hard = 1": on the team's expressions it
    read 4.5 % of brackets correctly as-is and 99 % after this blur, while the
    MNIST+symbols test set only moves 99.57 -> 99.41 %. Digit-only models do
    not need it (cnn_best on real digits: 90.0 % without, 88.6 % with).
    """
    import cv2
    out = np.stack([cv2.GaussianBlur(p[..., 0].astype("float32"), (3, 3), 0) for p in patches])
    return out[..., np.newaxis]


class Recogniser:
    """A frozen model plus its label names and the matching preprocessing."""

    DIGITS: Tuple[str, ...] = tuple(DEFAULT_LABELS)

    def __init__(self, checkpoint: str | Path = "cnn_best", model=None,
                 label_names: Optional[Sequence[str]] = None, soften: Optional[bool] = None):
        """Load from models/checkpoints/<checkpoint>.keras.

        `model` and `label_names` can be injected instead (used by the tests,
        so they run without a trained checkpoint).
        `soften`: blur photo crops slightly (see soften()). Default: the
        checkpoint's 'input_blur' flag if it has one (ext_v3 and later: off),
        otherwise on only for models with operator classes (ext_best).
        """
        if model is None:
            import tensorflow as tf
            path = resolve_checkpoint(checkpoint)
            model = tf.keras.models.load_model(str(path))
            label_names = label_names or load_label_names(path)
            if soften is None:
                soften = load_input_blur(path)   # retrained models say they need none
        self.name = Path(str(checkpoint)).stem
        self.model = model
        self.labels: List[str] = list(label_names or DEFAULT_LABELS)
        n_out = int(self.model.output_shape[-1])
        if n_out != len(self.labels):
            raise ValueError(f"{self.name}: model has {n_out} outputs but "
                             f"{len(self.labels)} label names")
        self.soften = self.reads_operators() if soften is None else soften

    #  classification 
    def _mask(self, allowed: Optional[Iterable[str]]) -> Optional[np.ndarray]:
        if allowed is None:
            return None
        allowed = set(allowed)
        unknown = allowed - set(self.labels)
        if unknown:
            raise ValueError(f"{self.name} has no classes {sorted(unknown)}")
        return np.array([l in allowed for l in self.labels])

    def probabilities(self, patches: np.ndarray,
                      allowed: Optional[Iterable[str]] = None) -> np.ndarray:
        """Softmax over the classes for a batch of (N, 28, 28[, 1]) patches.

        With `allowed`, every other class is zeroed and the rest renormalised,
        i.e. the model is asked "which of THESE is it?".
        """
        x = np.asarray(patches, dtype="float32")
        if x.ndim == 3:
            x = x[..., np.newaxis]
        if x.max() > 1.0:          # accept 0-255 input as well
            x = x / 255.0
        probs = np.asarray(self.model.predict(x, verbose=0), dtype="float64")
        mask = self._mask(allowed)
        if mask is not None:
            probs = probs * mask
            probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
        return probs

    def classify(self, patches: np.ndarray, allowed: Optional[Iterable[str]] = None
                 ) -> List[Dict]:
        """One dict per patch: label, confidence and the top-3 alternatives."""
        probs = self.probabilities(patches, allowed)
        out = []
        for p in probs:
            order = np.argsort(-p, kind="stable")[:3]   # ties -> lower class, like argmax
            out.append({
                "label": self.labels[order[0]],
                "confidence": float(p[order[0]]),
                "top3": [(self.labels[i], float(p[i])) for i in order],
            })
        return out

    #  whole images 
    def read_crops(self, crops: Sequence[np.ndarray],
                   allowed: Optional[Iterable[str]] = None) -> List[Dict]:
        """Classify crops of the ORIGINAL photo (dark ink on light paper)."""
        _src_on_path()
        from preprocessing import preprocess
        if len(crops) == 0:
            return []
        patches = np.stack([preprocess(c, add_channel=True) for c in crops])
        if self.soften:
            patches = soften(patches)
        details = self.classify(patches, allowed)
        for d, patch in zip(details, patches):
            d["input"] = patch[..., 0]      # the 28x28 image the model saw (0-1)
        return details

    def reads_operators(self, allowed: Optional[Iterable[str]] = None) -> bool:
        """True when this call can output an operator class (+ - * / ( ))."""
        classes = set(self.labels) if allowed is None else set(allowed)
        return bool(classes - set(DEFAULT_LABELS))

    def read_image(self, image: np.ndarray, allowed: Optional[Iterable[str]] = None,
                   method: Optional[str] = None) -> Tuple[str, List[Dict]]:
        """Segment a photo, classify every box left to right, return (text, details).

        Numbers: segment_number() - John's repaired segmentation by default
        (method="contours" for the Task 2 selection). Expressions: contours with
        operators kept (segment_expression). Preprocessing: otsu_mnist_box.
        Crops come from the photo, not the binary mask - that is what
        preprocess() expects (README section 3).
        """
        _src_on_path()
        if self.reads_operators(allowed):
            from segmentation import SELECTED_METHOD, segment_expression
            m = method if method and method != "repaired" else SELECTED_METHOD
            _, boxes = segment_expression(image, method=m)
            crops = [image[y:y + h, x:x + w] for (x, y, w, h) in boxes]
        else:
            crops, boxes = segment_number(image, method, reader=self)
        details = self.read_crops(crops, allowed)
        for d, box in zip(details, boxes):
            d["box"] = tuple(int(v) for v in box)
        return "".join(d["label"] for d in details), details


if __name__ == "__main__":
    import argparse
    import cv2

    ap = argparse.ArgumentParser(description="Read one photo with a frozen model.")
    ap.add_argument("image")
    ap.add_argument("--model", default="cnn_best")
    ap.add_argument("--digits-only", action="store_true",
                    help="with ext_best: switch the symbol classes off")
    args = ap.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Could not read {args.image}")
    rec = Recogniser(args.model)
    text, details = rec.read_image(img, Recogniser.DIGITS if args.digits_only else None)
    print(f"{rec.name}: {text!r}")
    for d in details:
        alts = ", ".join(f"{l} {p:.2f}" for l, p in d["top3"][1:])
        print(f"  {d['label']:>2}  {d['confidence']:.3f}  box={d['box']}  (next: {alts})")
