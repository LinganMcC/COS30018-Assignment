"""Real handwritten symbols cut from the team's expression photos.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from datasets import DIGIT_LABELS, SYMBOL_LABELS
from inference import REPO_ROOT

READY = REPO_ROOT / "data" / "custom_samples" / "expressions" / "ready"
CACHE = REPO_ROOT / "data" / "custom_samples" / "expressions" / "real_symbols_cache.npz"
LABELS = DIGIT_LABELS + SYMBOL_LABELS
TRAIN_SHEETS, TEST_SHEETS, VAL_WRITER = {"1", "2"}, {"3", "4"}, "7"


def split_of(sheet: str, writer: str) -> str:
    if sheet in TEST_SHEETS:
        return "test"
    return "val" if writer == VAL_WRITER else "train"


def _cut(image: np.ndarray) -> List[np.ndarray]:
    """28x28 patches exactly as Recogniser.read_image builds them (before soften)."""
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from preprocessing import preprocess
    from segmentation import SELECTED_METHOD, segment_expression
    _, boxes = segment_expression(image, method=SELECTED_METHOD)
    return [preprocess(image[y:y + h, x:x + w], add_channel=True) for (x, y, w, h) in boxes]


def build(folder: Path = READY) -> Dict[str, np.ndarray]:
    import cv2
    xs, ys, splits, files, skipped = [], [], [], [], 0
    for lab in csv.DictReader(open(folder / "labels.csv", encoding="utf-8")):
        img = cv2.imread(str(folder / lab["file"]))
        truth = lab["expression"]
        patches = _cut(img) if img is not None else []
        if len(patches) != len(truth):
            skipped += 1
            continue
        split = split_of(lab["sheet"], lab["writer"])
        for ch, p in zip(truth, patches):
            xs.append(p)
            ys.append(LABELS.index(ch))
            splits.append(split)
            files.append(lab["file"])
    print(f"real symbols: {len(xs)} crops, {skipped} expressions skipped (wrong box count)")
    return {"x": np.stack(xs).astype("float32"), "y": np.array(ys, "int64"),
            "split": np.array(splits), "source": np.array(files)}


def load_real_symbols(split: str, rebuild: bool = False) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """(x, y, info) for 'train', 'val' or 'test'. Cached after the first build."""
    if rebuild or not CACHE.exists():
        np.savez_compressed(CACHE, **build())
    d = np.load(CACHE)
    m = d["split"] == split
    return d["x"][m], d["y"][m], {"file": d["source"][m], "labels": LABELS}


if __name__ == "__main__":
    from collections import Counter
    for s in ("train", "val", "test"):
        x, y, _ = load_real_symbols(s, rebuild=(s == "train"))
        c = Counter(LABELS[i] for i in y)
        print(f"{s:5s} {len(y):5d}  " + " ".join(f"{k}:{c[k]}" for k in LABELS))
