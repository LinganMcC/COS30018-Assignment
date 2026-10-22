"""Task 3 classical baselines: HOG features + SVM, and HOG features + Random Forest.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

from common import CHECKPOINTS, RESULTS, ROOT, load_mnist_splits

# HOG settings compared on validation. 7x7 cells give a 4x4 grid (324 values),
# 4x4 cells a 7x7 grid (1,296 values): coarse and robust, or fine and detailed.
HOG_CONFIGS = {
    "hog7": dict(orientations=9, pixels_per_cell=(7, 7), cells_per_block=(2, 2)),
    "hog4": dict(orientations=9, pixels_per_cell=(4, 4), cells_per_block=(2, 2)),
}
SVM_GRID = {"C": [1.0, 5.0, 10.0, 20.0], "gamma": ["scale", "scale/2", "scale*2"]}   # 12 settings
RF_GRID = {"n_estimators": [200], "max_features": ["sqrt", 0.2], "min_samples_leaf": [1, 2]}
TUNE_SUBSET = 12_000          # grid search on a random 12k of the 54k training images
SEED = 42


def hog_features(images: np.ndarray, config: str = "hog7") -> np.ndarray:
    """HOG vector for every 28x28 image (any of (N,28,28), (N,28,28,1), values 0-1 or 0-255)."""
    from skimage.feature import hog
    x = np.asarray(images, "float32")
    if x.ndim == 4:
        x = x[..., 0]
    if x.max() > 1.0:
        x = x / 255.0
    cfg = HOG_CONFIGS[config]
    return np.stack([hog(img, block_norm="L2-Hys", **cfg) for img in x]).astype("float32")


@dataclass
class Baseline:
    """A fitted scikit-learn classifier plus the HOG setting it expects."""
    key: str
    hog: str
    clf: object

    def predict_proba(self, images: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(hog_features(images, self.hog))

    def predict(self, images: np.ndarray) -> np.ndarray:
        return self.predict_proba(images).argmax(1)

    def size(self) -> int:
        """Support vectors (SVM) or tree nodes (forest): the model's 'parameter count'."""
        if hasattr(self.clf, "n_support_"):
            return int(self.clf.n_support_.sum())
        return int(sum(t.tree_.node_count for t in self.clf.estimators_))


def _gamma(spec, x: np.ndarray):
    """'scale' is scikit-learn's 1 / (n_features * var); the grid also tries half and double."""
    if spec == "scale":
        return "scale"
    base = 1.0 / (x.shape[1] * x.var())
    return base / 2 if spec == "scale/2" else base * 2


def make_classifier(kind: str, params: Dict, x: np.ndarray):
    if kind == "svm":
        from sklearn.svm import SVC
        return SVC(kernel="rbf", C=params["C"], gamma=_gamma(params["gamma"], x),
                   probability=True, random_state=SEED, cache_size=1000)
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_jobs=-1, random_state=SEED, **params)


def grid(kind: str) -> List[Dict]:
    g = SVM_GRID if kind == "svm" else RF_GRID
    return [dict(zip(g, values)) for values in itertools.product(*g.values())]


_FEATURE_CACHE: Dict = {}


def cached_features(split: str, x: np.ndarray, config: str) -> np.ndarray:
    """HOG features for a whole split, computed once per run (about 0.6 s per 1,000 images)."""
    key = (split, config, len(x))
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE[key] = hog_features(x, config)
    return _FEATURE_CACHE[key]


def tune(kind: str, data, subset: int, log: List[Dict]) -> Dict:
    """Pick the HOG setting, then the classifier settings, on validation accuracy."""
    (x_tr, y_tr), (x_va, y_va), _ = data
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(x_tr))[:subset]
    best = None
    for hog_cfg in HOG_CONFIGS:
        f_tr = cached_features("train", x_tr, hog_cfg)[idx]
        f_va = cached_features("val", x_va, hog_cfg)
        # cell size is judged with the first grid point; the full grid then runs on the winner
        for params in grid(kind)[:1]:
            acc, secs = _fit_score(kind, params, f_tr, y_tr[idx], f_va, y_va)
            log.append({"model": kind, "stage": "hog", "hog": hog_cfg, "params": json.dumps(params),
                        "train_images": len(idx), "val_accuracy": acc, "seconds": secs})
            print(f"  {kind} {hog_cfg} {params}: val {acc:.4f} ({secs:.0f}s)")
            if best is None or acc > best["val"]:
                best = {"hog": hog_cfg, "params": params, "val": acc}
    f_tr = cached_features("train", x_tr, best["hog"])[idx]
    f_va = cached_features("val", x_va, best["hog"])
    for params in grid(kind)[1:]:
        acc, secs = _fit_score(kind, params, f_tr, y_tr[idx], f_va, y_va)
        log.append({"model": kind, "stage": "params", "hog": best["hog"], "params": json.dumps(params),
                    "train_images": len(idx), "val_accuracy": acc, "seconds": secs})
        print(f"  {kind} {best['hog']} {params}: val {acc:.4f} ({secs:.0f}s)")
        if acc > best["val"]:
            best = {"hog": best["hog"], "params": params, "val": acc}
    return best


def _fit_score(kind, params, f_tr, y_tr, f_va, y_va):
    start = time.time()
    clf = make_classifier(kind, params, f_tr)
    if kind == "svm":
        clf.set_params(probability=False)        # probabilities are not needed to score
    clf.fit(f_tr, y_tr)
    return float((clf.predict(f_va) == y_va).mean()), time.time() - start


def train_final(kind: str, best: Dict, data, quick: bool) -> Dict:
    """Fit the chosen setting on the full training split; score val and test once."""
    import joblib
    (x_tr, y_tr), (x_va, y_va), (x_te, y_te) = data
    if quick:
        x_tr, y_tr = x_tr[:3000], y_tr[:3000]
    start = time.time()
    f_tr = cached_features("train", x_tr, best["hog"])
    clf = make_classifier(kind, best["params"], f_tr)
    clf.fit(f_tr, y_tr)
    seconds = time.time() - start
    # Built from the importable module (not __main__) so joblib can load it from any script.
    import baselines as module
    model = module.Baseline(f"hog_{'rf' if kind == 'rf' else 'svm'}", best["hog"], clf)
    val = float((clf.predict(cached_features("val", x_va, best["hog"])) == y_va).mean())
    test = float((clf.predict(cached_features("test", x_te, best["hog"])) == y_te).mean())
    out = CHECKPOINTS / ("quick" if quick else "") / f"{model.key}.joblib"
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out, compress=("xz", 9))   # keeps the forest under 20 MB
    summary = {"model": model.key, "hog": best["hog"], "hog_settings": HOG_CONFIGS[best["hog"]],
               "params": best["params"], "tuning_val_accuracy": best["val"],
               "val_accuracy": val, "test_accuracy": test, "train_images": int(len(x_tr)),
               "training_time_s": seconds, "size": model.size(),
               "file_mb": round(out.stat().st_size / 1e6, 1)}
    out.with_suffix(".source.json").write_text(json.dumps(summary, indent=2))
    print(f"{model.key}: val {val:.4f} | test {test:.4f} | {seconds:.0f}s | "
          f"{summary['file_mb']} MB -> {out.relative_to(ROOT)}")
    if not quick:
        from experiment_logger import log_run
        log_run(model.key, f"HOG ({best['hog']}) + {'Random Forest' if kind == 'rf' else 'SVM (RBF)'}",
                0.0, 0, 0, 0.0, test, seconds, val_accuracy=val,
                notes=f"classical baseline; params={json.dumps(best['params'])}; "
                      f"chosen on validation (evaluation/results/baseline_tuning.csv)",
                log_path=ROOT / "models" / "experiments" / "experiment_log.csv")
    return summary


def load_baseline(key: str) -> Baseline:
    import joblib
    path = CHECKPOINTS / f"{key}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path.name} not found - run: python evaluation/baselines.py")
    return joblib.load(path)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Train the HOG + SVM / Random Forest baselines.")
    ap.add_argument("--model", choices=["svm", "rf", "both"], default="both")
    ap.add_argument("--quick", action="store_true", help="small subsets, nothing logged")
    args = ap.parse_args(argv)

    data = load_mnist_splits()
    kinds = ["svm", "rf"] if args.model == "both" else [args.model]
    log: List[Dict] = []
    for kind in kinds:
        print(f"\n== {kind}: tuning on {('2,000' if args.quick else f'{TUNE_SUBSET:,}')} "
              "training images, scored on validation")
        best = tune(kind, data, 2000 if args.quick else TUNE_SUBSET, log)
        print(f"   chosen: {best}")
        train_final(kind, best, data, args.quick)
    if not args.quick:
        RESULTS.mkdir(parents=True, exist_ok=True)
        path = RESULTS / "baseline_tuning.csv"
        old = list(csv.DictReader(open(path))) if path.exists() else []
        old = [r for r in old if r["model"] not in kinds]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0]))
            w.writeheader()
            w.writerows(old + log)


if __name__ == "__main__":
    main()
