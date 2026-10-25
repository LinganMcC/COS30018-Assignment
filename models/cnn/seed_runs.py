"""Retrain the configurations behind the key Sprint 3 decisions with seeds 43 and 44.

Each configuration is copied exactly from its original run (seed 42); only the seed
changes. Runs that already have a summary are skipped, so it can be resumed.

    python seed_runs.py          (about 3 h on a laptop CPU)
    python seed_variation.py     summary table
"""
import json
import sys
from pathlib import Path
sys.path.insert(0, ".")
from trainer import TrainConfig, train

VGG = dict(arch="vgg_deep", dataset="mnist", learning_rate=1e-3, max_epochs=30, patience=5)
PM = dict(dataset="mnist", augmentation="light", learning_rate=1e-3, batch_size=128,
          max_epochs=50, patience=8, dropout=0.3)
CONFIGS = {
    "B5": dict(VGG, augmentation="light", batch_size=64, dropout=0.5),
    "B3": dict(VGG, augmentation="light", batch_size=128, dropout=0.3),
    "A2": dict(VGG, augmentation="light", batch_size=128, dropout=0.5),
    "A1": dict(VGG, augmentation="none", batch_size=128, dropout=0.5),
    "lenet": dict(PM, arch="lenet"),
    "resnet": dict(PM, arch="resnet"),
}
ORDER = ["B5", "B3", "A2", "A1", "lenet", "resnet"]
for seed in (43, 44):
    for name in ORDER:
        run_id = f"seed_{name}_s{seed}"
        if Path(f"../experiments/tuning/{run_id}_summary.json").exists():
            continue
        s = train(TrainConfig(run_id=run_id, seed=seed, **CONFIGS[name]))
        print(run_id, json.dumps({k: s[k] for k in ("val_accuracy", "test_accuracy", "robust_test_accuracy", "epochs_run", "training_time_s")}), flush=True)
print("ALL_SEEDS_DONE")
