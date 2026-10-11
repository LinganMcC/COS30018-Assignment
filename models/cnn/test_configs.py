"""Checks every tuning grid in configs/ so a typo is caught before a long run.

Run from models/cnn/:  pytest test_configs.py -v
"""

import json
from pathlib import Path

import pytest

from augmentation import get_preset
from datasets import available_datasets
from model_zoo import available_architectures
from trainer import load_grid

CONFIGS = sorted((Path(__file__).parent / "configs").glob("*.json"))


@pytest.mark.parametrize("path", CONFIGS, ids=[p.name for p in CONFIGS])
def test_grid_is_valid(path):
    configs = load_grid(path)                       # also rejects unknown keys
    for c in configs:
        assert c.arch in available_architectures()
        assert c.dataset in available_datasets()
        get_preset(c.augmentation)                   # raises if unknown
        assert c.max_epochs >= c.patience > 0


def test_run_ids_are_unique_across_all_grids():
    ids = [c.run_id for path in CONFIGS for c in load_grid(path)]
    assert len(ids) == len(set(ids)), "run_ids must be unique across every config file"


def test_per_model_grid_settings():
    path = Path(__file__).parent / "configs" / "sprint3_per_model.json"
    by_id = {c.run_id: c for c in load_grid(path)}
    assert set(by_id) == {"pm_lenet_best", "pm_resnet_best", "pm_mlp_best"}
    for c in by_id.values():
        assert c.augmentation == "light"
        assert c.max_epochs == 50 and c.patience == 8
        assert c.dropout == 0.3
    assert json.loads(path.read_text())["defaults"]["dataset"] == "mnist"
