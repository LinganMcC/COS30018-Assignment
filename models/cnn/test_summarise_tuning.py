"""Tests for picking and freezing the best run (no training needed)."""
import json

import summarise_tuning as st


def summary(run, val, robust=0.99, time_s=100, val_real=None, dataset="mnist"):
    s = {"config": {"run_id": run, "dataset": dataset}, "val_accuracy": val,
         "robust_test_accuracy": robust, "training_time_s": time_s}
    if val_real is not None:
        s["val_real_accuracy"] = val_real
    return s


def test_fine_tuned_runs_are_picked_on_real_validation():
    runs = [summary("a", 0.995, val_real=0.990), summary("b", 0.993, val_real=1.0)]
    assert st.pick_best(runs)["config"]["run_id"] == "b"


def test_plain_runs_are_picked_on_val_accuracy():
    runs = [summary("a", 0.995), summary("b", 0.993)]
    assert st.pick_best(runs)["config"]["run_id"] == "a"


def test_load_summaries_keeps_datasets_apart(tmp_path):
    tuning = tmp_path / "experiments" / "tuning"
    tuning.mkdir(parents=True)
    for s in (summary("d", 0.99), summary("e", 0.99, dataset="mnist+symbols"),
              summary("f", 0.99, dataset="mnist+symbols+real")):
        (tuning / f"{s['config']['run_id']}_summary.json").write_text(json.dumps(s))
    pick = lambda ds: [s["config"]["run_id"] for s in st.load_summaries(tmp_path, ds)]
    assert pick("mnist") == ["d"]
    assert pick("mnist+symbols") == ["e"]
    assert pick("mnist+symbols+real") == ["f"]


def test_frozen_source_reads_the_run_behind_a_frozen_model(tmp_path):
    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    assert st.frozen_source("ext_best", tmp_path) is None
    (ckpts / "ext_best.source.json").write_text('{"source_run": "ext_v3"}')
    assert st.frozen_source("ext_best", tmp_path) == "ext_v3"
