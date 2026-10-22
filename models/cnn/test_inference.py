"""Tests for inference.Recogniser using a fake model (no checkpoint needed)."""
import cv2
import numpy as np
import pytest

from inference import DEFAULT_LABELS, Recogniser, load_label_names, soften

EXT_LABELS = DEFAULT_LABELS + ["+", "-", "*", "/", "(", ")"]


class FakeModel:
    """Returns fixed probabilities; records what it was given."""

    def __init__(self, probs):
        self.probs = np.asarray(probs, dtype="float32")
        self.output_shape = (None, self.probs.shape[-1])
        self.seen = None

    def predict(self, x, verbose=0):
        self.seen = x
        return np.repeat(self.probs[None], len(x), axis=0)


def one_hot_ish(labels, winner, runner_up, p_win=0.6, p_run=0.3):
    p = np.full(len(labels), (1 - p_win - p_run) / (len(labels) - 2))
    p[labels.index(winner)] = p_win
    p[labels.index(runner_up)] = p_run
    return p


def test_label_count_must_match_model():
    with pytest.raises(ValueError):
        Recogniser("fake", model=FakeModel(np.ones(16) / 16), label_names=DEFAULT_LABELS)


def test_classify_returns_label_names_not_indices():
    rec = Recogniser("fake", model=FakeModel(one_hot_ish(EXT_LABELS, "*", "4")),
                     label_names=EXT_LABELS)
    out = rec.classify(np.zeros((2, 28, 28, 1)))
    assert [o["label"] for o in out] == ["*", "*"]
    assert out[0]["top3"][1][0] == "4"
    assert out[0]["confidence"] == pytest.approx(0.6)


def test_allowed_restricts_and_renormalises():
    # The model prefers "/" but this is a plain number: only digits allowed.
    rec = Recogniser("fake", model=FakeModel(one_hot_ish(EXT_LABELS, "/", "1")),
                     label_names=EXT_LABELS)
    probs = rec.probabilities(np.zeros((1, 28, 28)), allowed=Recogniser.DIGITS)
    assert probs.sum() == pytest.approx(1.0)
    assert probs[0, EXT_LABELS.index("/")] == 0.0
    assert rec.classify(np.zeros((1, 28, 28)), Recogniser.DIGITS)[0]["label"] == "1"


def test_unknown_allowed_class_rejected():
    rec = Recogniser("fake", model=FakeModel(np.ones(10) / 10))
    with pytest.raises(ValueError):
        rec.classify(np.zeros((1, 28, 28)), allowed=["+"])


def test_input_scaled_and_channel_added():
    model = FakeModel(np.ones(10) / 10)
    Recogniser("fake", model=model).classify(np.full((3, 28, 28), 255, np.uint8))
    assert model.seen.shape == (3, 28, 28, 1) and model.seen.max() <= 1.0


def test_read_image_orders_boxes_left_to_right():
    # Three dark blobs on white paper; the fake model reads every one as "7".
    img = np.full((120, 360), 255, np.uint8)
    for x in (260, 30, 150):
        cv2.putText(img, "8", (x, 95), cv2.FONT_HERSHEY_SIMPLEX, 3, 0, 8)
    rec = Recogniser("fake", model=FakeModel(one_hot_ish(DEFAULT_LABELS, "7", "1")))
    text, details = rec.read_image(img)
    assert text == "777"
    xs = [d["box"][0] for d in details]
    assert xs == sorted(xs)


def test_label_file_fallback(tmp_path):
    ckpt = tmp_path / "old.keras"
    ckpt.write_bytes(b"")
    assert load_label_names(ckpt) == DEFAULT_LABELS
    ckpt.with_suffix(".labels.json").write_text('{"label_names": ["a", "b"]}')
    assert load_label_names(ckpt) == ["a", "b"]


def test_soften_is_on_only_for_operator_models():
    assert not Recogniser("fake", model=FakeModel(np.ones(10) / 10)).soften
    ext = Recogniser("fake", model=FakeModel(np.ones(16) / 16), label_names=EXT_LABELS)
    assert ext.soften
    assert not Recogniser("fake", model=FakeModel(np.ones(16) / 16),
                          label_names=EXT_LABELS, soften=False).soften


def test_soften_blurs_hard_edges_and_keeps_shape():
    patch = np.zeros((1, 28, 28, 1), "float32")
    patch[0, 4:24, 13:15, 0] = 1.0                 # a crisp vertical stroke
    out = soften(patch)
    assert out.shape == patch.shape
    assert 0 < out[0, 14, 12, 0] < 1               # edge pixel now partly lit


def test_input_blur_flag_from_label_file(tmp_path):
    from inference import load_input_blur
    ckpt = tmp_path / "m.keras"
    assert load_input_blur(ckpt) is None
    ckpt.with_suffix(".labels.json").write_text('{"label_names": ["0"], "input_blur": false}')
    assert load_input_blur(ckpt) is False


def test_frozen_label_files_name_a_registered_dataset():
    """evaluate_classes.py loads the test set named in each frozen model's label file."""
    import json
    from datasets import available_datasets
    from inference import CHECKPOINTS
    for label_file in CHECKPOINTS.glob("*.labels.json"):
        meta = json.loads(label_file.read_text())
        assert meta.get("dataset", "mnist") in available_datasets(), label_file.name


# number segmentation: John's repaired segmentation is the default

def _blobs(xs=(30, 150, 260)):
    img = np.full((120, 360, 3), 255, np.uint8)
    for x in xs:
        cv2.putText(img, "8", (x, 95), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
    return img


def test_numbers_use_repaired_segmentation_unless_told_otherwise(monkeypatch):
    import inference as inf
    seen = []

    def fake_segment_number(image, method=None, reader=None):
        seen.append(method)
        return [image[0:50, 0:50]], [(0, 0, 50, 50)]

    monkeypatch.setattr(inf, "segment_number", fake_segment_number)
    rec = Recogniser("fake", model=FakeModel(one_hot_ish(DEFAULT_LABELS, "7", "1")))
    rec.read_image(_blobs())
    rec.read_image(_blobs(), method="contours")
    assert inf.NUMBER_SEGMENTATION == "repaired" and seen == [None, "contours"]


def test_expressions_keep_the_operator_segmentation(monkeypatch):
    import inference as inf
    monkeypatch.setattr(inf, "segment_number",
                        lambda *a, **k: pytest.fail("expressions must not use segment_number"))
    rec = Recogniser("fake", model=FakeModel(one_hot_ish(EXT_LABELS, "7", "1")),
                     label_names=EXT_LABELS)
    text, _ = rec.read_image(_blobs(), method="repaired")      # ignored for expressions
    assert text == "777"


def test_contours_option_matches_task2_segmentation():
    import inference as inf
    import segmentation
    img = _blobs()
    crops, boxes = inf.segment_number(img, method="contours")
    assert boxes == [tuple(int(v) for v in b) for b in segmentation.segment_digits(img)[1]]
    assert all(c.shape[:2] == (h, w) for c, (x, y, w, h) in zip(crops, boxes))


def test_repaired_segmentation_cuts_a_touching_pair():
    """Two touching blobs, one connected region: contours gives 1 box, repaired 2.

    The scorer is injected (a reader named like the scorer model), so no checkpoint is loaded.
    """
    import inference as inf
    img = np.full((140, 300, 3), 255, np.uint8)
    cv2.putText(img, "0", (40, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.4, (0, 0, 0), 9)
    cv2.putText(img, "0", (92, 110), cv2.FONT_HERSHEY_SIMPLEX, 3.4, (0, 0, 0), 9)
    assert len(inf.segment_number(img, method="contours")[1]) == 1
    reader = Recogniser(inf.SEGMENTATION_SCORER, model=FakeModel(one_hot_ish(DEFAULT_LABELS, "0", "8", 0.9, 0.05)))
    crops, boxes = inf.segment_number(img, reader=reader)
    assert len(boxes) == 2 and boxes[0][0] < boxes[1][0]
