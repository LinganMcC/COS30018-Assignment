"""Tests for inference.Recogniser using a fake model (no checkpoint needed)."""
import cv2
import numpy as np
import pytest

from inference import DEFAULT_LABELS, Recogniser, load_label_names

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
