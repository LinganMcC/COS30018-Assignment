"""Tests for gui_backend.py - the CNN side of the GUI, without opening a window."""
import threading

import cv2
import numpy as np
import pytest
from PIL import Image

import gui_backend as gb
from inference import DEFAULT_LABELS, Recogniser
from test_inference import EXT_LABELS, FakeModel, one_hot_ish
from test_trainer import _synthetic


def _photo(text="38", size=(140, 300)):
    """Dark handwriting-like text on white paper, as a PIL image."""
    img = np.full(size + (3,), 255, np.uint8)
    cv2.putText(img, text, (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (30, 30, 30), 9)
    return Image.fromarray(img)


def test_only_cnn_options_are_routed_here():
    assert gb.is_cnn("CNN") and gb.is_cnn("CNN + operators")
    assert not gb.is_cnn("MLP") and not gb.is_cnn("SVM")


def test_recognise_returns_everything_the_gui_draws():
    rec = Recogniser("fake", model=FakeModel(one_hot_ish(DEFAULT_LABELS, "7", "1")))
    out = gb.recognise_cnn(rec, _photo("38"))
    assert out["text"] == "77"
    assert len(out["results"]) == len(out["boxes"]) == len(out["thumbs"]) == 2
    x0, y0, x1, y1 = out["boxes"][0]
    assert x1 > x0 and y1 > y0 and out["boxes"][0][0] < out["boxes"][1][0]
    assert out["thumbs"][0].size == (28, 28)
    assert out["binary"].size == (300, 140)


def test_operator_model_keeps_a_minus_sign():
    """The digit segmenter drops a short '-'; the operator model must keep it."""
    img = np.full((140, 420, 3), 255, np.uint8)
    cv2.putText(img, "8", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (30, 30, 30), 9)
    cv2.line(img, (170, 70), (230, 70), (30, 30, 30), 9)
    cv2.putText(img, "3", (290, 105), cv2.FONT_HERSHEY_SIMPLEX, 3.2, (30, 30, 30), 9)
    digits = Recogniser("fake", model=FakeModel(one_hot_ish(DEFAULT_LABELS, "7", "1")))
    ext = Recogniser("fake", model=FakeModel(one_hot_ish(EXT_LABELS, "7", "1")),
                     label_names=EXT_LABELS)
    assert len(gb.recognise_cnn(digits, Image.fromarray(img))["boxes"]) == 2
    assert len(gb.recognise_cnn(ext, Image.fromarray(img))["boxes"]) == 3


def test_train_cnn_reports_epochs_stops_and_returns_a_recogniser(tmp_path):
    seen, stop = [], threading.Event()

    def on_epoch(epoch, loss, acc):
        seen.append((epoch, loss, acc))
        if epoch == 2:
            stop.set()                      # the GUI's Stop button

    params = {"epochs": 6, "batch_size": 32, "learning_rate": 1e-3,
              "arch": "lenet", "augmentation": "none"}
    rec = gb.train_cnn("CNN", params, on_epoch, stop, data=_synthetic(),
                       output_root=tmp_path)
    assert [e for e, _, _ in seen] == [1, 2]
    assert all(0.0 <= acc <= 1.0 for _, _, acc in seen)
    assert isinstance(rec, Recogniser) and rec.labels == DEFAULT_LABELS
    assert list((tmp_path / "checkpoints" / "tuning").glob("gui_lenet_*.keras"))


def test_operator_option_rejects_a_digit_only_file(tmp_path):
    stop = threading.Event()
    params = {"epochs": 1, "batch_size": 32, "learning_rate": 1e-3, "arch": "lenet"}
    rec = gb.train_cnn("CNN", params, lambda *a: None, stop, data=_synthetic(),
                       output_root=tmp_path)
    path = next((tmp_path / "checkpoints" / "tuning").glob("*.keras"))
    with pytest.raises(ValueError):
        gb.load_cnn("CNN + operators", str(path))
    assert gb.load_cnn("CNN", str(path)).labels == rec.labels
