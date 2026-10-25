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


# MLP / SVM / Random Forest options

GUI_OPTIONS = ["CNN", "CNN + operators", "MLP", "SVM", "Random Forest"]   # gui.MODEL_OPTIONS


def test_every_gui_option_has_a_backend():
    assert all(gb.is_supported(name) for name in GUI_OPTIONS)
    assert gb.is_classical("SVM") and gb.is_classical("Random Forest")
    assert not gb.is_classical("MLP") and not gb.is_cnn("MLP")
    assert gb.default_checkpoint("SVM").name == "hog_svm.joblib"
    assert gb.default_checkpoint("Random Forest").name == "hog_rf.joblib"
    assert gb.default_checkpoint("MLP").name == "mlp_best.keras"
    assert gb.training_steps("SVM", {"epochs": 10}) == 1
    assert gb.training_steps("MLP", {"epochs": 10}) == 10


def _tiny_baseline(key="hog_svm"):
    """A HOG classifier fitted on the synthetic squares (no MNIST needed)."""
    b = gb._baselines()
    data = _synthetic()
    clf = b.make_classifier("svm" if key == "hog_svm" else "rf",
                            {"C": 5.0, "gamma": "scale"} if key == "hog_svm"
                            else {"n_estimators": 20, "max_features": "sqrt",
                                  "min_samples_leaf": 1},
                            b.hog_features(data.x_train, "hog7"))
    clf.fit(b.hog_features(data.x_train, "hog7"), data.y_train)
    return b.Baseline(key, "hog7", clf)


def test_classical_model_reads_a_photo_through_the_same_pipeline():
    rec = gb.classical_recogniser(_tiny_baseline(), "hog_svm")
    out = gb.recognise(rec, _photo("38"))
    assert len(out["text"]) == len(out["results"]) == len(out["boxes"]) == 2
    assert all(ch in "0123456789" for ch in out["text"])
    assert all(0.0 <= c <= 1.0 for _, c in out["results"])
    assert out["thumbs"][0].size == (28, 28)


@pytest.mark.parametrize("option,key", [("SVM", "hog_svm"), ("Random Forest", "hog_rf")])
def test_train_classical_reports_once_saves_and_reloads(tmp_path, option, key):
    seen = []
    rec = gb.train_model(option, {"epochs": 5}, lambda *a: seen.append(a),
                         threading.Event(), data=_synthetic(), output_root=tmp_path)
    assert len(seen) == 1 and seen[0][0] == 1 and 0.0 <= seen[0][2] <= 1.0
    saved = list((tmp_path / "checkpoints").glob(f"gui_{key}_*.joblib"))
    assert len(saved) == 1
    again = gb.load_model(option, str(saved[0]))
    patches = _synthetic().x_test
    np.testing.assert_allclose(again.probabilities(patches), rec.probabilities(patches))


def test_classical_option_rejects_the_other_classical_model(tmp_path):
    import joblib
    path = tmp_path / "forest.joblib"
    joblib.dump(_tiny_baseline("hog_rf"), path)
    with pytest.raises(ValueError):
        gb.load_model("SVM", str(path))
    assert gb.load_model("Random Forest", str(path)).labels == DEFAULT_LABELS


def test_train_mlp_always_uses_the_mlp_architecture(tmp_path):
    params = {"epochs": 1, "batch_size": 32, "learning_rate": 1e-3,
              "arch": "vgg_deep", "augmentation": "none"}    # CNN arch box is ignored
    gb.train_model("MLP", params, lambda *a: None, threading.Event(),
                   data=_synthetic(), output_root=tmp_path)
    assert list((tmp_path / "checkpoints" / "tuning").glob("gui_mlp_*.keras"))


@pytest.mark.parametrize("option", ["MLP", "SVM", "Random Forest"])
def test_frozen_models_load_and_read_a_photo(option):
    if not gb.default_checkpoint(option).is_file():
        pytest.skip(f"{gb.default_checkpoint(option).name} not in models/checkpoints/")
    out = gb.recognise(gb.load_model(option), _photo("38"))
    assert len(out["results"]) == 2


# segmentation choice and Generate from folder

def test_gui_segmentation_choice_reaches_read_image():
    seen = []

    class Spy(Recogniser):
        def read_image(self, image, allowed=None, method=None):
            seen.append(method)
            return super().read_image(image, allowed, method="contours")

    rec = Spy("fake", model=FakeModel(one_hot_ish(DEFAULT_LABELS, "7", "1")))
    for label, method in gb.SEGMENTATIONS.items():
        gb.recognise(rec, _photo("38"), method)
    assert seen == ["repaired", "contours"] and list(gb.SEGMENTATIONS)[0].startswith("Repaired")


def test_generate_from_folder_gives_an_image_the_gui_can_show():
    if not gb.DEFAULT_DIGIT_FOLDER.is_dir():
        pytest.skip("team digit photos not present")
    img, text = gb.generate_from_folder(text="4096", seed=0)
    assert text == "4096" and isinstance(img, Image.Image) and img.width > img.height
    img, text = gb.generate_from_folder(text="", n_digits=3, expression=True, seed=1)
    assert any(c in text for c in "+-*/")


def test_low_confidence_symbols_are_flagged():
    results = [("8", 0.99), ("9", 0.62), ("6", 0.95), ("1", 0.899)]
    assert gb.uncertain(results) == [1, 3]
    assert gb.uncertain(results, threshold=0.5) == []
    assert gb.LOW_CONFIDENCE == 0.9
