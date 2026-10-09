"""Tests for src/download_symbols.py using small fake copies of both datasets.
"""

import csv
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import download_symbols as ds  # noqa: E402

PNG = b"\x89PNG\r\n\x1a\nfake"


def _fake_hasy(tmp_path: Path) -> Path:
    """Tar with 2x '+', 1x '\\times', 1x '\\div', 1x '\\alpha' (not wanted)."""
    rows = [("hasy-data/v2-0.png", "+", "7"), ("hasy-data/v2-1.png", "+", "8"),
            ("hasy-data/v2-2.png", "\\times", "7"), ("hasy-data/v2-3.png", "\\div", "9"),
            ("hasy-data/v2-4.png", "\\alpha", "7")]
    archive = tmp_path / "HASYv2.tar.bz2"
    with tarfile.open(archive, "w:bz2") as tar:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
        for path, _, _ in rows:
            add(path, PNG)
        text = "path,symbol_id,latex,user_id\n" + "".join(
            f"{p},1,{l},{u}\n" for p, l, u in rows)
        add("hasy-data-labels.csv", text.encode())
    return archive


def _fake_kaggle(tmp_path: Path) -> Path:
    root = tmp_path / "kaggle" / "extracted_images"
    for name, n in [("(", 2), (")", 1), ("-", 3), ("alpha", 2)]:
        (root / name).mkdir(parents=True)
        for i in range(n):
            (root / name / f"img_{i}.jpg").write_bytes(b"jpg")
    return tmp_path / "kaggle"


def _manifest(out: Path) -> list[dict]:
    with (out / "manifest.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def test_hasy_scan_finds_only_wanted_classes(tmp_path):
    found = ds.scan_hasy(_fake_hasy(tmp_path))
    assert len(found["plus"]) == 2
    assert len(found["times"]) == 1
    assert len(found["div"]) == 1
    assert found["minus"] == []          # not in the fake archive
    assert sum(len(v) for v in found.values()) == 4   # \alpha ignored


def test_hasy_extract_writes_images_and_writer_ids(tmp_path):
    archive = _fake_hasy(tmp_path)
    out = tmp_path / "symbols"
    rows = ds.extract_hasy(archive, ds.scan_hasy(archive), out)
    assert len(rows) == 4
    assert (out / "plus" / "hasy_v2-0.png").read_bytes() == PNG
    assert {r["writer"] for r in rows if r["folder"] == "plus"} == {"7", "8"}
    assert {r["label"] for r in rows} == {"+", "*", "/"}


def test_kaggle_scan_finds_nested_folders(tmp_path):
    found = ds.scan_kaggle(_fake_kaggle(tmp_path))
    assert len(found["lparen"]) == 2
    assert len(found["rparen"]) == 1
    assert len(found["minus"]) == 3
    assert found["plus"] == []


def test_output_folder_names_are_windows_safe():
    bad = set('<>:"/\\|?*')
    for folder, *_ in ds.CLASSES:
        assert not (set(folder) & bad), folder


def test_labels_match_the_trainer():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "models" / "cnn"))
    from datasets import SYMBOL_LABELS
    assert sorted(label for _, label, *_ in ds.CLASSES) == sorted(SYMBOL_LABELS)


def test_cap_per_class_is_deterministic():
    found = {"plus": list(range(100)), "minus": list(range(5))}
    a = ds.cap_per_class(found, 10)
    b = ds.cap_per_class(found, 10)
    assert a == b and len(a["plus"]) == 10 and len(a["minus"]) == 5


def test_auto_mode_uses_both_sources_for_every_class(tmp_path, monkeypatch):
    archive = _fake_hasy(tmp_path)
    monkeypatch.setattr(ds, "download_hasy", lambda: archive)
    kaggle = _fake_kaggle(tmp_path)
    plus = kaggle / "extracted_images" / "+"          # '+' in BOTH sources
    plus.mkdir()
    (plus / "k.jpg").write_bytes(b"jpg")
    out = tmp_path / "symbols"
    ds.main(["--kaggle-dir", str(kaggle), "--out", str(out)])
    rows = _manifest(out)
    by_source = {(r["folder"], r["source"]) for r in rows}
    assert ("plus", "hasy") in by_source and ("plus", "kaggle") in by_source
    assert ("minus", "kaggle") in by_source
    assert ("lparen", "kaggle") in by_source
    assert all((out / r["file"]).exists() for r in rows)


def test_local_kaggle_folder_is_found_without_flag(tmp_path, monkeypatch):
    archive = _fake_hasy(tmp_path)
    monkeypatch.setattr(ds, "download_hasy", lambda: archive)
    monkeypatch.setattr(ds, "KAGGLE_LOCAL_DIR", _fake_kaggle(tmp_path))
    monkeypatch.setattr(ds, "download_kaggle",
                        lambda: pytest.fail("should not try to download"))
    out = tmp_path / "symbols"
    ds.main(["--out", str(out)])
    assert any(r["source"] == "kaggle" for r in _manifest(out))


def test_missing_kagglehub_does_not_crash_auto_mode(tmp_path, monkeypatch):
    archive = _fake_hasy(tmp_path)
    monkeypatch.setattr(ds, "download_hasy", lambda: archive)
    monkeypatch.setattr(ds, "KAGGLE_LOCAL_DIR", tmp_path / "nowhere")
    def no_kagglehub():
        raise RuntimeError("kagglehub is not installed")
    monkeypatch.setattr(ds, "download_kaggle", no_kagglehub)
    out = tmp_path / "symbols"
    ds.main(["--out", str(out)])
    assert {r["source"] for r in _manifest(out)} == {"hasy"}


def test_check_mode_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "KAGGLE_LOCAL_DIR", tmp_path / "nowhere")
    archive = _fake_hasy(tmp_path)
    monkeypatch.setattr(ds, "download_hasy", lambda: archive)
    out = tmp_path / "symbols"
    ds.main(["--check", "--kaggle-dir", str(_fake_kaggle(tmp_path)), "--out", str(out)])
    assert not out.exists()


def test_rerun_does_not_duplicate_manifest_rows(tmp_path, monkeypatch):
    archive = _fake_hasy(tmp_path)
    monkeypatch.setattr(ds, "download_hasy", lambda: archive)
    out = tmp_path / "symbols"
    args = ["--source", "hasy", "--out", str(out)]
    ds.main(args)
    ds.main(args)
    assert len(_manifest(out)) == 4


def test_hasy_failure_falls_back_to_kaggle(tmp_path, monkeypatch):
    def boom():
        raise OSError("no network")
    monkeypatch.setattr(ds, "download_hasy", boom)
    out = tmp_path / "symbols"
    ds.main(["--kaggle-dir", str(_fake_kaggle(tmp_path)), "--out", str(out)])
    assert {r["source"] for r in _manifest(out)} == {"kaggle"}


def test_duplicate_kaggle_copies_are_skipped(tmp_path):
    """Partial zip copy + full rar copy of the same class -> each file once."""
    root = tmp_path / "xainano"
    for sub in ["extracted_images", "data/extracted_images"]:
        d = root / sub / "("
        d.mkdir(parents=True)
        for i in range(3):
            (d / f"exp_{i}.jpg").write_bytes(b"jpg")
    (root / "data/extracted_images/(" / "exp_9.jpg").write_bytes(b"jpg")  # only in full copy
    found = ds.scan_kaggle(root)
    assert sorted(p.name for p in found["lparen"]) == ["exp_0.jpg", "exp_1.jpg", "exp_2.jpg", "exp_9.jpg"]


def test_clean_removes_stale_files(tmp_path, monkeypatch):
    archive = _fake_hasy(tmp_path)
    monkeypatch.setattr(ds, "download_hasy", lambda: archive)
    out = tmp_path / "symbols"
    stale = out / "plus" / "kaggle_99999.jpg"
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"old")
    ds.main(["--source", "hasy", "--out", str(out), "--clean"])
    assert not stale.exists()
    assert len(_manifest(out)) == 4
