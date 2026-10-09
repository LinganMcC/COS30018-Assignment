import argparse
import csv
import io
import json
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "symbols"
CACHE_DIR = ROOT / "data" / "downloads"
MANIFEST_FIELDS = ["file", "label", "folder", "source", "writer"]

HASY_RECORD_API = "https://zenodo.org/api/records/259444"
KAGGLE_DATASET = "xainano/handwrittenmathsymbols"
KAGGLE_LOCAL_DIR = CACHE_DIR / "xainano"     # where a manual download is looked for
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

# One entry per class: (folder, label, names in HASYv2, folder names in Kaggle).
# Folder names are spelled out because "*" and "/" are not allowed in Windows
# folder names. The label is what the model outputs and must match
# SYMBOL_LABELS in models/cnn/datasets.py.
# Multiplication accepts both the cross (x) and the asterisk (*): the spec's
# example image writes it as a star, but most datasets only have the cross.
CLASSES = [
    ("plus",   "+", ["+"],                    ["+"]),
    ("minus",  "-", ["-"],                    ["-"]),
    ("times",  "*", ["\\times", "\\ast", "*"], ["times"]),
    ("div",    "/", ["\\div"],                ["div"]),
    ("lparen", "(", ["("],                    ["("]),
    ("rparen", ")", [")"],                    [")"]),
]


# HASYv2

def download_hasy(cache_dir: Path = CACHE_DIR) -> Path:
    """Download the HASYv2 archive once and return its local path."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = sorted(cache_dir.glob("HASYv2*.tar*"))
    if cached:
        print(f"Using cached {cached[0].name}")
        return cached[0]

    # Ask Zenodo for the file list instead of hard-coding the file URL.
    with urllib.request.urlopen(HASY_RECORD_API, timeout=60) as resp:
        record = json.load(resp)
    files = [f for f in record["files"] if ".tar" in f["key"]]
    if not files:
        raise RuntimeError("No .tar archive found in the HASYv2 Zenodo record.")
    entry = files[0]
    target = cache_dir / entry["key"]
    print(f"Downloading {entry['key']} ({entry['size'] / 1e6:.0f} MB) ...")
    urllib.request.urlretrieve(entry["links"]["self"], target)
    return target


def _read_hasy_labels(archive: Path) -> tuple[str, list[dict]]:
    """Find hasy-data-labels.csv inside the archive; return (its folder, rows)."""
    with tarfile.open(archive) as tar:
        for member in tar:
            if member.name.endswith("hasy-data-labels.csv"):
                text = tar.extractfile(member).read().decode("utf-8")
                folder = member.name.rsplit("/", 1)[0] if "/" in member.name else ""
                return folder, list(csv.DictReader(io.StringIO(text)))
    raise RuntimeError("hasy-data-labels.csv not found in the archive.")


def scan_hasy(archive: Path) -> dict[str, list[dict]]:
    """Map each of our folders to the matching HASYv2 rows (nothing is written)."""
    base, rows = _read_hasy_labels(archive)
    wanted = {name: folder for folder, _, names, _ in CLASSES for name in names}
    found: dict[str, list[dict]] = {folder: [] for folder, *_ in CLASSES}
    for row in rows:
        folder = wanted.get(row["latex"])
        if folder:
            member = f"{base}/{row['path']}" if base else row["path"]
            found[folder].append({"member": member.replace("\\", "/"),
                                  "writer": row.get("user_id", "")})
    return found


def extract_hasy(archive: Path, found: dict[str, list[dict]],
                 out_dir: Path) -> list[dict]:
    """Copy the selected images out of the archive into out_dir/<folder>/."""
    lookup = {item["member"]: (folder, item["writer"])
              for folder, items in found.items() for item in items}
    labels = {folder: label for folder, label, *_ in CLASSES}
    written = []
    with tarfile.open(archive) as tar:
        for member in tar:
            hit = lookup.get(member.name)
            if not hit:
                continue
            folder, writer = hit
            # The output name is built by us - never from the archive path -
            # so a malformed archive cannot write outside out_dir.
            stem = Path(member.name).stem
            name = f"hasy_{stem}{Path(member.name).suffix.lower() or '.png'}"
            dest = out_dir / folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(tar.extractfile(member).read())
            written.append({"file": f"{folder}/{name}", "label": labels[folder],
                            "folder": folder, "source": "hasy", "writer": writer})
    return written


# Kaggle

def download_kaggle() -> Path:
    """Download the Kaggle dataset with kagglehub and return its folder."""
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "kagglehub is not installed. Run `pip install kagglehub`, or download "
            f"https://www.kaggle.com/datasets/{KAGGLE_DATASET} by hand and pass "
            "--kaggle-dir.") from exc
    return Path(kagglehub.dataset_download(KAGGLE_DATASET))


def scan_kaggle(root: Path) -> dict[str, list[Path]]:
    """Find our class folders anywhere under root (the dataset nests them)."""
    wanted = {name: folder for folder, _, _, names in CLASSES for name in names}
    found: dict[str, list[Path]] = {folder: [] for folder, *_ in CLASSES}
    # The Kaggle download can hold the same class twice (a partial
    # extracted_images/ from the zip AND the full one from data.rar). Copying
    # both would put identical images in train and test, so each file name is
    # kept only once per class.
    seen: dict[str, set[str]] = {folder: set() for folder in found}
    skipped = 0
    for directory in sorted([root, *root.rglob("*")]):
        if directory.is_dir() and directory.name in wanted:
            folder = wanted[directory.name]
            for p in sorted(directory.iterdir()):
                if p.suffix.lower() not in IMAGE_EXTS:
                    continue
                if p.name in seen[folder]:
                    skipped += 1
                    continue
                seen[folder].add(p.name)
                found[folder].append(p)
    if skipped:
        print(f"Skipped {skipped} duplicate images (same class found in more than one folder).")
    if not any(found.values()) and list(root.rglob("*.rar")):
        print(f"Only a .rar archive was found in {root}. Extract it, then re-run "
              "with --kaggle-dir pointing at the extracted folder.")
    return found


def copy_kaggle(found: dict[str, list[Path]], out_dir: Path) -> list[dict]:
    labels = {folder: label for folder, label, *_ in CLASSES}
    written = []
    for folder, paths in found.items():
        for i, src in enumerate(paths):
            name = f"kaggle_{i:05d}{src.suffix.lower()}"
            dest = out_dir / folder / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            written.append({"file": f"{folder}/{name}", "label": labels[folder],
                            "folder": folder, "source": "kaggle", "writer": ""})
    return written


# Shared helpers

def cap_per_class(found: dict[str, list], limit: int | None,
                  seed: int = 42) -> dict[str, list]:
    """Randomly keep at most `limit` items per class (same picks every run)."""
    if not limit:
        return found
    rng = random.Random(seed)
    return {k: (rng.sample(v, limit) if len(v) > limit else v) for k, v in found.items()}


def report(source: str, found: dict[str, list]) -> list[str]:
    """Print per-class counts and return the folders that have no images."""
    print(f"\n{source}:")
    missing = []
    for folder, label, *_ in CLASSES:
        n = len(found.get(folder, []))
        print(f"  {label:>2}  ({folder:6s})  {n:6d}" + ("   <- MISSING" if n == 0 else ""))
        if n == 0:
            missing.append(folder)
    return missing


def write_manifest(rows: list[dict], out_dir: Path) -> Path:
    """Merge new rows into manifest.csv, keeping only files that exist."""
    path = out_dir / "manifest.csv"
    merged = {}
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            merged = {r["file"]: r for r in csv.DictReader(f)}
    merged.update({r["file"]: r for r in rows})
    keep = [r for f, r in sorted(merged.items()) if (out_dir / f).exists()]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(keep)
    return path


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Download operator symbol images.")
    parser.add_argument("--source", choices=["auto", "hasy", "kaggle"], default="auto",
                        help="auto = both HASYv2 and Kaggle for every class")
    parser.add_argument("--kaggle-dir", type=Path,
                        help="use an already downloaded/extracted Kaggle folder")
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--check", action="store_true",
                        help="only report which classes each source has")
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--clean", action="store_true",
                        help="delete the output folder first (use when re-running)")
    args = parser.parse_args(argv)

    if args.clean and not args.check and args.out.exists():
        print(f"Removing old {args.out} ...")
        shutil.rmtree(args.out)

    rows: list[dict] = []
    missing = [folder for folder, *_ in CLASSES]

    if args.source in ("auto", "hasy"):
        try:
            archive = download_hasy()
        except Exception as exc:          # e.g. no internet access to Zenodo
            if args.source == "hasy":
                raise
            print(f"HASYv2 unavailable ({exc}); falling back to Kaggle for all classes.")
            archive = None
        if archive is not None:
            found = cap_per_class(scan_hasy(archive), args.max_per_class)
            missing = report("HASYv2", found)
            if not args.check:
                rows += extract_hasy(archive, found, args.out)

    if args.source in ("auto", "kaggle"):
        root = args.kaggle_dir
        if root is None and KAGGLE_LOCAL_DIR.exists():
            root = KAGGLE_LOCAL_DIR
            print(f"\nUsing the Kaggle download in {root}")
        if root is None:
            try:
                root = download_kaggle()
            except RuntimeError as exc:
                if args.source == "kaggle":
                    raise
                print(f"\nSkipping Kaggle: {exc}")
        if root is not None:
            found = cap_per_class(scan_kaggle(root), args.max_per_class)
            kaggle_missing = report("Kaggle", found)
            missing = [m for m in missing if m in kaggle_missing] \
                if args.source == "auto" else kaggle_missing
            if not args.check:
                rows += copy_kaggle(found, args.out)

    if args.check:
        print("\n--check: nothing was written.")
        return
    totals = {}
    for r in rows:
        totals[r["label"]] = totals.get(r["label"], 0) + 1
    print("\nTotal per class: " + "  ".join(f"{label} {totals.get(label, 0)}"
                                           for _, label, *_ in CLASSES))
    manifest = write_manifest(rows, args.out)
    print(f"\nWrote {len(rows)} images to {args.out}  (manifest: {manifest.name})")
    if missing:
        print(f"No images found for: {', '.join(missing)} - these classes need "
              "another source or the team's own handwritten samples.")


if __name__ == "__main__":
    main()
