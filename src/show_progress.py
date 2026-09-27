"""Print DONE/TODO for each deliverable.

Run:
    python src/show_progress.py
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

GREEN = "\033[92m"
RED = "\033[91m"
DIM = "\033[2m"
BOLD = "\033[1m"
END = "\033[0m"


def status(ok: bool) -> str:
    return f"{GREEN}[DONE]{END}" if ok else f"{RED}[TODO]{END}"


def count_files(folder: Path, pattern: str = "*") -> int:
    return len(list(folder.rglob(pattern))) if folder.exists() else 0


def header(text: str) -> None:
    print(f"\n{BOLD}{text}{END}")
    print("-" * 68)


def main() -> None:
    print("=" * 68)
    print(f"{BOLD}COS30018 Option B - HNRS - Progress Report{END}")
    print("Person A (John) - Data & Vision: Task 1, Task 2, image acquisition")
    print("=" * 68)

    # Environment
    header("Environment")
    try:
        import cv2, numpy, sklearn, skimage
        print(f"  {status(True)} All required libraries import correctly")
    except ImportError as exc:
        print(f"  {status(False)} Missing library: {exc.name}")

    # Dataset
    header("Dataset")
    raw = ROOT / "data" / "raw"
    n_raw = count_files(raw, "*.png")
    print(f"  {status(n_raw > 0)} Single-digit images exported: "
          f"{n_raw} files across {len(list(raw.glob('*'))) if raw.exists() else 0} digit folders")

    generated = ROOT / "data" / "generated"
    n_gen = count_files(generated, "number_*.png")
    print(f"  {status(n_gen > 0)} Multi-digit number images generated: {n_gen} images")

    labels = generated / "labels.csv"
    if labels.exists():
        with open(labels, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        sample = ", ".join(r["label"] for r in rows[:6])
        print(f"  {status(True)} Ground-truth labels recorded: {len(rows)} rows "
              f"{DIM}(e.g. {sample}){END}")
    else:
        print(f"  {status(False)} Ground-truth labels file missing")

    custom = ROOT / "data" / "custom_samples"
    n_digits = count_files(custom / "digits", "*.jpg") + count_files(custom / "digits", "*.png")
    n_numbers = count_files(custom / "numbers", "*.jpg") + count_files(custom / "numbers", "*.png")
    print(f"  {status(n_digits + n_numbers > 0)} Real handwritten test images: "
          f"{n_digits} single digits, {n_numbers} numbers")

    # Task 1
    header("Task 1 - Image Preprocessing (5 marks)")
    comp = ROOT / "reports" / "preprocessing_comparison.csv"
    if comp.exists():
        with open(comp, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        # The CSV format has changed as the experiment grew (k-NN, then one CNN
        # run, then repeated runs), so read whichever accuracy column is there.
        key = next(k for k in ("cnn_mean", "cnn_accuracy", "test_accuracy") if k in rows[0])
        runs = rows[0].get("runs", "1")
        print(f"  {status(True)} {len(rows)} configurations compared quantitatively "
              f"{DIM}({runs} run(s) each){END}")
        for r in rows:
            spread = f" +/- {r['cnn_std']}" if "cnn_std" in r else ""
            print(f"      {r['config']:26s} accuracy {r[key]}{spread}")
        best = max(rows, key=lambda r: float(r[key]))
        print(f"  {DIM}Highest on MNIST: {best['config']} ({best[key]}){END}")
    else:
        print(f"  {status(False)} Comparison not run yet")

    real = ROOT / "reports" / "custom_preprocessing.csv"
    if real.exists():
        with open(real, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        print(f"  {status(True)} Compared on real handwriting ({rows[0]['images']} photos)")
        for r in rows:
            mark = "  <- selected" if r.get("selected") else ""
            print(f"      {r['config']:26s} {r['correct']}/{r['images']} "
                  f"({float(r['accuracy']):.0%}){mark}")
    else:
        print(f"  {status(False)} Not yet compared on real handwriting")

    visual = ROOT / "reports" / "preprocessing_visual.png"
    print(f"  {status(visual.exists())} Visual before/after figure for the report")

    # Task 2
    header("Task 2 - Image Segmentation (5 marks)")
    seg = ROOT / "reports" / "segmentation_comparison.csv"
    if seg.exists():
        with open(seg, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        print(f"  {status(True)} {len(rows)} segmentation methods compared")
        for r in rows:
            print(f"      {r['method']:26s} {r['correct_digit_count']}/{r['images_tested']} "
                  f"correct digit count ({float(r['accuracy']):.0%})")
    else:
        print(f"  {status(False)} Comparison not run yet")

    demo = ROOT / "reports" / "segmentation_demo"
    n_demo = count_files(demo, "*.png")
    print(f"  {status(n_demo > 0)} Annotated demo images saved: {n_demo}")

    # Tests
    header("Testing")
    try:
        # sys.executable so pytest runs under the same interpreter as this script
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        last = [ln for ln in result.stdout.strip().splitlines() if ln.strip()][-1]
        print(f"  {status(result.returncode == 0)} {last.strip()}")
    except Exception as exc:
        print(f"  {status(False)} Could not run tests: {exc}")

    # Version control
    header("Version control")
    try:
        log = subprocess.run(["git", "log", "--oneline"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
        commits = log.stdout.strip().splitlines()
        print(f"  {status(bool(commits))} {len(commits)} commits in local history")
        for line in commits[:8]:
            print(f"      {DIM}{line}{END}")

        # the upstream only resolves once the branch has been pushed
        upstream = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        pushed = upstream.returncode == 0
        if pushed:
            print(f"  {status(True)} Branch pushed, tracking {upstream.stdout.strip()}")
        else:
            branch = subprocess.run(["git", "branch", "--show-current"], cwd=ROOT,
                                    capture_output=True, text=True, timeout=20)
            print(f"  {status(False)} Branch not pushed yet  "
                  f"{DIM}(git push -u origin {branch.stdout.strip()}){END}")
    except Exception as exc:
        print(f"  {status(False)} Git not available: {exc}")

    # Documentation
    header("Documentation")
    worklog = ROOT / "reports" / "worklog_john.md"
    print(f"  {status(worklog.exists())} Individual worklog maintained")
    print(f"  {status((ROOT / 'README.md').exists())} README with setup and run instructions")

    print("\n" + "=" * 68)
    print(f"{DIM}Generated by src/show_progress.py{END}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
