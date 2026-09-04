"""
Progress summary - one screenshot for the team and the tutor.
COS30018 Option B - Handwritten Number Recognition System
Owner: John (Person A)

Checks every deliverable this project should have produced so far and prints
whether it exists, so weekly progress can be demonstrated in a single command
instead of running each script again.

Run:
    python src/show_progress.py
"""

from __future__ import annotations

import csv
import subprocess
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

    # ---- Environment -----------------------------------------------------
    header("Environment")
    try:
        import cv2, numpy, sklearn, skimage        # noqa: F401
        print(f"  {status(True)} All required libraries import correctly")
    except ImportError as exc:                      # noqa: BLE001
        print(f"  {status(False)} Missing library: {exc.name}")

    # ---- Dataset ---------------------------------------------------------
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
    n_custom = count_files(custom, "*.png") + count_files(custom, "*.jpg")
    print(f"  {status(n_custom > 0)} Real handwritten test images: {n_custom} "
          f"{DIM}(needed before final technique selection){END}")

    # ---- Task 1 ----------------------------------------------------------
    header("Task 1 - Image Preprocessing (5 marks)")
    comp = ROOT / "reports" / "preprocessing_comparison.csv"
    if comp.exists():
        with open(comp, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        print(f"  {status(True)} {len(rows)} configurations compared quantitatively")
        for r in rows:
            print(f"      {r['config']:26s} accuracy {r['test_accuracy']}")
        best = max(rows, key=lambda r: float(r["test_accuracy"]))
        print(f"  {DIM}Best on MNIST: {best['config']} ({best['test_accuracy']}){END}")
    else:
        print(f"  {status(False)} Comparison not run yet")

    visual = ROOT / "reports" / "preprocessing_visual.png"
    print(f"  {status(visual.exists())} Visual before/after figure for the report")

    # ---- Task 2 ----------------------------------------------------------
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

    # ---- Tests -----------------------------------------------------------
    header("Testing")
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-q", "--no-header"],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        last = [ln for ln in result.stdout.strip().splitlines() if ln.strip()][-1]
        print(f"  {status(result.returncode == 0)} {last.strip()}")
    except Exception as exc:                        # noqa: BLE001
        print(f"  {status(False)} Could not run tests: {exc}")

    # ---- Version control -------------------------------------------------
    header("Version control")
    try:
        log = subprocess.run(["git", "log", "--oneline"], cwd=ROOT,
                             capture_output=True, text=True, timeout=20)
        commits = log.stdout.strip().splitlines()
        print(f"  {status(bool(commits))} {len(commits)} commits in local history")
        for line in commits[:8]:
            print(f"      {DIM}{line}{END}")

        remote = subprocess.run(["git", "remote", "-v"], cwd=ROOT,
                                capture_output=True, text=True, timeout=20)
        has_remote = bool(remote.stdout.strip())
        print(f"  {status(has_remote)} Pushed to GitHub remote"
              f"{'' if has_remote else f'  {DIM}(create repo and push, then invite tutor){END}'}")
    except Exception as exc:                        # noqa: BLE001
        print(f"  {status(False)} Git not available: {exc}")

    # ---- Documentation ---------------------------------------------------
    header("Documentation")
    worklog = ROOT / "reports" / "worklog_john.md"
    print(f"  {status(worklog.exists())} Individual worklog maintained")
    print(f"  {status((ROOT / 'README.md').exists())} README with setup and run instructions")

    print("\n" + "=" * 68)
    print(f"{DIM}Generated by src/show_progress.py{END}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
