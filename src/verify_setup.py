"""
Environment verification.
COS30018 Option B - Handwritten Number Recognition System

Run this first, on every machine, before doing anything else:
    python src/verify_setup.py

If every line says OK, the environment is ready.
"""

from __future__ import annotations

import platform
import sys

REQUIRED = [
    ("numpy", "numpy"),
    ("opencv-python", "cv2"),
    ("scikit-image", "skimage"),
    ("scikit-learn", "sklearn"),
    ("matplotlib", "matplotlib"),
    ("Pillow", "PIL"),
]
OPTIONAL = [
    ("tensorflow", "tensorflow"),   # needed by Thien and Liam, not by John
    ("pytest", "pytest"),
]


def check(label: str, module: str, required: bool = True) -> bool:
    try:
        mod = __import__(module)
        version = getattr(mod, "__version__", "unknown")
        print(f"  OK       {label:16s} {version}")
        return True
    except ImportError:
        tag = "MISSING " if required else "optional"
        print(f"  {tag} {label:16s} not installed")
        return not required


def main() -> int:
    print("Environment check")
    print("-" * 46)
    print(f"  Python  {sys.version.split()[0]}  on  {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 9):
        print("  WARNING: Python 3.9+ recommended for this project.")
    print()

    print("Required packages:")
    required_ok = all(check(label, mod, True) for label, mod in REQUIRED)
    print()
    print("Optional packages:")
    for label, mod in OPTIONAL:
        check(label, mod, False)
    print("-" * 46)

    if required_ok:
        print("All required packages present. Environment is ready.")
        print("Next: python src/prepare_data.py")
        return 0

    print("Some required packages are missing. With your venv active, run:")
    print("    pip install -r requirements.txt")
    return 1


if __name__ == "__main__":
    sys.exit(main())
