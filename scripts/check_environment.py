"""Report Python, package, and available accelerator details."""

import importlib.metadata
import sys


REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "Pillow",
    "pandas",
    "jiwer",
    "datasets",
    "accelerate",
)


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if sys.version_info[:2] != (3, 11):
        print("ERROR: This project targets Python 3.11.")
        python_ok = False
    else:
        python_ok = True

    missing = []
    for package in REQUIRED_PACKAGES:
        try:
            print(f"{package}: {importlib.metadata.version(package)}")
        except importlib.metadata.PackageNotFoundError:
            print(f"{package}: MISSING")
            missing.append(package)

    if missing:
        print("ERROR: Install missing packages with: python -m pip install -r requirements.txt")
        return 1

    try:
        import torch
    except ImportError as error:
        print(f"ERROR: Could not import PyTorch: {error}")
        return 1

    if torch.cuda.is_available():
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        print("Device: CPU (CUDA unavailable; CPU inference/training is supported)")
    return 0 if python_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
