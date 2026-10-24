"""CLI Server Launcher for SmartVision Backend Service."""

import uvicorn
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from smartvision_backend.api import app


def main():
    print("\n[SmartVision Backend] Launching REST Service on http://127.0.0.1:8000 ...")
    uvicorn.run("smartvision_backend.api:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
