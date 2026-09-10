"""Portable Shorts Studio launcher used by the Windows .exe build."""
from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path


def main() -> None:
    root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    os.chdir(root)
    # Prefer the bundled ffmpeg/ffprobe binaries in the portable release.
    os.environ["PATH"] = str(root) + os.pathsep + os.environ.get("PATH", "")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import uvicorn

    def open_browser() -> None:
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:7860")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("web.app:app", host="127.0.0.1", port=7860, log_level="info")


if __name__ == "__main__":
    main()
