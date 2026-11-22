#!/usr/bin/env python3
"""Fetch a static ffmpeg (with libx264) into assets/ for machines without one.

    python scripts/get_ffmpeg.py

Uses imageio-ffmpeg, which ships a self-contained binary for linux/macos/win.
This is what the reference runs used; any system ffmpeg with libx264 works too
(`--ffmpeg /path/to/ffmpeg` or `$POSE_FFMPEG`).
"""
import os
import shutil
import subprocess
import sys

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def main() -> int:
    if shutil.which("ffmpeg"):
        print("system ffmpeg already on PATH:", shutil.which("ffmpeg"))
        return 0
    try:
        import imageio_ffmpeg
    except ImportError:
        print("installing imageio-ffmpeg ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"])
        import imageio_ffmpeg
    src = imageio_ffmpeg.get_ffmpeg_exe()
    os.makedirs(ASSETS, exist_ok=True)
    dst = os.path.join(ASSETS, os.path.basename(src))
    shutil.copy2(src, dst)
    os.chmod(dst, 0o755)
    print("ffmpeg copied to", dst)
    print("check:", subprocess.run([dst, "-version"], capture_output=True, text=True).stdout.splitlines()[0])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
