"""One-shot lip-to-text inference on a single video file.

Usage:
    python scripts/run_inference.py path/to/video.mp4
    python scripts/run_inference.py path/to/video.mp4 --modality v   # visual only
    python scripts/run_inference.py path/to/video.mp4 --backend mock # smoke test

Backend selection follows config.AVSR_BACKEND unless --backend overrides.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path, help="Path to video file")
    parser.add_argument("--modality", choices=["v", "a", "av"], default=None)
    parser.add_argument("--backend", default=None, help="Override LIPSYNC_BACKEND")
    args = parser.parse_args()

    if not args.video.exists():
        print(f"Video not found: {args.video}", file=sys.stderr)
        return 1

    if args.backend:
        os.environ["LIPSYNC_BACKEND"] = args.backend

    # Import AFTER setting env vars so config picks them up.
    import avsr  # noqa: E402

    try:
        result = avsr.transcribe(args.video, mode=args.modality)
    except avsr.NotInstalled as e:
        print(f"Backend not installed:\n  {e}", file=sys.stderr)
        return 2

    print(f"Backend: {result.model_id}")
    print(f"Mode:    {result.mode}")
    print(f"Text:    {result.text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
