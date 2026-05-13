"""Webcam capture for VSR / AVSR data collection.

Two backends:
  - _FFmpegRecorder (default): AVFoundation video + audio, single mp4.
                                Requires `ffmpeg` on PATH (`brew install ffmpeg`).
  - _OpenCVRecorder (fallback): video-only, no audio.

Selection:
  LIPSYNC_AUDIO=0           → force OpenCV backend (no audio)
  LIPSYNC_FFMPEG_VIDEO_DEV  → AVFoundation video device index (default "0")
  LIPSYNC_FFMPEG_AUDIO_DEV  → AVFoundation audio device index (default "0")

To list devices:
  ffmpeg -f avfoundation -list_devices true -i ""

Both backends honor MAX_SECONDS as a hard cap and call the optional
`on_complete` callback when they auto-stop, so the main state machine can
transition out of RECORDING without requiring a second hotkey press.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

FPS = 30
WIDTH = 640
HEIGHT = 480
MAX_SECONDS = 8
AUDIO_SAMPLE_RATE = 16000  # matches AV-HuBERT / Auto-AVSR convention


@dataclass
class CaptureMeta:
    path: Path
    duration_seconds: float
    fps: int
    resolution: str
    audio_present: bool


def Recorder() -> "_OpenCVRecorder | _FFmpegRecorder":
    """Factory — pick backend based on env + ffmpeg availability."""
    if os.environ.get("LIPSYNC_AUDIO", "1") == "0":
        return _OpenCVRecorder()
    if shutil.which("ffmpeg") is None:
        print("[recorder] ffmpeg not found; falling back to video-only. "
              "Install with `brew install ffmpeg` for AVSR-quality capture.")
        return _OpenCVRecorder()
    return _FFmpegRecorder()


class _OpenCVRecorder:
    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._writer: cv2.VideoWriter | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._path: Path | None = None
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._on_complete: Callable[[], None] | None = None
        self._fired_complete = False

    def start(self, on_complete: Callable[[], None] | None = None) -> Path:
        self._stop.clear()
        self._on_complete = on_complete
        self._fired_complete = False
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("Could not open webcam. Check camera permissions.")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        self._cap.set(cv2.CAP_PROP_FPS, FPS)

        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        self._path = Path(tmp.name)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, FPS, (WIDTH, HEIGHT))

        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self._path

    def _loop(self) -> None:
        assert self._cap is not None and self._writer is not None
        auto_stopped = False
        while not self._stop.is_set():
            if time.time() - self._started_at > MAX_SECONDS:
                self._stop.set()
                auto_stopped = True
                break
            ok, frame = self._cap.read()
            if not ok:
                continue
            self._writer.write(frame)
        if auto_stopped and self._on_complete and not self._fired_complete:
            self._fired_complete = True
            self._on_complete()

    def stop(self) -> CaptureMeta:
        self._stop.set()
        self._stopped_at = time.time()
        if self._thread:
            self._thread.join(timeout=2)
        if self._writer:
            self._writer.release()
        if self._cap:
            self._cap.release()
        assert self._path is not None
        return CaptureMeta(
            path=self._path,
            duration_seconds=round(self._stopped_at - self._started_at, 3),
            fps=FPS,
            resolution=f"{WIDTH}x{HEIGHT}",
            audio_present=False,
        )


class _FFmpegRecorder:
    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._path: Path | None = None
        self._started_at = 0.0
        self._stopped_at = 0.0
        self._on_complete: Callable[[], None] | None = None
        self._fired_complete = False
        self._watcher: threading.Thread | None = None
        self._video_dev = os.environ.get("LIPSYNC_FFMPEG_VIDEO_DEV", "0")
        self._audio_dev = os.environ.get("LIPSYNC_FFMPEG_AUDIO_DEV", "0")

    def start(self, on_complete: Callable[[], None] | None = None) -> Path:
        self._on_complete = on_complete
        self._fired_complete = False
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        self._path = Path(tmp.name)

        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-f", "avfoundation",
            "-framerate", str(FPS),
            "-video_size", f"{WIDTH}x{HEIGHT}",
            "-i", f"{self._video_dev}:{self._audio_dev}",
            "-t", str(MAX_SECONDS),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", str(AUDIO_SAMPLE_RATE), "-ac", "1",
            "-y",
            str(self._path),
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        self._started_at = time.time()
        self._watcher = threading.Thread(target=self._watch, daemon=True)
        self._watcher.start()
        return self._path

    def _watch(self) -> None:
        assert self._proc is not None
        self._proc.wait()
        if self._on_complete and not self._fired_complete and time.time() - self._started_at >= MAX_SECONDS * 0.95:
            self._fired_complete = True
            self._on_complete()

    def stop(self) -> CaptureMeta:
        assert self._proc is not None and self._path is not None
        self._stopped_at = time.time()
        if self._proc.poll() is None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.write(b"q")
                    self._proc.stdin.flush()
                    self._proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.terminate()
                self._proc.wait(timeout=2)
        # SIGINT exit (255) and graceful exit (0) are both fine.
        if self._proc.returncode not in (0, 255, -2):
            stderr = b""
            if self._proc.stderr:
                stderr = self._proc.stderr.read()
            raise RuntimeError(
                f"ffmpeg exited {self._proc.returncode}: "
                f"{stderr.decode(errors='replace')[:500]}"
            )
        return CaptureMeta(
            path=self._path,
            duration_seconds=round(self._stopped_at - self._started_at, 3),
            fps=FPS,
            resolution=f"{WIDTH}x{HEIGHT}",
            audio_present=True,
        )
