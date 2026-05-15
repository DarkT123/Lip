"""Runtime shim around USR2's demo.py.

Why this exists
---------------
USR2's demo.py decodes video frames via torchvision.io.read_video(), which
internally uses PyAV. On macOS arm64 with PyAV 17.0.1, libswscale's scaling
graph fails to initialize for yuv420p+bt709 → rgb24 conversion (EAGAIN /
"Resource temporarily unavailable"). PyAV 14/15/16 don't ship arm64 wheels
for Python 3.13, so a downgrade isn't an option.

This shim monkey-patches `torchvision.io.read_video` to use an ffmpeg
subprocess decoder instead, then hands off to USR2's demo.py via runpy so
__file__, __name__, sys.argv, and the working directory are all preserved
for Hydra. No third-party code is modified.

Invocation
----------
The exact arguments USR2's demo.py would accept, just routed through us:

    cd third_party/usr2
    python /abs/path/to/scripts/usr2_shim.py \
        video=/abs/path/to/clip.mp4 \
        model.pretrained_model_path=/abs/path/to/ckpt.pth \
        modality=av detector=mediapipe

USR2Backend.transcribe() handles the invocation; see src/avsr.py.
"""
from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torchvision.io

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USR2_DIR = PROJECT_ROOT / "third_party" / "usr2"
DEMO_PATH = USR2_DIR / "demo.py"


def _ffprobe_streams(path: str) -> tuple[dict | None, dict | None]:
    """Return (video_stream, audio_stream) dicts from ffprobe."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_streams", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    info = json.loads(out)
    v = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    return v, a


def _read_video_ffmpeg(
    filename: str,
    start_pts: float = 0,
    end_pts: float | None = None,
    pts_unit: str = "pts",
    output_format: str = "THWC",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Drop-in replacement for torchvision.io.read_video using ffmpeg subprocess.

    Returns (video[T,H,W,C] uint8 RGB, audio[C,S] float32, info_dict).
    Matches torchvision.io.read_video's public contract.
    """
    v, a = _ffprobe_streams(str(filename))
    if v is None:
        raise RuntimeError(f"No video stream found in {filename}")

    width = int(v["width"])
    height = int(v["height"])
    num, den = (int(x) for x in v.get("r_frame_rate", "30/1").split("/"))
    fps = num / den if den else 30.0

    # --- video: decode to raw RGB24 frames via ffmpeg subprocess ---
    # Pass through start/end via -ss / -to when given.
    cmd_v = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if start_pts and pts_unit == "sec":
        cmd_v += ["-ss", str(float(start_pts))]
    cmd_v += ["-i", str(filename)]
    if end_pts is not None and pts_unit == "sec":
        cmd_v += ["-to", str(float(end_pts))]
    cmd_v += ["-f", "rawvideo", "-pix_fmt", "rgb24", "-vsync", "passthrough", "-"]

    proc_v = subprocess.run(cmd_v, check=True, capture_output=True)
    frame_size = width * height * 3
    n_frames = len(proc_v.stdout) // frame_size
    video = torch.frombuffer(
        bytearray(proc_v.stdout[: n_frames * frame_size]),
        dtype=torch.uint8,
    ).reshape(n_frames, height, width, 3)

    if output_format == "TCHW":
        video = video.permute(0, 3, 1, 2).contiguous()

    # --- audio: decode to float32 PCM via ffmpeg subprocess ---
    audio = torch.zeros(1, 0, dtype=torch.float32)
    audio_fps = 0
    if a is not None:
        audio_fps = int(a.get("sample_rate", 16000))
        n_channels = int(a.get("channels", 1))
        cmd_a = ["ffmpeg", "-hide_banner", "-loglevel", "error",
                 "-i", str(filename),
                 "-f", "f32le", "-acodec", "pcm_f32le",
                 "-ac", str(n_channels), "-ar", str(audio_fps),
                 "-"]
        proc_a = subprocess.run(cmd_a, check=True, capture_output=True)
        flat = torch.frombuffer(bytearray(proc_a.stdout), dtype=torch.float32)
        # Reshape to (C, S). ffmpeg writes interleaved (S, C) for >1 channel.
        if n_channels > 1:
            audio = flat.reshape(-1, n_channels).T.contiguous()
        else:
            audio = flat.unsqueeze(0)

    return video, audio, {"video_fps": fps, "audio_fps": audio_fps}


# Install the patch before USR2's demo.py runs.
torchvision.io.read_video = _read_video_ffmpeg

# Make demo.py's sibling packages (data/, preprocessing/, espnet/, utils/, models/)
# importable — runpy.run_path does not prepend the script's dir to sys.path the
# way `python demo.py` does, so `from data.transforms import …` would fail.
sys.path.insert(0, str(USR2_DIR))

# Hand off to demo.py. runpy gives us correct __file__, __name__,
# and sys.argv handling so Hydra's @hydra.main behaves identically
# to a direct `python demo.py ...` invocation.
os.chdir(USR2_DIR)
sys.argv = [str(DEMO_PATH)] + sys.argv[1:]
runpy.run_path(str(DEMO_PATH), run_name="__main__")
