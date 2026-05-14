"""Visual / audio-visual speech recognition backend interface.

Selection is via config.AVSR_BACKEND. All backends implement the same
Protocol — switch backends without changing call sites.

Usage:
    from avsr import transcribe
    result = transcribe(Path("clip.mp4"))
    print(result.text)
"""
from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import config

Mode = Literal["v", "a", "av"]


@dataclass
class AVSRResult:
    text: str
    mode: Mode
    model_id: str
    confidence: float | None = None


class NotInstalled(RuntimeError):
    """Raised when the configured backend is selected but its dependencies are missing."""


class Backend(Protocol):
    name: str
    def load(self) -> None: ...
    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult: ...


# ---- Mock backend -----------------------------------------------------------

class MockBackend:
    """Returns a placeholder string. For testing the rest of the app without
    needing a multi-GB checkpoint installed."""

    name = "mock"

    def load(self) -> None:
        pass

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        return AVSRResult(
            text=f"[mock backend — set LIPSYNC_BACKEND=usr2 and install the checkpoint for real transcription. video={video_path.name}]",
            mode=mode,
            model_id="mock-v0",
        )


# ---- USR 2.0 backend (subprocess against their demo.py) --------------------

class USR2Backend:
    """USR 2.0 (Haliassos et al., ICLR 2026).

    Setup (see scripts/setup_usr2.sh):
        git clone https://github.com/ahaliassos/usr2 third_party/usr2
        pip install -r third_party/usr2/requirements.txt
        # Download a fine-tuned checkpoint (Base+ or Huge) to
        # checkpoints/usr2_base_plus.pth (or set LIPSYNC_CHECKPOINT)

    We invoke their demo.py via subprocess — more stable across their Hydra
    config layout than importing internals directly. demo.py handles face
    detection and mouth cropping itself; we just pass the raw video.

    Output contract (from their README):
        ============================================================
         Modality : av
         Video    : /path/to/video.mp4
         Result   : THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG
        ============================================================
    """

    name = "usr2"
    # Strict: matches the labeled "Result   : ..." banner line.
    _RESULT_RE = re.compile(r"^\s*Result\s*:\s*(.+?)\s*$", re.MULTILINE)

    def load(self) -> None:
        repo = config.USR2_REPO_DIR
        if not repo.exists():
            raise NotInstalled(
                f"USR2 repo not found at {repo}.\n"
                f"Fix: bash scripts/setup_usr2.sh"
            )
        if not (repo / "demo.py").exists():
            raise NotInstalled(
                f"USR2 demo.py missing at {repo}/demo.py — the clone may be incomplete.\n"
                f"Fix: rm -rf {repo} && bash scripts/setup_usr2.sh"
            )
        ckpt = config.AVSR_CHECKPOINT_PATH
        if ckpt is None:
            raise NotInstalled(
                "No checkpoint configured. Set LIPSYNC_CHECKPOINT=/path/to/your.pth "
                "or save a fine-tuned checkpoint at checkpoints/usr2_base_plus.pth.\n"
                "Get one from https://github.com/ahaliassos/usr2 → Pretrained Models "
                "→ Fine-tuned (full model) → High-resource → Base+ LRS3+Vox2."
            )
        if not ckpt.exists():
            raise NotInstalled(
                f"Checkpoint not found at {ckpt}.\n"
                "Download from https://github.com/ahaliassos/usr2 → Pretrained Models "
                "→ Fine-tuned (full model) → High-resource → Base+ LRS3+Vox2 (recommended "
                "first run; Huge LRS2+LRS3+Vox2+AVS for max accuracy). Save / rename the .pth to:\n"
                f"  {ckpt}\n"
                "The README's checkpoint table is the source of truth — any hardcoded "
                "URL in this project may have rotated."
            )
        # Smoke-check the active venv has torch — saves a confusing demo.py traceback.
        try:
            import torch  # noqa: F401
        except ImportError:
            raise NotInstalled(
                "PyTorch not installed in the current environment. "
                "Run: bash scripts/setup_usr2.sh"
            )

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        self.load()
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        repo = config.USR2_REPO_DIR
        ckpt = config.AVSR_CHECKPOINT_PATH
        assert ckpt is not None  # load() guarantees this
        cmd = [
            sys.executable, "demo.py",
            f"video={video_path.resolve()}",
            f"model.pretrained_model_path={ckpt.resolve()}",
            f"modality={mode}",
            "detector=mediapipe",  # CUDA-free; RetinaFace requires GPU + ibug.
        ]
        proc = subprocess.run(
            cmd,
            cwd=repo,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"USR2 demo.py exited {proc.returncode}.\n"
                f"Command: cd {repo} && {' '.join(cmd)}\n"
                f"STDERR:\n{proc.stderr[-1500:]}\n"
                f"STDOUT (tail):\n{proc.stdout[-500:]}"
            )
        text = self._extract_transcript(proc.stdout)
        if not text:
            raise RuntimeError(
                "USR2 demo.py exited 0 but no 'Result :' line was found in stdout. "
                "Their output format may have changed.\n"
                f"STDOUT:\n{proc.stdout[-1500:]}"
            )
        return AVSRResult(
            text=text,
            mode=mode,
            model_id=f"usr2/{ckpt.stem}",
        )

    def _extract_transcript(self, stdout: str) -> str:
        """Pull the last 'Result   : <text>' line from the banner output."""
        matches = self._RESULT_RE.findall(stdout)
        if matches:
            return matches[-1].strip()
        return ""


# ---- Auto-AVSR skeleton -----------------------------------------------------

class AutoAVSRBackend:
    """Auto-AVSR (Ma et al., ICASSP 2023). Skeleton — not yet wired up."""

    name = "auto_avsr"

    def load(self) -> None:
        raise NotInstalled(
            "auto_avsr backend not implemented. See "
            "https://github.com/mpc001/auto_avsr — clone + checkpoint then wire "
            "their lightning.ModelModule into this class."
        )

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        self.load()
        raise NotInstalled("auto_avsr backend not implemented.")


# ---- AV-HuBERT skeleton -----------------------------------------------------

class AVHuBERTBackend:
    """AV-HuBERT (Shi et al., ICLR 2022). Skeleton — not yet wired up."""

    name = "avhubert"

    def load(self) -> None:
        raise NotInstalled(
            "avhubert backend not implemented. See "
            "https://github.com/facebookresearch/av_hubert — requires fairseq."
        )

    def transcribe(self, video_path: Path, mode: Mode = "av") -> AVSRResult:
        self.load()
        raise NotInstalled("avhubert backend not implemented.")


# ---- Factory ----------------------------------------------------------------

_BACKENDS: dict[str, type[Backend]] = {
    "mock": MockBackend,
    "usr2": USR2Backend,
    "auto_avsr": AutoAVSRBackend,
    "avhubert": AVHuBERTBackend,
}

_instance: Backend | None = None


def get_backend() -> Backend:
    global _instance
    if _instance is None:
        name = config.AVSR_BACKEND
        if name not in _BACKENDS:
            raise ValueError(
                f"Unknown AVSR_BACKEND {name!r}. Pick one of {sorted(_BACKENDS)}."
            )
        _instance = _BACKENDS[name]()
    return _instance


def transcribe(video_path: Path, mode: Mode | None = None) -> AVSRResult:
    backend = get_backend()
    return backend.transcribe(video_path, mode or config.AVSR_MODALITY)  # type: ignore[arg-type]
