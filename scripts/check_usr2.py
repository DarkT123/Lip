"""Pre-flight check for the USR2 backend.

Verifies the four conditions that must hold before real inference can run:
  1. third_party/usr2/demo.py exists
  2. A checkpoint file exists and is non-empty
  3. Required Python packages are importable
  4. ffmpeg is on PATH

Exits 0 if all pass, 1 otherwise. Prints a per-check pass/fail summary with
actionable next steps.

Usage:
    python scripts/check_usr2.py
    LIPSYNC_CHECKPOINT=/path/to/x.pth python scripts/check_usr2.py
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import config  # noqa: E402


GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def ok(msg: str) -> None:
    print(f"{GREEN}[ ok ]{RESET} {msg}")


def fail(msg: str, fix: str) -> int:
    print(f"{RED}[FAIL]{RESET} {msg}")
    print(f"       fix: {fix}")
    return 1


def warn(msg: str) -> None:
    print(f"{YELLOW}[warn]{RESET} {msg}")


def check_usr2_repo() -> int:
    repo = config.USR2_REPO_DIR
    if not repo.exists():
        return fail(
            f"USR2 repo not found at {repo}",
            "bash scripts/setup_usr2.sh",
        )
    demo = repo / "demo.py"
    if not demo.exists():
        return fail(
            f"demo.py missing inside {repo} — clone is incomplete",
            f"rm -rf {repo} && bash scripts/setup_usr2.sh",
        )
    ok(f"USR2 repo present at {repo} (demo.py found)")
    return 0


def check_checkpoint() -> int:
    ckpt = config.AVSR_CHECKPOINT_PATH
    if ckpt is None:
        return fail(
            "No checkpoint path configured",
            "set LIPSYNC_CHECKPOINT=/path/to/x.pth, or save one at "
            "checkpoints/usr2_base_plus.pth",
        )
    if not ckpt.exists():
        return fail(
            f"Checkpoint not found at {ckpt}",
            "open https://github.com/ahaliassos/usr2 → "
            "Pretrained Models → Fine-tuned (full model) → High-resource. "
            "Download the 'Base+ LRS3+Vox2' checkpoint (recommended for first run), "
            f"then save / rename the .pth to:\n              {ckpt}\n"
            "            The README's checkpoint table is the source of truth — "
            "direct URLs may rotate.",
        )
    size = ckpt.stat().st_size
    if size == 0:
        return fail(
            f"Checkpoint at {ckpt} is empty (0 bytes)",
            f"redownload it. Expected size is ~hundreds of MB to a few GB.",
        )
    if size < 50 * 1024 * 1024:
        warn(
            f"Checkpoint at {ckpt} is only {size // (1024 * 1024)} MB — "
            f"smaller than expected for a USR2 fine-tuned model. "
            f"Verify the download wasn't truncated (Google Drive often returns "
            f"an HTML 'confirm download' page when the file is too large)."
        )
    ok(f"checkpoint present at {ckpt} ({size // (1024 * 1024)} MB)")
    return 0


def check_python_packages() -> int:
    required = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("torchvision", "TorchVision"),
        ("pytorch_lightning", "PyTorch Lightning"),
        ("hydra", "Hydra"),
        ("omegaconf", "OmegaConf"),
        ("mediapipe", "MediaPipe"),
        ("cv2", "OpenCV"),
        ("av", "PyAV"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
    ]
    failed = 0
    versions = []
    for mod, label in required:
        try:
            m = importlib.import_module(mod)
            versions.append((label, mod, getattr(m, "__version__", "?")))
        except ImportError as e:
            failed += fail(
                f"Python package '{label}' ({mod}) not importable: {e}",
                "bash scripts/setup_usr2.sh",
            )
    if not failed:
        labels = ", ".join(f"{l}={v}" for l, _, v in versions)
        ok(f"all required packages importable: {labels}")
    return failed


def check_ffmpeg() -> int:
    if shutil.which("ffmpeg") is None:
        return fail(
            "ffmpeg not found on PATH",
            "brew install ffmpeg   (macOS) / sudo apt-get install ffmpeg   (Linux)",
        )
    ok(f"ffmpeg present ({shutil.which('ffmpeg')})")
    return 0


def main() -> int:
    print(f"Project root: {ROOT}")
    print(f"Python:       {sys.executable}")
    print(f"Backend:      {config.AVSR_BACKEND}")
    print(f"Checkpoint:   {config.AVSR_CHECKPOINT_PATH}")
    print()

    fails = 0
    fails += check_usr2_repo()
    fails += check_python_packages()
    fails += check_ffmpeg()
    fails += check_checkpoint()
    print()
    if fails:
        print(f"{RED}{fails} check(s) failed.{RESET} Resolve the items above, then re-run.")
        return 1
    print(f"{GREEN}All checks passed.{RESET} Ready for real USR2 inference.")
    print("Run:")
    print(f'  LIPSYNC_BACKEND=usr2 LIPSYNC_CHECKPOINT={config.AVSR_CHECKPOINT_PATH} \\')
    print(f"      python scripts/run_inference.py path/to/sample.mp4")
    return 0


if __name__ == "__main__":
    sys.exit(main())
