"""Runtime configuration.

All settings can be overridden by env vars (the LIPSYNC_* equivalents).
This module is the single source of truth — read from it everywhere.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---- Recognition backend ----------------------------------------------------
# "usr2"       — USR 2.0 (ICLR 2026), recommended. Requires third_party/usr2 clone + checkpoint.
# "auto_avsr"  — Auto-AVSR (ICASSP 2023). Skeleton; not wired up yet.
# "avhubert"   — AV-HuBERT (ICLR 2022). Skeleton; not wired up yet.
# "mock"       — Returns a placeholder string. For testing wire-up without weights.
AVSR_BACKEND: str = os.environ.get("LIPSYNC_BACKEND", "usr2")

# Path to the model checkpoint file (.pth) for the active backend.
AVSR_CHECKPOINT_PATH: Path | None = (
    Path(os.environ["LIPSYNC_CHECKPOINT"]).expanduser()
    if "LIPSYNC_CHECKPOINT" in os.environ
    else ROOT / "checkpoints" / "usr2_base_plus.pth"
)

# Modality: "v" (visual only), "a" (audio only), or "av" (audio-visual, default).
AVSR_MODALITY: str = os.environ.get("LIPSYNC_MODALITY", "av")

# USR2 backbone config name — MUST match the checkpoint's architecture.
# Options (from third_party/usr2/conf/model/backbone/):
#   resnet_transformer_base       (adim=512, ~small)
#   resnet_transformer_baseplus   (adim=768, default — matches the Base+ checkpoints)
#   resnet_transformer_large      (adim=1024)
#   resnet_transformer_huge       (adim=1280, USR2's own demo.py default — for the Huge checkpoint)
# Mismatching backbone vs checkpoint produces shape errors at load time
# (e.g. "shape torch.Size([768]) from checkpoint vs torch.Size([1280]) in current model").
USR2_BACKBONE: str = os.environ.get("LIPSYNC_USR2_BACKBONE", "resnet_transformer_baseplus")

# Path to the cloned backend repo (for subprocess-based backends like USR2).
USR2_REPO_DIR: Path = ROOT / "third_party" / "usr2"

# ---- App behavior -----------------------------------------------------------
# When True, every recorded clip is saved to data/manifest.jsonl with the
# review dialog. When False (default), the clip is transcribed and then deleted.
COLLECT_MODE: bool = os.environ.get("LIPSYNC_COLLECT", "0") == "1"

# When True, AVSR output is post-processed by rescore.py:polish() (LLM cleanup).
# Off by default — the recognition model's output is the primary product.
AVSR_RESCORE: bool = os.environ.get("LIPSYNC_RESCORE", "0") == "1"

# ---- Speaker identity (for future multi-speaker manifest entries) ----------
SPEAKER_ID: str = os.environ.get("LIPSYNC_SPEAKER_ID", "default")
