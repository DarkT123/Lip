#!/usr/bin/env bash
# Set up the USR 2.0 (ICLR 2026) backend for lipsync-text.
#
# What this does:
#   1. ffmpeg sanity check (USR2 needs it for video/audio decode)
#   2. Clones https://github.com/ahaliassos/usr2 to third_party/usr2
#   3. Installs PyTorch appropriate for the host (macOS = default MPS-capable wheels,
#      Linux = CUDA 12.1 wheels per USR2's README)
#   4. Installs USR2's requirements.txt
#   5. Creates the checkpoints/ directory and prints the next-step download URL
#
# What this does NOT do:
#   - Download the multi-GB checkpoint. Google Drive doesn't expose a stable
#     `wget`-able URL for those files; do it manually in your browser.
#
# Re-run safe: clone is skipped if already present; pip installs are idempotent.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT/third_party"
USR2_DIR="$THIRD_PARTY/usr2"
CKPT_DIR="$ROOT/checkpoints"
CKPT_PATH="$CKPT_DIR/usr2_base_plus.pth"

bold()  { printf '\033[1m%s\033[0m\n' "$*"; }
warn()  { printf '\033[33m[warn]\033[0m %s\n' "$*"; }
error() { printf '\033[31m[err]\033[0m %s\n' "$*" >&2; }

bold "[1/5] Checking ffmpeg"
if ! command -v ffmpeg >/dev/null 2>&1; then
  error "ffmpeg not found. Install it first:"
  echo "  macOS: brew install ffmpeg"
  echo "  Linux: sudo apt-get install ffmpeg"
  exit 1
fi
echo "  ok: $(ffmpeg -version | head -1)"

bold "[2/5] Cloning USR2 to $USR2_DIR"
mkdir -p "$THIRD_PARTY"
if [ -d "$USR2_DIR/.git" ]; then
  echo "  already cloned — fetching latest"
  git -C "$USR2_DIR" fetch --quiet origin
  git -C "$USR2_DIR" reset --quiet --hard origin/HEAD
else
  git clone --depth 1 https://github.com/ahaliassos/usr2 "$USR2_DIR"
fi
if [ ! -f "$USR2_DIR/demo.py" ]; then
  error "demo.py is missing from the clone at $USR2_DIR. Aborting."
  exit 1
fi

bold "[3/5] Installing PyTorch"
PLATFORM="$(uname -s)-$(uname -m)"
case "$PLATFORM" in
  Darwin-arm64|Darwin-x86_64)
    echo "  macOS detected — installing default PyTorch wheels (MPS-capable on arm64)"
    python3 -m pip install --upgrade torch torchaudio torchvision
    ;;
  Linux-x86_64)
    echo "  Linux x86_64 detected — installing CUDA 12.1 wheels per USR2 README"
    python3 -m pip install --upgrade torch torchaudio torchvision \
      --index-url https://download.pytorch.org/whl/cu121
    ;;
  *)
    warn "Unknown platform $PLATFORM — falling back to default PyTorch wheels"
    python3 -m pip install --upgrade torch torchaudio torchvision
    ;;
esac

bold "[4/5] Installing USR2 requirements"
python3 -m pip install -r "$USR2_DIR/requirements.txt"

bold "[5/5] Preparing checkpoints directory"
mkdir -p "$CKPT_DIR"
if [ -f "$CKPT_PATH" ]; then
  echo "  checkpoint already present: $CKPT_PATH"
else
  cat <<EOF

Next step — download a USR2 fine-tuned checkpoint manually.

  Source of truth: the checkpoint table in the USR2 GitHub README.
  Direct download URLs rotate, so do NOT trust any hardcoded link below
  if it 404s — go to the README and grab the current link.

  1. Open https://github.com/ahaliassos/usr2
  2. Scroll to:  Pretrained Models → Fine-tuned (full model) → High-resource
  3. Download:   Base+ LRS3+Vox2     (recommended for first run, ~hundreds of MB)
       or:      Huge LRS2+LRS3+Vox2+AVS   (best accuracy, multi-GB)
  4. Save / rename the downloaded .pth to:
        $CKPT_PATH
     (or set LIPSYNC_CHECKPOINT=/path/to/your.pth before running inference)

Then verify and run:
  python scripts/check_usr2.py
  python scripts/run_inference.py path/to/sample.mp4

EOF
fi

bold "Setup complete."
