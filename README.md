# lipsync-text

A personal **visual / audio-visual speech recognition (VSR / AVSR)** desktop app. Records a short clip from the webcam, runs a real pretrained lip-reading model on it, and gives you the text on your clipboard. Future-direction goal: fine-tune the same model on your own collected data.

Inference uses **USR 2.0** (Haliassos et al., ICLR 2026) by default. Auto-AVSR and AV-HuBERT skeletons are present in `src/avsr.py` if you want to swap. Multimodal LLMs (Gemini, Grok, MiniMax) are **not** the recognition model — they only show up as optional post-process rescorers.

## Quick start

```bash
# 1. Project setup
cd ~/lipsync-text
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
brew install ffmpeg                          # webcam capture + USR2 video decode

# 2. Install USR 2.0 backend (clones repo + installs PyTorch + deps)
bash scripts/setup_usr2.sh

# 3. Download a checkpoint manually from Google Drive
#    (Google Drive's UI gating makes wget unreliable, so this is a manual step.)
#    Base+ ~recommended for first run; Huge for max accuracy:
#      https://drive.google.com/file/d/18vmJjdem5XPOA8bmizybIW5sLJuHMdRR/view   (Base+)
#      https://drive.google.com/file/d/1LzFOTYu45zCLOHGVLQt7pMGjw6jmmo9Y/view   (Huge)
#    Save to checkpoints/usr2_base_plus.pth, or set LIPSYNC_CHECKPOINT to point elsewhere.

# 4. Real inference on one video
LIPSYNC_BACKEND=usr2 \
  LIPSYNC_CHECKPOINT=checkpoints/usr2_base_plus.pth \
  python scripts/run_inference.py path/to/sample.mp4

# 5. Live hotkey app
python src/main.py
# Press ⌘⇧L → mouth a phrase → press again (or wait 8s).
# Transcript lands on your clipboard + macOS notification.
```

**Smoke-test the wire-up without a checkpoint (mock backend):**

```bash
LIPSYNC_BACKEND=mock python scripts/run_inference.py any.mp4
```

Mock returns a recognizable placeholder string so you can confirm the rest of the app works before pulling down 1–4 GB of model weights.

## Modes

Mode is controlled by env vars (`src/config.py` is the source of truth).

| Mode | Set | Behavior |
|---|---|---|
| **Transcribe-only (default)** | nothing | Record → AVSR → display → discard clip. Nothing saved. |
| **Collect for fine-tuning** | `LIPSYNC_COLLECT=1` | Same flow, but after transcription a review dialog opens. Saving writes the clip and your corrected transcript to `data/manifest.jsonl`. See `TRAINING_PLAN.md`. |
| **LLM rescore** | `LIPSYNC_RESCORE=1` | AVSR text is post-processed by `rescore.py:polish()` (Gemini, punctuation cleanup only — never adds words). |

Modality:

| Set | Behavior |
|---|---|
| `LIPSYNC_MODALITY=av` (default) | audio + visual (best accuracy when audio is captured) |
| `LIPSYNC_MODALITY=v` | visual only — for silent mouthing |
| `LIPSYNC_MODALITY=a` | audio only |

Backend:

```bash
LIPSYNC_BACKEND=usr2         # default
LIPSYNC_BACKEND=mock         # placeholder, for testing without weights
LIPSYNC_BACKEND=auto_avsr    # skeleton (not yet wired up)
LIPSYNC_BACKEND=avhubert     # skeleton (not yet wired up)
```

## Architecture

```
                 ┌──────────────────┐
                 │  recorder.py     │  ffmpeg AVFoundation, mp4 with audio
                 │  ⌘⇧L hotkey      │  (or scripts/run_inference.py video.mp4)
                 └────────┬─────────┘
                          ▼
                 ┌──────────────────┐    src/config.py picks backend
                 │  avsr.py         │ ─▶ MockBackend / USR2Backend
                 │  Backend         │     / AutoAVSRBackend / AVHuBERTBackend
                 │  interface       │
                 └────────┬─────────┘
                          ▼
            ┌───────── result.text ─────────┐
            ▼                                ▼
   ┌──────────────────┐                ┌──────────────────┐
   │  rescore.py      │ (optional)     │  output.py        │
   │  LLM polish      │ ─────────────▶ │  clipboard + notif│
   └──────────────────┘                └──────────────────┘
                                              │
                                              ▼ (only if LIPSYNC_COLLECT=1)
                                       ┌──────────────────┐
                                       │  review.py +     │
                                       │  dataset.py      │
                                       │  manifest.jsonl  │
                                       └──────────────────┘
```

## macOS permissions

- **Camera, Microphone** — auto-prompt on first capture.
- **Accessibility** — required by `pynput` for the ⌘⇧L global hotkey. System Settings → Privacy & Security → Accessibility → add and restart your terminal/IDE.

To pick a specific audio device (e.g. AirPods):

```bash
ffmpeg -f avfoundation -list_devices true -i ""
export LIPSYNC_FFMPEG_AUDIO_DEV=<index>
```

## Switching backends

To swap USR2 for Auto-AVSR or AV-HuBERT, those backends are scaffolded but not wired up — implement `load()` and `transcribe()` in the relevant class in `src/avsr.py`, then `export LIPSYNC_BACKEND=auto_avsr` (or `avhubert`).

## Optional: fine-tuning on your own data

Data collection is **opt-in** now (`LIPSYNC_COLLECT=1`), not the primary workflow. When enabled, sessions flow through the manifest schema described below and you eventually fine-tune on top of USR2's checkpoint. Detailed plan in [TRAINING_PLAN.md](./TRAINING_PLAN.md).

Manifest layout (gitignored):

```
data/
├── clips/<session_id>.mp4         raw 640×480 video (+ audio)
├── rois/<session_id>.mp4          96×96 grayscale mouth crop
├── landmarks/<session_id>.npz     478-point face mesh per frame
├── audio/<session_id>.wav         16 kHz mono PCM
└── manifest.jsonl                 append-only, latest row per session_id wins
```

Pipeline commands when collecting:

```bash
LIPSYNC_COLLECT=1 python src/main.py   # record + review
python src/preprocess.py               # mouth ROI + landmarks + audio extract
python src/split.py --by session       # train/val/test (speaker-disjoint when multi-speaker)
python scripts/verify_pipeline.py      # integrity gate before any training
```

Schema is in `src/dataset.py:Entry`; full field list and validation rules described inline there.

## Role of Gemini / Grok / MiniMax

These are **support models**, not the recognition system. Only used for:
- Punctuation polish / capitalization (`rescore.py:polish`).
- N-best selection from beam-search output (`rescore.py:rescore_nbest`) — constrained to picking one of the existing hypotheses, never free-text generation.
- Code generation / project automation during development.

If you fine-tune any LLM, it's only as a text post-processor on top of AVSR output. Never as the core recognition model.

## Honest limitations

- USR2 inference is invoked via subprocess against their `demo.py`. Adds 1–3 s of Python startup overhead per call. If we hit a latency wall, switch to in-process import.
- Mouth crop in `src/preprocess.py` doesn't do face alignment / rotation normalization — center crop around the lip centroid. Good enough as a baseline.
- Single-speaker assumption everywhere — `speaker_id` field exists for future multi-speaker support but defaults to `"default"`.
- Hold-to-record / streaming inference / menu-bar app — deferred.
