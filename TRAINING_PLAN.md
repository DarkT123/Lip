# Training plan

How we go from "inference works on a pretrained model" to "personalized model fine-tuned on my own data." Inference-first is now the priority; training is the **future** direction.

## Where we are

```
Stage 0  Inference-first MVP .............. CURRENT GOAL
         pretrained USR2 → clip → text → clipboard.
         Optional data collection enabled by LIPSYNC_COLLECT=1.

Stage 1  Collected dataset ready .......... GATED on Stage 0 working
         hundreds of label_quality="gold" rows accumulated.
         scripts/verify_pipeline.py exits 0.

Stage 2  Public-corpus fine-tune .......... LRS3 if accessible, otherwise skip
         light adaptation of USR2 on LRS3 train split.

Stage 3  Personal fine-tune ............... fine-tune on your manifest.
         low LR, mixed batches with LRS3 to avoid catastrophic forgetting.

Stage 4  Evaluation ........................ WER/CER vs. baseline pretrained USR2.
         VSR vs. AVSR reported separately.

Stage 5  Optional rescoring ................ if WER doesn't move, try rescore.py.
```

Each stage gates the next.

## Stage 0 — Inference-first MVP (current)

**Acceptance gate:** `python scripts/run_inference.py <some video>` returns plausible text. The hotkey app produces a transcript end-to-end.

What we have:
- `src/avsr.py` — backend interface with `MockBackend`, `USR2Backend`, `AutoAVSRBackend` (skeleton), `AVHuBERTBackend` (skeleton).
- `scripts/run_inference.py` — one-shot inference on a video file.
- `src/main.py` — hotkey-driven live mode, transcribe-only by default.

What you need to do:
1. `git clone https://github.com/ahaliassos/usr2 third_party/usr2`
2. `pip install -r third_party/usr2/requirements.txt`
3. Download a USR2 fine-tuned checkpoint (Base+ or Huge) from the README's Google Drive links.
4. Set `LIPSYNC_CHECKPOINT=/path/to/checkpoint.pth` or place it at `checkpoints/usr2_base_plus.pth`.
5. Sanity check: `python scripts/run_inference.py path/to/test.mp4`. Should print a transcript.

If their `demo.py` output format ever drifts, edit `USR2Backend._TRANSCRIPT_RE` in `src/avsr.py`.

## Stage 1 — Collected dataset ready

This is the work the previous version of this doc treated as Stage 0. Now it's optional and only happens if you want personalization.

Switch to collect mode and accumulate clips:

```bash
LIPSYNC_COLLECT=1 python src/main.py
```

Each session: record → AVSR transcribes → notification + clipboard → review dialog (Save / Skip / Delete). Saved entries land in `data/manifest.jsonl` with `transcript_human` as your corrected ground truth.

When you have at least ~100 reviewed gold rows:

```bash
python src/preprocess.py             # MediaPipe lip ROI + landmarks + audio extract
python src/split.py --by session     # train/val/test (use --by speaker once multi-speaker)
python scripts/verify_pipeline.py    # gate — must exit 0
```

**Acceptance gate:** `verify_pipeline.py` exits 0 and you have a meaningful number of `label_quality=="gold"` rows in train/val/test.

## Stage 2 — Public-corpus fine-tune (LRS3)

Optional. Recommended only if you can get LRS3 (apply at https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs3.html — approval takes days).

USR2 is already pretrained on LRS3, so this is "light re-adaptation," not a primary training run. Use it if you notice the public checkpoint is weak on your camera conditions.

Hyperparameters (rule of thumb):
- LR: 1e-5 to 5e-5
- Epochs: 3–5
- Track val WER per epoch, early-stop on plateau

USR2's training scripts in `third_party/usr2/` are the reference; their Hydra configs handle most of this.

**Acceptance gate:** WER on LRS3 val ≤ public-checkpoint WER (i.e., you didn't regress).

## Stage 3 — Personal fine-tune

The actual goal. Adapt the (USR2 or USR2-after-LRS3-fine-tune) checkpoint to your face, voice, and vocabulary.

Read the gold rows assigned to `split=="train"`:

```python
import json
gold_train = [
    r for line in open("data/manifest.jsonl")
    if (r := json.loads(line))
    and r.get("label_quality") == "gold"
    and not r.get("deleted")
    and r.get("split") == "train"
]
```

Pass each row's `mouth_roi_path` + `audio_path` + `transcript_human` to USR2's training loop. Things to watch:

- **LR ≈ 1e-5.** An order of magnitude below LRS3 fine-tuning. Smaller dataset = easier to overfit.
- **Epochs ≤ 10.** Watch val WER every epoch on your `split=="val"` rows. The moment it goes up, stop.
- **Mixed batches.** Alternate LRS3 batches and personal batches ~4:1 if you have LRS3 access — prevents catastrophic forgetting of general English.
- **Consider LoRA / adapter-only training** if you only have a few hundred clips. Full fine-tune of a 100M+ param model on <1000 examples is overkill and unstable.

**Acceptance gate:** Val WER on your `split=="val"` rows lower than the public USR2 checkpoint on the same val set, AND no regression on a held-out LRS3 sample if you have it.

Crucial: never train on `split=="val"` or `split=="test"` rows. `scripts/verify_pipeline.py` enforces speaker- and session-disjointness so the splits are trustworthy.

## Stage 4 — Evaluation

Numbers we report (and re-report after every change):

- **VSR mode** (lip ROI only): WER, CER on `split=="test"` rows.
- **AVSR mode** (lip + audio): WER, CER on `split=="test"` rows with `audio_present=true`.
- Same metrics on a held-out LRS3 test slice if you fine-tuned on LRS3 — catches regressions.

To be implemented: `src/eval.py`. Uses `jiwer` for WER/CER:

```python
# Sketch
import jiwer
preds, refs = [], []
for row in gold_test_rows:
    r = avsr.transcribe(Path(row["mouth_roi_path"]).resolve())
    preds.append(r.text); refs.append(row["transcript_human"])
print("WER:", jiwer.wer(refs, preds))
print("CER:", jiwer.cer(refs, preds))
```

Reference numbers from the USR2 ICLR 2026 paper (LRS3 test): VSR WER in single-digit territory for the Huge model, lower for AVSR. Your numbers will be worse — your test set is small and noisier.

Don't chase absolute numbers; track **delta** between (a) public USR2, (b) public USR2 + your fine-tune.

## Stage 5 — Optional LLM rescoring

After Stage 4 numbers exist, check whether `rescore.polish()` (punctuation cleanup) or `rescore.rescore_nbest()` (n-best selection from beam search) reduces WER on `split=="test"`.

Rules:
- Always report rescored AND raw numbers. Rescoring is a post-processor, not part of the model claim.
- N-best rescoring must be constrained to pick **one of the existing hypotheses**. The LLM is not allowed to emit free-form text — it will hallucinate.
- If rescoring hurts WER, don't ship it.

## What we explicitly are NOT doing

- **Training the backbone from scratch.** Your manifest is too small; always fine-tune on top of USR2's public weights.
- **Treating Gemini / pseudo-labels as ground truth in training.** `transcript_human` is the only training label; `pseudo_transcript` is metadata.
- **Mixing LRS3 audio into our manifest.** Different corpus, different licensing, different preprocessing — keep them parallel and separate.
- **Fine-tuning Gemini / Grok / MiniMax as the recognition model.** If we fine-tune any LLM, it's only as a text post-processor on top of AVSR.

## Reproducibility checklist (run before any training)

1. `python scripts/verify_pipeline.py` exits 0.
2. `data/manifest.jsonl` backed up locally (it's gitignored).
3. Random seeds pinned (torch, numpy, python).
4. Each training run writes `runs/<timestamp>/config.yaml` with the full hyperparameter set + `git rev-parse HEAD` of this repo + USR2 commit SHA.
5. Checkpoints saved per epoch with `val_wer` and `git_sha` embedded in the filename.

## Open questions to revisit before Stage 3

- USR2's `demo.py` does its own MediaPipe ROI extraction. Does that conflict with our `data/rois/<id>.mp4`, or should we hand it the raw clip and let it preprocess? Likely: pass raw clip, skip our preprocessing.
- Audio resampling: USR2 expects 16 kHz mono; we already capture at 16 kHz. Verify USR2 doesn't down/upsample.
- Speaker embeddings as auxiliary input — relevant only once multi-speaker data accumulates.

This document evolves with the project. Update each stage's "acceptance gate" outcomes here as they're hit.
