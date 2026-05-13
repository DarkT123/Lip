"""Mouth ROI extraction + face landmarks + audio extraction for VSR/AVSR.

Run as a batch job after collecting clips:

    python src/preprocess.py             # process all unprocessed sessions
    python src/preprocess.py --redo      # force-reprocess everything

Per session this produces:
    data/rois/<session_id>.mp4         96x96 grayscale mouth crop, fps preserved
    data/landmarks/<session_id>.npz    478 face mesh landmarks per frame
    data/audio/<session_id>.wav        16 kHz mono PCM, if audio_present

A new manifest row is appended with mouth_roi_path, landmarks_path,
audio_path, and preprocessing_version set.

Bump PREPROCESSING_VERSION whenever the crop pipeline changes. Old derived
files stay on disk until removed manually; rerun with --redo to regenerate.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from dataset import (
    AUDIO_DIR,
    DATA_DIR,
    LANDMARKS_DIR,
    ROIS_DIR,
    Entry,
    append,
    iter_active,
)

PREPROCESSING_VERSION = "v1"
ROI_SIZE = 96  # AV-HuBERT / Auto-AVSR convention
CROP_PAD_FACTOR = 1.6  # mouth width × this = crop side

# Canonical MediaPipe Face Mesh lip landmarks (outer + inner rings).
# Source: github.com/google/mediapipe/blob/master/mediapipe/python/solutions/face_mesh_connections.py
LIP_LANDMARK_IDXS: tuple[int, ...] = (
    # outer ring
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
    # inner ring
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
)


def _load_face_mesh():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def _crop_mouth(frame: np.ndarray, landmarks) -> np.ndarray:
    h, w = frame.shape[:2]
    pts = np.array(
        [(landmarks[i].x * w, landmarks[i].y * h) for i in LIP_LANDMARK_IDXS],
        dtype=np.float32,
    )
    cx, cy = pts.mean(axis=0)
    mouth_w = pts[:, 0].max() - pts[:, 0].min()
    mouth_h = pts[:, 1].max() - pts[:, 1].min()
    half = int(max(mouth_w, mouth_h) * CROP_PAD_FACTOR / 2)
    half = max(half, 48)
    x0 = max(0, int(cx - half))
    y0 = max(0, int(cy - half))
    x1 = min(w, int(cx + half))
    y1 = min(h, int(cy + half))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (ROI_SIZE, ROI_SIZE), interpolation=cv2.INTER_AREA)


def _extract_audio(src: Path, dst: Path, sample_rate: int = 16000) -> bool:
    """Demux audio from a clip to mono PCM WAV. Returns False if no audio stream."""
    if shutil.which("ffmpeg") is None:
        print("[warn] ffmpeg missing; cannot extract audio")
        return False
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(src)],
        capture_output=True, text=True,
    )
    if not probe.stdout.strip():
        return False
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-i", str(src),
         "-vn", "-ac", "1", "-ar", str(sample_rate),
         "-c:a", "pcm_s16le",
         str(dst)],
        capture_output=True,
    )
    return result.returncode == 0 and dst.exists()


def process_session(row: dict) -> bool:
    session_id = row["session_id"]
    if not row.get("raw_clip_path"):
        return False
    src = DATA_DIR / row["raw_clip_path"]
    if not src.exists():
        print(f"[skip] {session_id}: source clip missing")
        return False

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or row.get("fps") or 30

    roi_path = ROIS_DIR / f"{session_id}.mp4"
    lm_path = LANDMARKS_DIR / f"{session_id}.npz"
    audio_path = AUDIO_DIR / f"{session_id}.wav"

    writer = cv2.VideoWriter(
        str(roi_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (ROI_SIZE, ROI_SIZE),
        isColor=False,
    )

    face_mesh = _load_face_mesh()
    all_landmarks: list[np.ndarray] = []
    frame_count = 0
    missing_count = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            all_landmarks.append(np.full((478, 3), np.nan, dtype=np.float32))
            writer.write(np.zeros((ROI_SIZE, ROI_SIZE), dtype=np.uint8))
            missing_count += 1
            frame_count += 1
            continue
        lms = result.multi_face_landmarks[0].landmark
        all_landmarks.append(
            np.array([(p.x, p.y, p.z) for p in lms], dtype=np.float32)
        )
        roi = _crop_mouth(frame, lms)
        writer.write(roi)
        frame_count += 1

    cap.release()
    writer.release()
    face_mesh.close()

    if not all_landmarks:
        print(f"[skip] {session_id}: source has no decodable frames")
        roi_path.unlink(missing_ok=True)
        return False

    np.savez_compressed(lm_path, landmarks=np.stack(all_landmarks))

    audio_out: str | None = None
    if row.get("audio_present"):
        if _extract_audio(src, audio_path):
            audio_out = str(audio_path.relative_to(DATA_DIR))

    updated = Entry(
        **{k: v for k, v in row.items() if k in Entry.__dataclass_fields__}
    )
    updated.mouth_roi_path = str(roi_path.relative_to(DATA_DIR))
    updated.landmarks_path = str(lm_path.relative_to(DATA_DIR))
    updated.audio_path = audio_out
    updated.preprocessing_version = PREPROCESSING_VERSION
    append(updated)

    detect_rate = 1.0 - (missing_count / max(frame_count, 1))
    print(
        f"[done] {session_id}: {frame_count} frames, "
        f"face-detect rate {detect_rate:.0%}, audio={'yes' if audio_out else 'no'}"
    )
    return True


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo", action="store_true", help="reprocess even if up to date")
    args = parser.parse_args(argv)

    processed = 0
    skipped = 0
    for row in iter_active():
        if not args.redo and row.get("preprocessing_version") == PREPROCESSING_VERSION:
            skipped += 1
            continue
        if process_session(row):
            processed += 1
    print(f"\nProcessed {processed} session(s). Skipped {skipped} already up-to-date.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
