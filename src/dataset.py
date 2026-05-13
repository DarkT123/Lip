"""Dataset / manifest management for the lip-to-text VSR pipeline.

The manifest is an append-only JSONL log at data/manifest.jsonl.
Each line is a full snapshot of one session's metadata. Readers should treat
the LAST entry for a given session_id as authoritative — later writes
supersede earlier ones (preprocessing adds ROI/audio paths, splitting adds
the split label, tombstoning marks deleted=True).

See README.md for the field-by-field schema.
"""
from __future__ import annotations

import json
import os
import shutil
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
ROIS_DIR = DATA_DIR / "rois"
LANDMARKS_DIR = DATA_DIR / "landmarks"
AUDIO_DIR = DATA_DIR / "audio"
MANIFEST = DATA_DIR / "manifest.jsonl"

SCHEMA_VERSION = 1

REVIEW_STATUSES = {"unreviewed", "reviewed", "needs_recheck"}
LABEL_QUALITIES = {"gold", "silver", "unknown"}
SPLITS = {"unassigned", "train", "val", "test"}
RESOLUTION_RE = re.compile(r"^\d+x\d+$")


@dataclass
class Entry:
    session_id: str
    speaker_id: str = "default"
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    raw_clip_path: str | None = None
    mouth_roi_path: str | None = None
    landmarks_path: str | None = None
    audio_path: str | None = None
    duration_seconds: float | None = None
    fps: int | None = None
    resolution: str | None = None
    audio_present: bool = False
    language: str = "en"
    pseudo_transcript: str | None = None
    pseudo_label_model: str | None = None
    prompt_version: str | None = None
    transcript_human: str | None = None
    review_status: str = "unreviewed"
    label_quality: str = "unknown"
    split: str = "unassigned"
    preprocessing_version: str | None = None
    deleted: bool = False


def _ensure_dirs() -> None:
    for d in (CLIPS_DIR, ROIS_DIR, LANDMARKS_DIR, AUDIO_DIR):
        d.mkdir(parents=True, exist_ok=True)


def new_session_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def speaker_id() -> str:
    return os.environ.get("LIPSYNC_SPEAKER_ID", "default")


def import_clip(src: Path, session_id: str) -> Path:
    """Move a temp clip into the data dir and return its new path."""
    _ensure_dirs()
    dst = CLIPS_DIR / f"{session_id}.mp4"
    shutil.move(str(src), dst)
    return dst


def append(entry: Entry) -> None:
    _ensure_dirs()
    entry.ts = datetime.now(timezone.utc).isoformat()
    with open(MANIFEST, "a") as f:
        f.write(json.dumps(asdict(entry)) + "\n")


def read_latest() -> dict[str, dict]:
    """Return latest-wins entry per session_id."""
    if not MANIFEST.exists():
        return {}
    latest: dict[str, dict] = {}
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            latest[row["session_id"]] = row
    return latest


def iter_active() -> Iterable[dict]:
    """Yield non-deleted entries (latest snapshot per session)."""
    for row in read_latest().values():
        if not row.get("deleted"):
            yield row


def validate_entry(row: dict) -> list[str]:
    """Return list of validation errors. Empty list means the entry is OK."""
    errs: list[str] = []
    sid = row.get("session_id")
    if not sid or not isinstance(sid, str):
        errs.append("session_id: missing or not a string")
    if not row.get("speaker_id"):
        errs.append("speaker_id: missing")
    if not row.get("ts"):
        errs.append("ts: missing")
    else:
        try:
            datetime.fromisoformat(row["ts"])
        except (TypeError, ValueError):
            errs.append(f"ts: not ISO 8601 ({row['ts']!r})")

    if row.get("review_status") not in REVIEW_STATUSES:
        errs.append(f"review_status: must be one of {sorted(REVIEW_STATUSES)}")
    if row.get("label_quality") not in LABEL_QUALITIES:
        errs.append(f"label_quality: must be one of {sorted(LABEL_QUALITIES)}")
    if row.get("split") not in SPLITS:
        errs.append(f"split: must be one of {sorted(SPLITS)}")

    res = row.get("resolution")
    if res is not None and not (isinstance(res, str) and RESOLUTION_RE.match(res)):
        errs.append(f"resolution: must match WIDTHxHEIGHT (got {res!r})")

    fps = row.get("fps")
    if fps is not None and not (isinstance(fps, int) and fps > 0):
        errs.append(f"fps: must be positive int (got {fps!r})")

    dur = row.get("duration_seconds")
    if dur is not None and not (isinstance(dur, (int, float)) and dur > 0):
        errs.append(f"duration_seconds: must be positive number (got {dur!r})")

    if not isinstance(row.get("audio_present"), bool):
        errs.append("audio_present: must be bool")
    if not isinstance(row.get("deleted"), bool):
        errs.append("deleted: must be bool")

    # Consistency checks (only for non-deleted entries)
    if not row.get("deleted"):
        if row.get("label_quality") == "gold":
            if not row.get("transcript_human"):
                errs.append("label_quality=gold requires transcript_human")
            if row.get("review_status") != "reviewed":
                errs.append("label_quality=gold requires review_status=reviewed")
        if row.get("review_status") == "reviewed" and not row.get("transcript_human"):
            errs.append("review_status=reviewed requires transcript_human")
        if row.get("mouth_roi_path") and not row.get("preprocessing_version"):
            errs.append("mouth_roi_path set but preprocessing_version is empty")
        if row.get("preprocessing_version"):
            for f in ("mouth_roi_path", "landmarks_path"):
                if not row.get(f):
                    errs.append(f"preprocessing_version set but {f} is empty")
        if row.get("audio_path") and not row.get("audio_present"):
            errs.append("audio_path set but audio_present is False")

    return errs


def tombstone(session_id: str, reason: str = "user_discarded") -> None:
    """Mark a session as deleted, remove its derived files."""
    latest = read_latest().get(session_id)
    if latest:
        for key in ("raw_clip_path", "mouth_roi_path", "landmarks_path", "audio_path"):
            p = latest.get(key)
            if p:
                full = DATA_DIR / p
                if full.exists():
                    full.unlink()
            latest[key] = None
        latest["label_quality"] = "unknown"
        latest["deleted"] = True
        base = Entry(**{k: v for k, v in latest.items() if k in Entry.__dataclass_fields__})
    else:
        base = Entry(session_id=session_id, deleted=True)
    append(base)
