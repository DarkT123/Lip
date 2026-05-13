"""End-to-end integrity checks on the data pipeline.

Run anytime — especially after collecting clips, preprocessing, or splitting:

    python scripts/verify_pipeline.py

Exits non-zero if any check fails. Designed so CI / pre-training scripts
can gate on it.

Checks:
  - Schema validity (required fields, enums, type/consistency)
  - Raw clip files exist for non-deleted rows
  - Tombstoned rows have no files left on disk
  - Reviewed rows have transcript_human
  - Preprocessed rows have mouth_roi and landmarks on disk
  - Audio rows have audio_path file on disk
  - val/test rows are all label_quality == "gold" (unless explicitly opted out)
  - No speaker_id appears in multiple splits (leakage check, speaker-disjoint)
  - No session_id appears more than once (sanity)
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

# Make src/ importable.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import dataset  # noqa: E402


def _exists(rel: str | None) -> bool:
    return bool(rel) and (dataset.DATA_DIR / rel).exists()


def check_schema(rows: dict[str, dict]) -> list[str]:
    errs = []
    for sid, row in rows.items():
        for missing in (set(dataset.Entry.__dataclass_fields__) - set(row.keys())):
            errs.append(f"{sid}: missing schema field '{missing}'")
        for e in dataset.validate_entry(row):
            errs.append(f"{sid}: {e}")
    return errs


def check_raw_clips(rows: dict[str, dict]) -> list[str]:
    errs = []
    for sid, row in rows.items():
        if row.get("deleted"):
            for f in ("raw_clip_path", "mouth_roi_path", "landmarks_path", "audio_path"):
                if _exists(row.get(f)):
                    errs.append(f"{sid}: tombstoned but {f} still exists on disk")
            continue
        rcp = row.get("raw_clip_path")
        if not rcp:
            errs.append(f"{sid}: active row missing raw_clip_path")
        elif not _exists(rcp):
            errs.append(f"{sid}: raw_clip_path={rcp} missing on disk")
    return errs


def check_reviewed_transcripts(rows: dict[str, dict]) -> list[str]:
    errs = []
    for sid, row in rows.items():
        if row.get("deleted"):
            continue
        if row.get("review_status") == "reviewed" and not row.get("transcript_human"):
            errs.append(f"{sid}: review_status=reviewed but transcript_human empty")
    return errs


def check_preprocessed_files(rows: dict[str, dict]) -> list[str]:
    errs = []
    for sid, row in rows.items():
        if row.get("deleted"):
            continue
        if not row.get("preprocessing_version"):
            continue
        for f in ("mouth_roi_path", "landmarks_path"):
            if not _exists(row.get(f)):
                errs.append(f"{sid}: preprocessing_version set but {f} missing on disk")
        ap = row.get("audio_path")
        if ap and not _exists(ap):
            errs.append(f"{sid}: audio_path={ap} missing on disk")
    return errs


def check_val_test_gold_only(rows: dict[str, dict]) -> list[str]:
    errs = []
    for sid, row in rows.items():
        if row.get("deleted"):
            continue
        if row.get("split") in ("val", "test") and row.get("label_quality") != "gold":
            errs.append(
                f"{sid}: in split={row['split']} but "
                f"label_quality={row.get('label_quality')!r} (val/test must be gold)"
            )
    return errs


def check_speaker_leakage(rows: dict[str, dict]) -> list[str]:
    errs = []
    speaker_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows.values():
        if row.get("deleted"):
            continue
        sp = row.get("split")
        if sp in ("unassigned", None):
            continue
        speaker_splits[row["speaker_id"]].add(sp)
    for sp, splits in speaker_splits.items():
        if len(splits) > 1:
            errs.append(f"speaker_id={sp!r} appears in multiple splits: {sorted(splits)}")
    return errs


def check_session_leakage(rows: dict[str, dict]) -> list[str]:
    """A given session should only ever be in one split (always true after read_latest, but double-check)."""
    errs = []
    seen = defaultdict(set)
    for row in rows.values():
        if row.get("deleted"):
            continue
        sp = row.get("split")
        if sp not in ("train", "val", "test"):
            continue
        seen[row["session_id"]].add(sp)
    for sid, splits in seen.items():
        if len(splits) > 1:
            errs.append(f"session_id={sid} resolved to multiple splits: {sorted(splits)}")
    return errs


CHECKS = [
    ("Schema + per-entry validation", check_schema),
    ("Raw clip files exist", check_raw_clips),
    ("Reviewed rows have transcripts", check_reviewed_transcripts),
    ("Preprocessing outputs exist", check_preprocessed_files),
    ("val/test are gold-only", check_val_test_gold_only),
    ("No speaker leakage across splits", check_speaker_leakage),
    ("No session crosses splits", check_session_leakage),
]


def main() -> int:
    rows = dataset.read_latest()
    print(f"Loaded {len(rows)} session(s) from {dataset.MANIFEST}")
    if not rows:
        print("(manifest is empty — record some clips first)")
        return 0

    total_errors = 0
    for name, fn in CHECKS:
        errs = fn(rows)
        if errs:
            print(f"\n[FAIL] {name}: {len(errs)} error(s)")
            for e in errs[:20]:
                print(f"    - {e}")
            if len(errs) > 20:
                print(f"    ... and {len(errs) - 20} more")
            total_errors += len(errs)
        else:
            print(f"[ ok ] {name}")

    if total_errors:
        print(f"\nVerification FAILED: {total_errors} total error(s)")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
