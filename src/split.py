"""Train / val / test split assignment.

Splits should be SPEAKER-DISJOINT to measure real generalization. With a
single speaker, this collapses to session-disjoint, which still avoids
trivial intra-session leakage but does not measure cross-speaker generalization.

Leakage policy:
  - When --by speaker, a speaker is assigned to exactly one of {train, val, test}.
  - Silver-quality rows belonging to a val/test speaker are DROPPED to
    "unassigned" (not pushed into train) — otherwise the same speaker would
    appear in both train and val/test.
  - Gold-only val/test by default. Use --include-silver to allow silver
    labels in val/test (not recommended).

Usage:
    python src/split.py                     # 80/10/10 by speaker, gold-only val/test
    python src/split.py --by session        # split by session_id
    python src/split.py --ratio 70 15 15    # custom ratio
    python src/split.py --include-silver    # allow silver in val/test
"""
from __future__ import annotations

import argparse
import hashlib
import sys

from dataset import Entry, append, iter_active


def _bucket(key: str, ratios: tuple[float, float, float]) -> str:
    h = int(hashlib.sha256(key.encode()).hexdigest(), 16) / 2 ** 256
    train_end = ratios[0]
    val_end = ratios[0] + ratios[1]
    if h < train_end:
        return "train"
    if h < val_end:
        return "val"
    return "test"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--by", choices=["speaker", "session"], default="speaker")
    parser.add_argument("--ratio", nargs=3, type=float, default=[80, 10, 10])
    parser.add_argument(
        "--include-silver",
        action="store_true",
        help="allow silver-quality entries in val/test (default: gold only)",
    )
    args = parser.parse_args(argv)

    total = sum(args.ratio)
    ratios = (args.ratio[0] / total, args.ratio[1] / total, args.ratio[2] / total)

    rows = list(iter_active())
    if not rows:
        print("No active rows in manifest.")
        return 0

    keying = "speaker_id" if args.by == "speaker" else "session_id"
    keys = {r[keying] for r in rows}

    if args.by == "speaker" and len(keys) < 3:
        print(
            f"WARNING: only {len(keys)} unique speaker(s); speaker-disjoint splits "
            f"will be degenerate. Consider `--by session` for now."
        )

    bucket_for_key = {k: _bucket(k, ratios) for k in keys}

    counts = {"train": 0, "val": 0, "test": 0, "unassigned": 0}
    updates = 0
    for row in rows:
        bucket = bucket_for_key[row[keying]]
        quality = row.get("label_quality")

        if bucket in ("val", "test"):
            if quality == "gold" or args.include_silver:
                new_split = bucket
            else:
                # Speaker/session is held out for val/test but this row isn't gold.
                # Moving it to train would leak the speaker. Drop it.
                new_split = "unassigned"
        else:
            new_split = "train"

        counts[new_split] += 1

        if row.get("split") == new_split:
            continue
        updated = Entry(**{k: v for k, v in row.items() if k in Entry.__dataclass_fields__})
        updated.split = new_split
        append(updated)
        updates += 1

    print(f"Split summary (by {args.by}, ratio {args.ratio}, gold-only={not args.include_silver}):")
    for k in ("train", "val", "test", "unassigned"):
        print(f"  {k}: {counts[k]}")
    print(f"  ({updates} manifest row(s) updated)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
