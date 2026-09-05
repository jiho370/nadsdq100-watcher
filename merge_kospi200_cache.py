#!/usr/bin/env python3
"""Git merge driver for output/kospi200_cache.json.

The file is a complete KRX snapshot, not an append-only record.  When two
writers overlap, retain the valid snapshot with the later as_of trading date.
For an equal date, retain "ours" (the branch currently being rebased onto).
Invalid or schema-incomplete input fails closed so Git leaves the conflict
visible rather than silently publishing a bad cache.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def read_snapshot(path: str, label: str) -> tuple[dict, str]:
    try:
        with open(path, encoding="utf-8") as fh:
            snapshot = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc

    as_of = str(snapshot.get("as_of", ""))
    data = snapshot.get("data")
    if len(as_of) != 8 or not as_of.isdecimal() or not isinstance(data, dict) or not data:
        raise ValueError(f"{label} does not have a valid as_of/data snapshot")
    return snapshot, as_of


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: merge_kospi200_cache.py BASE OURS THEIRS", file=sys.stderr)
        return 2

    _base, ours_path, theirs_path = sys.argv[1:]
    try:
        ours, ours_as_of = read_snapshot(ours_path, "ours")
        theirs, theirs_as_of = read_snapshot(theirs_path, "theirs")
    except ValueError as exc:
        print(f"KOSPI200 cache merge refused: {exc}", file=sys.stderr)
        return 1

    # On a rebase, "ours" is the already-published upstream snapshot.  Keeping
    # it for the same trading date avoids replacing a published cache with an
    # indistinguishable-date rewrite.
    winner, source = (theirs, "theirs") if theirs_as_of > ours_as_of else (ours, "ours")

    target = Path(ours_path)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(winner, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(f"KOSPI200 cache merge: kept {source} (as_of={winner['as_of']})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
