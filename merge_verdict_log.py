#!/usr/bin/env python3
"""Git merge driver for state/ai_verdict_log.json.

The file is an append-only log of AI verdict decisions (one entry per
date/market/symbol). Concurrent CI runs each append their own entries, so a
real conflict just means both sides added rows the other side hasn't seen
yet -- the correct merge is a union, deduplicated by (date, market, symbol)
since that's the natural one-verdict-per-stock-per-day key. Invalid input
fails closed so Git leaves the conflict visible rather than silently
publishing a bad log.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path


def read_log(path: str, label: str) -> list:
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid JSON: {exc}") from exc
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise ValueError(f"{label} does not have an 'entries' list")
    return entries


def key_of(entry: dict) -> tuple:
    return (entry.get("date"), entry.get("market"), entry.get("symbol"))


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: merge_verdict_log.py BASE OURS THEIRS", file=sys.stderr)
        return 2

    _base, ours_path, theirs_path = sys.argv[1:]
    try:
        ours = read_log(ours_path, "ours")
        theirs = read_log(theirs_path, "theirs")
    except ValueError as exc:
        print(f"Verdict log merge refused: {exc}", file=sys.stderr)
        return 1

    merged: dict[tuple, dict] = {}
    order: list[tuple] = []
    for entry in ours + theirs:      # theirs applied after ours: wins on an exact-key collision
        k = key_of(entry)
        if k not in merged:
            order.append(k)
        merged[k] = entry
    merged_entries = [merged[k] for k in order]
    merged_entries.sort(key=lambda e: (e.get("date") or "", e.get("market") or "", e.get("symbol") or ""))

    target = Path(ours_path)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"entries": merged_entries}, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    print(f"Verdict log merge: {len(ours)} ours + {len(theirs)} theirs -> {len(merged_entries)} entries",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
