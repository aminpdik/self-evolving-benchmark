"""Append-only JSONL persistence layer.

Stores accepted benchmark items and run logs as JSON Lines (one JSON object per line)."""


from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator


def utc_now_iso() -> str:
    """Get the current UTC timestamp as an ISO-8601 string.

Returns:
    Current time in UTC, e.g. '2026-02-17T22:01:02.123456+00:00'."""

    return datetime.now(timezone.utc).isoformat()


@dataclass
class StorePaths:
    """Lightweight container for output file paths.

Attributes:
    benchmark_jsonl: Path to the JSONL file that stores accepted benchmark items.
    runlog_jsonl: Path to the JSONL file that stores run events (accepts/rejects/errors)."""

    benchmark_jsonl: str
    runlog_jsonl: str


class JSONLStore:
    """Append-only JSONL store for benchmark items and run logs."""

    def __init__(self, paths: StorePaths):
        """Create a JSONLStore.

Args:
    paths: StorePaths containing output file paths for benchmark items and run logs.

Returns:
    None.

How it works:
    Stores the paths and ensures the parent directories exist so appends will succeed."""

        self.paths = paths
        os.makedirs(os.path.dirname(paths.benchmark_jsonl), exist_ok=True)
        os.makedirs(os.path.dirname(paths.runlog_jsonl), exist_ok=True)

    def append_benchmark_item(self, item: Dict[str, Any]) -> None:
        """Append one accepted benchmark item to the benchmark JSONL file.

Args:
    item: Dict representing a benchmark item. This method will add 'created_at' if missing.

Returns:
    None.

How it works:
    Writes one JSON object per line (JSONL). This is append-only to keep runs incremental and easy to stream."""

        item = dict(item)
        item.setdefault("created_at", utc_now_iso())
        with open(self.paths.benchmark_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def append_runlog(self, record: Dict[str, Any]) -> None:
        """Append a run log record to the run log JSONL file.

Args:
    record: Dict representing an event (e.g., reject reason, errors, acceptance stats). Adds 'created_at' if missing.

Returns:
    None."""

        record = dict(record)
        record.setdefault("created_at", utc_now_iso())
        with open(self.paths.runlog_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def iter_benchmark_items(self) -> Iterator[Dict[str, Any]]:
        """Iterate previously stored benchmark items.

Returns:
    An iterator of dicts loaded from the benchmark JSONL file. If the file does not exist,
    returns an empty iterator.

How it works:
    Lazily reads the file line-by-line, skipping blank lines and json.loads() each record."""

        if not os.path.exists(self.paths.benchmark_jsonl):
            return iter(())

        def gen():
            """Generator over benchmark items stored in the JSONL file.

Returns:
    Yields one decoded JSON object (dict) per non-empty line in the benchmark file.

How it works:
    Opens the JSONL file and streams it line-by-line to avoid loading everything into memory."""

            with open(self.paths.benchmark_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)

        return gen()
