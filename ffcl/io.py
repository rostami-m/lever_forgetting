"""Resumable JSONL result store. Each experiment appends one record per completed cell and can
skip cells already present, so runs are interruptible/restartable on any machine.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Callable, Iterable, List, Tuple


class Results:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def records(self) -> List[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return out

    def done_keys(self, key: Callable[[dict], Tuple]) -> set:
        return {key(r) for r in self.records()}

    def append(self, record: dict) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def count(self) -> int:
        return len(self.records())
