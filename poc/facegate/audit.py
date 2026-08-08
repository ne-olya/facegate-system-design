"""Журнал решений о доступе.

Append-only JSONL: одна строка на решение, без кадра и без вектора. Этого хватает,
чтобы поднять цепочку по event_id при разборе, и нечего терять при утечке журнала.
"""

import json
from pathlib import Path


class AuditLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> str:
        audit_id = f"a-{abs(hash(record['decision_id'])) % 10**6}"
        line = dict(record, audit_id=audit_id)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
        return audit_id

    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line]
