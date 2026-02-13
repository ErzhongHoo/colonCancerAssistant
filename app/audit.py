from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class AuditEvent:
    timestamp: float
    event_type: str
    session_id: str
    details: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class AuditLogger:
    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._lock = threading.Lock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, session_id: str, details: dict[str, Any]) -> None:
        event = AuditEvent(
            timestamp=time.time(),
            event_type=event_type,
            session_id=session_id,
            details=details,
        )
        line = json.dumps(event.as_dict(), ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as fp:
                fp.write(line + "\n")

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.log_path.exists():
            return []
        limit = max(1, min(limit, 500))
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        items = []
        for line in lines[-limit:]:
            try:
                items.append(json.loads(line))
            except Exception:
                continue
        return items

    def ttl_proof(self, session_id: str | None = None) -> list[dict[str, Any]]:
        events = self.recent(limit=500)
        proofs = [item for item in events if item.get("event_type") == "session_expired"]
        if session_id:
            proofs = [item for item in proofs if str(item.get("session_id", "")) == session_id]
        return proofs
