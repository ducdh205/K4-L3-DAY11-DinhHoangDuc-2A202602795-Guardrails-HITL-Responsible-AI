"""Forensic audit log for the defense pipeline."""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path


def default_audit_log_path() -> str:
    """Always resolve to <repo>/outputs/… (safe when cwd is src/)."""
    repo_root = Path(__file__).resolve().parents[2]
    return str(repo_root / "outputs" / "audit_log.json")


class AuditLogPlugin:
    """Framework-agnostic audit logger (wire into ADK callbacks or a pipeline)."""

    def __init__(self):
        self.name = "audit_log"
        self.logs: list[dict] = []
        self._open: dict[str, dict] = {}

    @staticmethod
    def _key(user_id: str, request_id: str | None) -> str:
        return request_id or user_id

    def record_input(self, *, user_id: str, text: str, request_id: str | None = None):
        """Record the input and start a latency timer for this request."""
        key = self._key(user_id, request_id)
        self._open[key] = {
            "request_id": request_id or key,
            "user_id": user_id,
            "input": text,
            "started_at": utc_now_iso(),
            "_started_monotonic": time.perf_counter(),
        }
        return key

    def record_output(
        self,
        *,
        user_id: str,
        text: str,
        blocked: bool = False,
        layer: str | None = None,
        request_id: str | None = None,
    ):
        """Complete an audit event with outcome, layer, and elapsed time."""
        key = self._key(user_id, request_id)
        event = self._open.pop(key, None)
        if event is None:
            event = {
                "request_id": request_id or key,
                "user_id": user_id,
                "input": None,
                "started_at": utc_now_iso(),
                "_started_monotonic": time.perf_counter(),
            }

        started = event.pop("_started_monotonic")
        event.update(
            {
                "output": text,
                "blocked": bool(blocked),
                "layer": layer,
                "completed_at": utc_now_iso(),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        )
        self.logs.append(event)
        return event

    def export_json(self, filepath: str | None = None):
        """Write logs as a JSON array under repo-root ``outputs/`` by default."""
        path = Path(filepath or default_audit_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.logs, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(path)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
