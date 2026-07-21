"""Structured status, event, metric, and console-log output for one job."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


TERMINAL_STATES = {"SUCCEEDED", "FAILED", "CANCELLED"}
_UNSET = object()
_TEE_LOCK = threading.RLock()
_TEE_PATH: Optional[Path] = None
_TEE_FILE = None
_ORIGINAL_STDOUT = None
_ORIGINAL_STDERR = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    item = getattr(value, "item", None)
    if callable(item):
        return item()
    raise TypeError("Object of type {} is not JSON serializable".format(type(value).__name__))


class _TeeStream:
    def __init__(self, console):
        self._console = console

    def write(self, value):
        if not value:
            return 0
        with _TEE_LOCK:
            written = self._console.write(value)
            if _TEE_FILE is not None:
                _TEE_FILE.write(value)
        return written if written is not None else len(value)

    def flush(self):
        with _TEE_LOCK:
            self._console.flush()
            if _TEE_FILE is not None:
                _TEE_FILE.flush()

    def __getattr__(self, name):
        return getattr(self._console, name)


def install_run_log(path: Path) -> None:
    """Mirror stdout and stderr to one append-only run.log file."""
    global _ORIGINAL_STDERR, _ORIGINAL_STDOUT, _TEE_FILE, _TEE_PATH
    resolved = path.resolve()
    if _TEE_PATH == resolved:
        return
    if _TEE_PATH is not None:
        raise RuntimeError("run.log has already been initialized at {}".format(_TEE_PATH))
    resolved.parent.mkdir(parents=True, exist_ok=True)
    _TEE_FILE = resolved.open("a", encoding="utf-8", buffering=1)
    _TEE_PATH = resolved
    _ORIGINAL_STDOUT = sys.stdout
    _ORIGINAL_STDERR = sys.stderr
    sys.stdout = _TeeStream(_ORIGINAL_STDOUT)
    sys.stderr = _TeeStream(_ORIGINAL_STDERR)


def close_run_log() -> None:
    global _ORIGINAL_STDERR, _ORIGINAL_STDOUT, _TEE_FILE, _TEE_PATH
    with _TEE_LOCK:
        if _TEE_FILE is None:
            return
        sys.stdout.flush()
        sys.stderr.flush()
        if _ORIGINAL_STDOUT is not None:
            sys.stdout = _ORIGINAL_STDOUT
        if _ORIGINAL_STDERR is not None:
            sys.stderr = _ORIGINAL_STDERR
        _TEE_FILE.close()
        _TEE_FILE = None
        _TEE_PATH = None
        _ORIGINAL_STDOUT = None
        _ORIGINAL_STDERR = None


class JobRuntime:
    """Own the filesystem runtime protocol for a single train or validate job."""

    def __init__(
        self,
        job_root: Path,
        job_type: str,
        task_id: Optional[str],
        model_name: str,
        test_id: Optional[str] = None,
        status_interval_seconds: float = 1.0,
    ) -> None:
        self.job_root = Path(job_root).resolve()
        self.runtime_dir = self.job_root / "runtime"
        self.logs_dir = self.job_root / "logs"
        self.status_path = self.runtime_dir / "status.json"
        self.events_path = self.runtime_dir / "events.jsonl"
        self.metrics_path = self.runtime_dir / "metrics.jsonl"
        self.log_path = self.logs_dir / "run.log"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        install_run_log(self.log_path)

        self.job_type = job_type
        self.task_id = task_id
        self.model_name = model_name
        self.test_id = test_id
        self.status_interval_seconds = status_interval_seconds
        self._lock = threading.RLock()
        self._last_status_write = 0.0
        self._status: Dict[str, Any] = self._read_existing_status()
        self._sequence = self._read_last_sequence()

    def _read_existing_status(self) -> Dict[str, Any]:
        if not self.status_path.is_file():
            return {}
        try:
            with self.status_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _read_last_sequence(self) -> int:
        if not self.events_path.is_file():
            return 0
        last_sequence = 0
        try:
            with self.events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    last_sequence = max(last_sequence, int(record.get("seq", 0)))
        except (OSError, ValueError, json.JSONDecodeError):
            return last_sequence
        return last_sequence

    @staticmethod
    def _default_progress() -> Dict[str, Any]:
        return {
            "current": 0,
            "total": None,
            "unit": "step",
            "percentage": 0.0,
            "epoch": 0,
            "total_epochs": None,
            "estimated_seconds_remaining": None,
        }

    def start(self, created_at: Optional[str] = None) -> None:
        now = utc_now()
        existing_created_at = self._status.get("created_at")
        existing_started_at = self._status.get("started_at")
        progress = self._default_progress()
        if isinstance(self._status.get("progress"), dict):
            progress.update(self._status["progress"])
        self._status = {
            "schema_version": 1,
            "task_id": self._status.get("task_id", self.task_id),
            "model_name": self._status.get("model_name", self.model_name),
            "test_id": self._status.get("test_id", self.test_id),
            "job_type": self.job_type,
            "state": "RUNNING",
            "phase": "INITIALIZING",
            "created_at": existing_created_at or created_at or now,
            "started_at": existing_started_at or now,
            "updated_at": now,
            "finished_at": None,
            "progress": progress,
            "error": None,
        }
        self._write_status(force=True)
        self.emit_event("job_started", message="{} job started".format(self.job_type))

    def _atomic_write_json(self, path: Path, payload: Mapping[str, Any]) -> None:
        temp_path = path.with_name("{}.tmp.{}".format(path.name, os.getpid()))
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                default=_json_default,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temp_path), str(path))

    def _write_status(self, force: bool = False) -> bool:
        now_monotonic = time.monotonic()
        if (
            not force
            and now_monotonic - self._last_status_write < self.status_interval_seconds
        ):
            return False
        self._status["updated_at"] = utc_now()
        self._atomic_write_json(self.status_path, self._status)
        self._last_status_write = now_monotonic
        return True

    def change_phase(self, phase: str, message: Optional[str] = None) -> None:
        with self._lock:
            old_phase = self._status.get("phase")
            if old_phase == phase:
                return
            self._status["phase"] = phase
            self._write_status(force=True)
            self.emit_event(
                "phase_changed",
                message=message or "Phase changed to {}".format(phase),
                data={"from": old_phase, "to": phase},
            )

    def update_progress(
        self,
        current: int,
        total: Optional[int],
        epoch: Optional[int] = None,
        total_epochs: Optional[int] = None,
        unit: str = "step",
        estimated_seconds_remaining: Optional[int] = None,
        force: bool = False,
    ) -> None:
        with self._lock:
            percentage = 0.0
            if total is not None and total > 0:
                percentage = round(min(100.0, 100.0 * current / float(total)), 2)
            progress = self._default_progress()
            progress.update(
                {
                    "current": int(current),
                    "total": int(total) if total is not None else None,
                    "unit": unit,
                    "percentage": percentage,
                    "epoch": int(epoch) if epoch is not None else None,
                    "total_epochs": int(total_epochs) if total_epochs is not None else None,
                    "estimated_seconds_remaining": estimated_seconds_remaining,
                }
            )
            self._status["progress"] = progress
            self._write_status(force=force)

    def emit_event(
        self,
        event_type: str,
        level: str = "INFO",
        message: str = "",
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._sequence += 1
            record = {
                "seq": self._sequence,
                "timestamp": utc_now(),
                "level": level,
                "type": event_type,
                "message": message,
                "data": dict(data or {}),
            }
            self._append_jsonl(self.events_path, record, durable=True)

    def emit_metrics(
        self,
        phase: str,
        values: Mapping[str, Any],
        step: Optional[int] = None,
        epoch: Optional[int] = None,
    ) -> None:
        record = {
            "timestamp": utc_now(),
            "step": int(step) if step is not None else None,
            "epoch": int(epoch) if epoch is not None else None,
            "phase": phase,
            "values": dict(values),
        }
        self._append_jsonl(self.metrics_path, record, durable=False)

    def _append_jsonl(
        self, path: Path, payload: Mapping[str, Any], durable: bool
    ) -> None:
        line = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            default=_json_default,
        )
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            if durable:
                os.fsync(handle.fileno())

    def checkpoint_saved(self, path: Path, step: Optional[int] = None) -> None:
        try:
            relative_path = str(Path(path).resolve().relative_to(self.job_root))
        except ValueError:
            relative_path = str(path)
        data: Dict[str, Any] = {"path": relative_path}
        if step is not None:
            data["step"] = int(step)
        self.emit_event(
            "checkpoint_saved",
            message="Checkpoint saved",
            data=data,
        )

    def succeed(self) -> None:
        with self._lock:
            progress = self._status.get("progress")
            if not isinstance(progress, dict):
                progress = self._default_progress()
            if progress.get("total") is not None:
                progress["current"] = progress["total"]
            progress["percentage"] = 100.0
            self._status.update(
                {
                    "state": "SUCCEEDED",
                    "phase": "FINALIZING",
                    "finished_at": utc_now(),
                    "progress": progress,
                    "error": None,
                }
            )
            self._write_status(force=True)
            self.emit_event("job_succeeded", message="{} job succeeded".format(self.job_type))

    def fail(self, code: str, message: str) -> None:
        with self._lock:
            self._status.update(
                {
                    "state": "FAILED",
                    "finished_at": utc_now(),
                    "error": {"code": code, "message": message},
                }
            )
            self._write_status(force=True)
            self.emit_event(
                "job_failed",
                level="ERROR",
                message=message,
                data={"code": code},
            )

    def cancel(self, message: str = "Job cancelled") -> None:
        with self._lock:
            self._status.update(
                {
                    "state": "CANCELLED",
                    "finished_at": utc_now(),
                    "error": None,
                }
            )
            self._write_status(force=True)
            self.emit_event("job_cancelled", level="WARNING", message=message)

    @property
    def state(self) -> Optional[str]:
        return self._status.get("state")

    def close(self) -> None:
        close_run_log()
