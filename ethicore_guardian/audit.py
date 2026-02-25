"""
Ethicore Engine™ - Guardian SDK — Append-Only Audit Log
Version: 1.0.0

Copyright © 2026 Oracles Technologies LLC
All Rights Reserved

Principle 13 (Ultimate Accountability): every Guardian decision is recorded so
that developers and operators can give an account of every analysis performed.
"God will bring every deed into judgment" (Ecclesiastes 12:14) — our systems
must maintain the same standard of transparency.

Principle 12 (Sacred Privacy): the audit log records *decisions* and
*metadata*, never raw prompt text.  Text is stored only as a SHA-256
fingerprint so the log cannot become a surveillance database.

Log location: ~/.ethicore/guardian_audit.log  (JSON Lines, one record per line)

The log is append-only by design.  Records are never modified or deleted
programmatically; rotation/archival is left to the host operating system's
log-management tooling (logrotate, etc.).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default log directory / file
# ---------------------------------------------------------------------------
_DEFAULT_LOG_DIR = Path.home() / ".ethicore"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "guardian_audit.log"


class AuditLogger:
    """
    Append-only audit logger for Guardian analysis decisions.

    Each call to ``record()`` appends a single JSON object (terminated by
    ``\\n``) to the log file.  The file is opened and closed for every write
    so that partial writes do not corrupt existing records even if the process
    is killed mid-operation.

    Thread / async safety: ``record()`` is synchronous and uses ``os.open``
    with ``O_APPEND`` which is atomic for small writes on POSIX systems.
    On Windows, ``open(..., 'a')`` in text mode is similarly safe for
    single-threaded / single-process usage.
    """

    def __init__(
        self,
        log_path: Optional[Path] = None,
        enabled: bool = True,
    ) -> None:
        self.log_path = Path(log_path) if log_path else _DEFAULT_LOG_FILE
        self.enabled = enabled
        self._records_written: int = 0

        if self.enabled:
            self._ensure_log_dir()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(
        self,
        text: str,
        analysis_result: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append one audit record to the log.

        Args:
            text:            The raw prompt that was analysed.  Only a SHA-256
                             fingerprint is stored — raw text is never written.
            analysis_result: A ``ThreatAnalysis`` (or any object with the same
                             public attributes).
            context:         Optional caller-supplied context dict (e.g. model
                             name, session ID).  Values are stored as-is; do
                             not put secrets in context.
        """
        if not self.enabled:
            return

        # Principle 12: store hash, not plaintext
        text_hash = hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()[:16]

        entry: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "text_hash": text_hash,
            "text_length": len(text),
            "is_safe": getattr(analysis_result, "is_safe", None),
            "threat_level": getattr(analysis_result, "threat_level", None),
            "threat_score": round(getattr(analysis_result, "threat_score", 0.0), 4),
            "recommended_action": getattr(analysis_result, "recommended_action", None),
            "confidence": round(getattr(analysis_result, "confidence", 0.0), 4),
            "analysis_time_ms": getattr(analysis_result, "analysis_time_ms", None),
            "threat_types": getattr(analysis_result, "threat_types", []),
            "timed_out": getattr(analysis_result, "metadata", {}).get("timed_out", False),
            "input_truncated": getattr(analysis_result, "metadata", {}).get(
                "input_truncated", False
            ),
            "context": context or {},
        }

        self._append(entry)

    def get_stats(self) -> Dict[str, Any]:
        """Return basic stats about this logger instance."""
        return {
            "enabled": self.enabled,
            "log_path": str(self.log_path),
            "records_written_this_session": self._records_written,
            "log_exists": self.log_path.exists(),
            "log_size_bytes": self.log_path.stat().st_size if self.log_path.exists() else 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_log_dir(self) -> None:
        """Create the log directory if it does not exist."""
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            logger.warning(
                "AuditLogger: could not create log directory %s: %s — "
                "audit logging disabled for this session.",
                self.log_path.parent,
                exc,
            )
            self.enabled = False

    def _append(self, entry: Dict[str, Any]) -> None:
        """Write one JSON record to the log file, appending atomically."""
        try:
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            with open(self.log_path, "a", encoding="utf-8") as fh:
                fh.write(line)
            self._records_written += 1
        except OSError as exc:
            logger.error(
                "AuditLogger: failed to write record: %s — "
                "continuing without audit log.",
                exc,
            )


# ---------------------------------------------------------------------------
# Module-level singleton (lazy, created on first access)
# ---------------------------------------------------------------------------

_default_logger: Optional[AuditLogger] = None


def get_default_logger(enabled: bool = True) -> AuditLogger:
    """
    Return (or create) the process-wide default ``AuditLogger``.

    The singleton uses ``~/.ethicore/guardian_audit.log`` and is shared
    across all ``Guardian`` instances in the same process.
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = AuditLogger(enabled=enabled)
    return _default_logger
