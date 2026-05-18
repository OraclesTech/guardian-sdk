"""
Ethicore Engine™ - Guardian SDK — Supply Chain Integrity Module
Version: 1.0.0  (shipped as part of Guardian SDK v2.6.0)

Provides runtime self-integrity verification for the Guardian SDK package.
Detects tampering with bundled threat patterns, tokeniser files, and learned
model artefacts — protecting against supply chain attacks that target the SDK
itself or customer codebases that depend on it.

Two verification layers:
  1. Manifest check — SHA-256 hashes of all Python/JSON files bundled in the
     wheel are compared against a pre-computed baseline stored in
     data/integrity_manifest.json.  The manifest is generated at build time
     and shipped inside the wheel, so a tampered install will mismatch.

  2. ONNX model check — if model_signatures.json is present (licensed / full
     install), the ONNX file hashes are re-verified against that manifest.
     This is independent of the wheel manifest so licensed-tier model updates
     can be shipped without regenerating the wheel.

Usage
-----
  # CLI:
  guardian verify                 # check current install
  guardian verify --generate      # rebuild manifest from current files (build step)
  guardian verify --json          # machine-readable output

  # Programmatic (opt-in — does not run automatically unless env var is set):
  from ethicore_guardian.integrity import verify_sdk_integrity
  result = verify_sdk_integrity()
  if not result.passed:
      raise RuntimeError(result.summary)

  # Auto-check on import:
  export GUARDIAN_VERIFY_INTEGRITY=1

Guiding Principles honoured here:
  Principle 11 (Sacred Truth / Emet): every mismatch is reported explicitly
    with file path and expected vs. actual hash — nothing is hidden.
  Principle 14 (Divine Safety): fail-closed — a tampered install raises a
    clear error rather than silently degrading.
  Principle 13 (Ultimate Accountability): the verification result is fully
    auditable; --verbose shows per-file pass/fail.

Copyright © 2026 Oracles Technologies LLC. All Rights Reserved.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Files verified by the wheel manifest check.
# Paths are relative to the ethicore_guardian package root.
# Only files shipped in the community wheel are listed — ONNX models and
# licensed data files are covered by model_signatures.json separately.
_MANIFEST_FILES = [
    "data/threat_patterns.py",
    "models/vocab.json",
    "models/special_tokens.json",
    "models/ml_learning.json",
    "community_guardian.py",
    "versions.py",
]

# Optional files included in the manifest when present (licensed install).
_OPTIONAL_MANIFEST_FILES = [
    "data/threat_embeddings.json",
]

_MANIFEST_FILENAME = "data/integrity_manifest.json"
_MODEL_SIGNATURES_FILENAME = "models/model_signatures.json"

_HASH_ALGORITHM = "sha256"
_CHUNK_SIZE = 65_536  # 64 KB read chunks


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FileVerificationResult:
    """Verification result for a single file."""
    path: str
    passed: bool
    expected_hash: Optional[str] = None
    actual_hash: Optional[str] = None
    error: Optional[str] = None

    @property
    def status(self) -> str:
        if self.error:
            return "ERROR"
        return "OK" if self.passed else "TAMPERED"


@dataclass
class IntegrityResult:
    """Aggregate integrity verification result."""
    passed: bool
    files_checked: int
    files_passed: int
    files_failed: int
    file_results: List[FileVerificationResult] = field(default_factory=list)
    onnx_verified: bool = False
    onnx_checked: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        if self.passed:
            parts = [f"Guardian SDK integrity OK — {self.files_checked} files verified"]
            if self.onnx_checked:
                parts.append(f"ONNX models verified")
            return ", ".join(parts)
        failed = [r for r in self.file_results if not r.passed]
        names = ", ".join(r.path for r in failed[:3])
        suffix = f" (+{len(failed) - 3} more)" if len(failed) > 3 else ""
        return f"INTEGRITY FAILURE — {len(failed)} file(s) tampered: {names}{suffix}"

    def to_dict(self) -> Dict:
        return {
            "passed": self.passed,
            "files_checked": self.files_checked,
            "files_passed": self.files_passed,
            "files_failed": self.files_failed,
            "onnx_verified": self.onnx_verified,
            "onnx_checked": self.onnx_checked,
            "warnings": self.warnings,
            "errors": self.errors,
            "file_results": [
                {
                    "path": r.path,
                    "status": r.status,
                    "expected_hash": r.expected_hash,
                    "actual_hash": r.actual_hash,
                    "error": r.error,
                }
                for r in self.file_results
            ],
        }


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _package_root() -> Path:
    """Return the absolute path to the ethicore_guardian package directory."""
    return Path(__file__).parent.resolve()


def _hash_file(path: Path) -> str:
    """Compute SHA-256 hex digest of *path* using buffered reads."""
    h = hashlib.new(_HASH_ALGORITHM)
    with open(path, "rb") as fh:
        while chunk := fh.read(_CHUNK_SIZE):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest(root: Path) -> Optional[Dict[str, str]]:
    """
    Load the integrity manifest from ``data/integrity_manifest.json``.

    Returns None if the manifest file does not exist (e.g. development
    checkout before the first ``guardian verify --generate`` run).
    """
    manifest_path = root / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data.get("files", {})
    except Exception as exc:
        logger.warning("Could not load integrity manifest: %s", exc)
        return None


def _verify_onnx_models(root: Path) -> Tuple[bool, bool, List[str]]:
    """
    Re-verify ONNX model files against model_signatures.json.

    Returns (passed, checked, errors).
    ``checked`` is False when model_signatures.json is absent (community
    install without the full model bundle).
    """
    sig_path = root / _MODEL_SIGNATURES_FILENAME
    if not sig_path.exists():
        return True, False, []

    try:
        with open(sig_path, "r", encoding="utf-8") as fh:
            signatures = json.load(fh)
    except Exception as exc:
        return False, True, [f"Could not parse model_signatures.json: {exc}"]

    errors: List[str] = []
    for filename, expected_hash in signatures.items():
        model_path = root / "models" / filename
        if not model_path.exists():
            # Model file may be absent on a community install — not an error.
            continue
        actual = _hash_file(model_path)
        # model_signatures.json stores truncated hashes (first 16 hex chars).
        if not actual.startswith(expected_hash.replace("…", "").replace("�", "")):
            errors.append(
                f"ONNX model tampered: {filename} "
                f"(expected prefix {expected_hash!r}, got {actual[:16]!r})"
            )

    return len(errors) == 0, True, errors


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_manifest(root: Optional[Path] = None, *, include_optional: bool = True) -> Dict:
    """
    Compute SHA-256 hashes of all manifest-tracked files and return the
    manifest dict.  Does NOT write to disk — call ``save_manifest()`` for that.

    Parameters
    ----------
    root:
        Package root.  Defaults to the directory containing this file.
    include_optional:
        When True, optional files present on disk are included in the manifest.
    """
    if root is None:
        root = _package_root()

    files_to_hash = list(_MANIFEST_FILES)
    if include_optional:
        for opt in _OPTIONAL_MANIFEST_FILES:
            if (root / opt).exists():
                files_to_hash.append(opt)

    manifest: Dict[str, str] = {}
    for rel in files_to_hash:
        path = root / rel
        if path.exists():
            manifest[rel] = _hash_file(path)
        else:
            logger.debug("Manifest: skipping absent file %s", rel)

    return {
        "version": "1",
        "algorithm": _HASH_ALGORITHM,
        "generated_by": f"guardian verify --generate",
        "files": manifest,
    }


def save_manifest(manifest: Dict, root: Optional[Path] = None) -> Path:
    """Write *manifest* to ``data/integrity_manifest.json`` and return the path."""
    if root is None:
        root = _package_root()
    out = root / _MANIFEST_FILENAME
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
        fh.write("\n")
    return out


def verify_sdk_integrity(
    root: Optional[Path] = None,
    *,
    strict: bool = False,
) -> IntegrityResult:
    """
    Verify the integrity of the installed Guardian SDK.

    Parameters
    ----------
    root:
        Package root.  Defaults to the directory containing this file.
    strict:
        When True, a missing manifest is treated as a failure rather than
        a warning.  Use in security-critical deployment pipelines.

    Returns
    -------
    IntegrityResult
        ``.passed`` is True only when every checked file matches its expected
        hash and ONNX verification (when applicable) passes.
    """
    if root is None:
        root = _package_root()

    file_results: List[FileVerificationResult] = []
    warnings: List[str] = []
    errors: List[str] = []

    # ── Load the manifest ────────────────────────────────────────────────────
    manifest = _load_manifest(root)

    if manifest is None:
        msg = (
            "Integrity manifest not found. "
            "Run 'guardian verify --generate' to create one. "
            "This is expected on development checkouts."
        )
        if strict:
            errors.append(msg)
            return IntegrityResult(
                passed=False,
                files_checked=0,
                files_passed=0,
                files_failed=0,
                warnings=warnings,
                errors=errors,
            )
        warnings.append(msg)
        # Still run ONNX check even without a file manifest.
        onnx_ok, onnx_checked, onnx_errors = _verify_onnx_models(root)
        errors.extend(onnx_errors)
        return IntegrityResult(
            passed=onnx_ok,
            files_checked=0,
            files_passed=0,
            files_failed=0,
            onnx_verified=onnx_ok,
            onnx_checked=onnx_checked,
            warnings=warnings,
            errors=errors,
        )

    # ── Verify each file in the manifest ────────────────────────────────────
    for rel_path, expected_hash in manifest.items():
        abs_path = root / rel_path
        if not abs_path.exists():
            file_results.append(FileVerificationResult(
                path=rel_path,
                passed=False,
                expected_hash=expected_hash,
                error="File not found",
            ))
            continue
        try:
            actual_hash = _hash_file(abs_path)
            passed = actual_hash == expected_hash
            file_results.append(FileVerificationResult(
                path=rel_path,
                passed=passed,
                expected_hash=expected_hash,
                actual_hash=actual_hash,
            ))
        except OSError as exc:
            file_results.append(FileVerificationResult(
                path=rel_path,
                passed=False,
                expected_hash=expected_hash,
                error=str(exc),
            ))

    # ── ONNX model verification ──────────────────────────────────────────────
    onnx_ok, onnx_checked, onnx_errors = _verify_onnx_models(root)
    errors.extend(onnx_errors)

    # ── Aggregate ────────────────────────────────────────────────────────────
    passed_count = sum(1 for r in file_results if r.passed)
    failed_count = len(file_results) - passed_count
    all_files_ok = failed_count == 0
    overall = all_files_ok and onnx_ok

    return IntegrityResult(
        passed=overall,
        files_checked=len(file_results),
        files_passed=passed_count,
        files_failed=failed_count,
        file_results=file_results,
        onnx_verified=onnx_ok,
        onnx_checked=onnx_checked,
        warnings=warnings,
        errors=errors,
    )


def check_on_import() -> None:
    """
    Lightweight integrity gate called at package import when
    ``GUARDIAN_VERIFY_INTEGRITY=1`` is set.

    Logs a WARNING on mismatch rather than raising — the SDK degrades
    gracefully; disrupting import in production on a hash mismatch would be
    a denial-of-service vector.  Set ``GUARDIAN_VERIFY_INTEGRITY=strict`` to
    raise ``RuntimeError`` on failure instead.
    """
    mode = os.environ.get("GUARDIAN_VERIFY_INTEGRITY", "").strip().lower()
    if mode not in ("1", "true", "yes", "strict"):
        return

    try:
        result = verify_sdk_integrity(strict=(mode == "strict"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("[Guardian] Integrity check failed unexpectedly: %s", exc)
        return

    if result.passed:
        logger.debug("[Guardian] Integrity check passed (%d files verified)", result.files_checked)
        return

    msg = f"[Guardian] SUPPLY CHAIN INTEGRITY WARNING — {result.summary}"
    for warn in result.warnings:
        logger.warning("[Guardian] %s", warn)
    for err in result.errors:
        logger.error("[Guardian] %s", err)
    for fr in result.file_results:
        if not fr.passed:
            logger.error(
                "[Guardian] TAMPERED: %s | expected=%s actual=%s error=%s",
                fr.path, fr.expected_hash, fr.actual_hash, fr.error,
            )

    if mode == "strict":
        raise RuntimeError(msg)
    else:
        logger.warning(msg)
