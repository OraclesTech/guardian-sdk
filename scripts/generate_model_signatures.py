#!/usr/bin/env python3
"""
generate_model_signatures.py
Ethicore Engine™ - Guardian SDK

Generates (or refreshes) ethicore_guardian/models/model_signatures.json with
SHA-256 integrity hashes for every ONNX model file.  Run this script once
after updating model files so that SemanticAnalyzer can verify them at load
time.

Principle 14 (Divine Safety): integrity verification before trusting ML output.

Usage:
    python scripts/generate_model_signatures.py

    # Force regeneration even if manifest already exists:
    python scripts/generate_model_signatures.py --force

The script is safe to re-run; it overwrites the manifest with fresh hashes.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys

# Resolve paths relative to the project root
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_MODELS_DIR = _PROJECT_ROOT / "ethicore_guardian" / "models"
_MANIFEST_PATH = _MODELS_DIR / "model_signatures.json"

# File patterns to include in the manifest
_INCLUDE_PATTERNS = ["*.onnx", "*.onnx.data"]


def _sha256(path: pathlib.Path, chunk_size: int = 1 << 20) -> str:
    """Compute SHA-256 of a file in streaming chunks to avoid loading it all at once."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate ONNX model signature manifest.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing manifest without prompting.",
    )
    args = parser.parse_args(argv)

    if not _MODELS_DIR.exists():
        print(f"ERROR: models directory not found: {_MODELS_DIR}", file=sys.stderr)
        return 1

    # Collect model files
    candidates: list[pathlib.Path] = []
    for pattern in _INCLUDE_PATTERNS:
        candidates.extend(sorted(_MODELS_DIR.glob(pattern)))

    if not candidates:
        print(f"No ONNX model files found in {_MODELS_DIR}", file=sys.stderr)
        return 1

    print(f"Found {len(candidates)} model file(s) in {_MODELS_DIR.relative_to(_PROJECT_ROOT)}")

    # Hash each file
    file_hashes: dict[str, str] = {}
    for model_path in candidates:
        rel = model_path.name
        print(f"  Hashing {rel} … ", end="", flush=True)
        sha = _sha256(model_path)
        file_hashes[rel] = sha
        print(f"{sha[:16]}…")

    # Build manifest
    manifest = {
        "_comment": (
            "SHA-256 integrity manifest for ONNX model files. "
            "Regenerate with: python scripts/generate_model_signatures.py"
        ),
        "_generated_at": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "_principle": (
            "Principle 14 (Divine Safety): verify model integrity before "
            "trusting inference output"
        ),
        "files": file_hashes,
    }

    # Write
    _MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"\nManifest written to {_MANIFEST_PATH.relative_to(_PROJECT_ROOT)}")
    print("Done.  Commit this file alongside any model updates.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
