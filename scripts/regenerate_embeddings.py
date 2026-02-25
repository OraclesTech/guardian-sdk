#!/usr/bin/env python3
"""
regenerate_embeddings.py
Ethicore Engine™ - Guardian SDK

Regenerates ethicore_guardian/data/threat_embeddings.json from the complete
THREAT_PATTERNS semantic fingerprint set.  Run this script after adding new
threat categories so that SemanticAnalyzer loads embeddings for all categories.

Principle 17 (Sanctified Continuous Improvement): keeps the embedding set in
sync with the ever-growing threat knowledge base.

Usage:
    python scripts/regenerate_embeddings.py
    python scripts/regenerate_embeddings.py --dry-run   # count only, do not write
    python scripts/regenerate_embeddings.py --force     # overwrite without prompting
"""
from __future__ import annotations

import argparse
import asyncio
import pathlib
import sys

# ---------------------------------------------------------------------------
# Resolve paths relative to the project root
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_DATA_DIR = _PROJECT_ROOT / "ethicore_guardian" / "data"
_MODELS_DIR = _PROJECT_ROOT / "ethicore_guardian" / "models"
_EMBEDDINGS_PATH = _DATA_DIR / "threat_embeddings.json"

# Make sure the package is importable when run from any working directory
sys.path.insert(0, str(_PROJECT_ROOT))


async def _regenerate(dry_run: bool) -> int:
    from ethicore_guardian.analyzers.semantic_analyzer import SemanticAnalyzer
    from ethicore_guardian.data.threat_patterns import get_threat_statistics

    stats = get_threat_statistics()
    total_fingerprints = stats["totalSemanticFingerprints"]
    total_categories = stats["totalCategories"]

    print(
        f"Threat library: {total_categories} categories, "
        f"{total_fingerprints} semantic fingerprints"
    )

    if dry_run:
        print("[dry-run] No files written.")
        return 0

    # Delete the existing file so _ensure_threat_embeddings() regenerates it
    if _EMBEDDINGS_PATH.exists():
        _EMBEDDINGS_PATH.unlink()
        print(f"Removed stale embeddings: {_EMBEDDINGS_PATH.relative_to(_PROJECT_ROOT)}")

    # Initialise SemanticAnalyzer — this triggers full embedding generation
    print("Initialising SemanticAnalyzer and generating embeddings …")
    analyzer = SemanticAnalyzer(
        models_dir=str(_MODELS_DIR),
        data_dir=str(_DATA_DIR),
    )
    success = await analyzer.initialize()

    if not success:
        print("ERROR: SemanticAnalyzer initialisation failed.", file=sys.stderr)
        return 1

    count = len(analyzer.threat_embeddings)
    print(f"\n[OK] Generated {count} embeddings across {total_categories} categories")
    print(f"[OK] Written to {_EMBEDDINGS_PATH.relative_to(_PROJECT_ROOT)}")
    print("Done.  Commit this file alongside any threat_patterns.py updates.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate ONNX threat embedding manifest."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not modify threat_embeddings.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings without prompting.",
    )
    args = parser.parse_args(argv)

    if not args.dry_run and not args.force and _EMBEDDINGS_PATH.exists():
        answer = input(
            f"threat_embeddings.json already exists.  Overwrite? [y/N] "
        ).strip().lower()
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0

    return asyncio.run(_regenerate(dry_run=args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
