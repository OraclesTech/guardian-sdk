#!/usr/bin/env python3
"""
regenerate_embeddings.py
Ethicore Engine™ — Guardian SDK

Regenerates threat_embeddings.json from the complete semantic fingerprint set.

Works with both Community (5 categories) and Licensed (51 categories) editions:
  - Community: reads from ethicore_guardian.data.threat_patterns (default)
  - Licensed:  reads from threat_patterns_licensed.py via the 4-step resolution
               chain when ETHICORE_LICENSE_KEY + ETHICORE_ASSETS_DIR are set.

Run after adding new threat categories to keep SemanticAnalyzer in sync.

Principle 17 (Sanctified Continuous Improvement): keeps the embedding database
in sync with the ever-growing threat knowledge base.

Usage:
    # Community (writes to ethicore_guardian/data/threat_embeddings.json):
    python scripts/regenerate_embeddings.py

    # Licensed (reads/writes from ~/.ethicore by default):
    ETHICORE_LICENSE_KEY="EG-PRO-..." python scripts/regenerate_embeddings.py

    # Custom asset location:
    ETHICORE_LICENSE_KEY="EG-PRO-..." ETHICORE_ASSETS_DIR="/opt/ethicore" \\
        python scripts/regenerate_embeddings.py

    # Flags:
    python scripts/regenerate_embeddings.py --dry-run   # count only, no write
    python scripts/regenerate_embeddings.py --force     # overwrite without prompt
    python scripts/regenerate_embeddings.py --out /path/to/threat_embeddings.json
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import os
import pathlib
import sys

# ---------------------------------------------------------------------------
# Resolve project root so the package is importable from any working directory
# ---------------------------------------------------------------------------
_SCRIPT_DIR = pathlib.Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _resolve_output_path(
    explicit_out: str | None,
    assets_dir: str | None,
    license_key: str | None,
) -> pathlib.Path:
    """
    Determine where threat_embeddings.json should be written.

    Priority:
      1. Explicit --out argument
      2. <assets_dir>/data/              (licensed tier, explicit assets dir)
      3. ~/.ethicore/data/               (licensed tier, home dir convention)
      4. <package>/data/                 (community fallback)
    """
    if explicit_out:
        return pathlib.Path(explicit_out)

    if license_key:
        if assets_dir:
            return pathlib.Path(assets_dir) / "data" / "threat_embeddings.json"
        return pathlib.Path.home() / ".ethicore" / "data" / "threat_embeddings.json"

    # Community path — write next to community threat_patterns.py
    return _PROJECT_ROOT / "ethicore_guardian" / "data" / "threat_embeddings.json"


def _resolve_models_dir(assets_dir: str | None) -> str | None:
    """Resolve the ONNX models directory (used by SemanticAnalyzer).

    Returns None to let SemanticAnalyzer apply its own 4-step chain.
    """
    if assets_dir:
        candidate = pathlib.Path(assets_dir) / "models"
        if candidate.exists():
            return str(candidate)
    home_candidate = pathlib.Path.home() / ".ethicore" / "models"
    if home_candidate.exists():
        return str(home_candidate)
    return None  # SemanticAnalyzer will fall back to package models/


# ---------------------------------------------------------------------------
# Fingerprint stats loader (reads from the correct tier)
# ---------------------------------------------------------------------------

def _load_fingerprint_stats(
    license_key: str | None,
    assets_dir: str | None,
) -> tuple[int, int, str]:
    """Return (total_fingerprints, total_categories, edition).

    Loads from the licensed module when a key is supplied and the asset file
    is found; falls back to the community module otherwise.
    """
    if license_key:
        candidates = []
        if assets_dir:
            candidates.append(
                pathlib.Path(assets_dir) / "data" / "threat_patterns_licensed.py"
            )
        candidates.append(
            pathlib.Path.home() / ".ethicore" / "data" / "threat_patterns_licensed.py"
        )
        # Local dev: licensed/ directory alongside sdks/Python/
        candidates.append(
            _PROJECT_ROOT / "licensed" / "data" / "threat_patterns_licensed.py"
        )

        for path in candidates:
            if not path.exists():
                continue
            try:
                spec = importlib.util.spec_from_file_location(
                    "_tpl_licensed_stats", str(path)
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)  # type: ignore[union-attr]
                stats = mod.get_threat_statistics()
                return (
                    stats["totalSemanticFingerprints"],
                    stats["totalCategories"],
                    stats.get("edition", "licensed"),
                )
            except Exception as exc:
                print(f"[WARN]  Could not load {path}: {exc}", file=sys.stderr)

        print(
            "[WARN]  License key supplied but licensed asset file not found.\n"
            "        Tip: unzip ethicore-guardian-assets-pro.zip -d ~/.ethicore/\n"
            "        Falling back to community fingerprints.",
            file=sys.stderr,
        )

    # Community module
    from ethicore_guardian.data.threat_patterns import get_threat_statistics
    stats = get_threat_statistics()
    return (
        stats["totalSemanticFingerprints"],
        stats["totalCategories"],
        stats.get("edition", "community"),
    )


# ---------------------------------------------------------------------------
# Core regeneration logic
# ---------------------------------------------------------------------------

async def _regenerate(
    dry_run: bool,
    license_key: str | None,
    assets_dir: str | None,
    output_path: pathlib.Path,
) -> int:
    from ethicore_guardian.analyzers.semantic_analyzer import SemanticAnalyzer

    total_fingerprints, total_categories, edition = _load_fingerprint_stats(
        license_key, assets_dir
    )

    print("=" * 60)
    print("  Guardian SDK — Embedding Regeneration")
    print("=" * 60)
    print(f"  Edition:               {edition}")
    print(f"  Categories:            {total_categories}")
    print(f"  Semantic fingerprints: {total_fingerprints}")
    print(f"  Output path:           {output_path}")
    print("=" * 60)

    if dry_run:
        print("\n[dry-run] No files written.")
        return 0

    # Ensure the output directory exists before SemanticAnalyzer tries to write
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Delete the existing file so _ensure_threat_embeddings() regenerates it
    if output_path.exists():
        output_path.unlink()
        print(f"\n  Removed stale embeddings: {output_path}")

    # Resolve models dir (for ONNX MiniLM, if available)
    models_dir = _resolve_models_dir(assets_dir)

    print("\n  Initialising SemanticAnalyzer …")
    analyzer = SemanticAnalyzer(
        # Point data_dir at the parent of our output path so the analyzer
        # writes threat_embeddings.json to exactly where we want it.
        data_dir=str(output_path.parent),
        models_dir=models_dir,
        license_key=license_key,
        assets_dir=assets_dir,
    )
    success = await analyzer.initialize()

    if not success:
        print("\n[ERR]  SemanticAnalyzer initialisation failed.", file=sys.stderr)
        return 1

    count = len(analyzer.threat_embeddings)
    model_used = "ONNX MiniLM" if analyzer.session is not None else "fallback (hash-based)"

    print(f"\n  Embedding model:       {model_used}")
    print(f"  Embeddings generated:  {count}")
    print(f"  Written to:            {output_path}")

    if count < total_fingerprints:
        print(
            f"\n[WARN]  Generated {count} embeddings but expected {total_fingerprints}.\n"
            "        Some fingerprints may have failed.  "
            "Check the log output above for errors.",
            file=sys.stderr,
        )
    else:
        print(f"\n[OK]  All {count} fingerprints embedded successfully.")

    print(
        "\n  Done.  Commit threat_embeddings.json alongside your "
        "threat_patterns updates."
    )
    return 0


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate Guardian SDK threat embedding manifest.\n"
            "Works with both Community (5 categories) and "
            "Licensed (51 categories) editions."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not write threat_embeddings.json.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing embeddings without prompting.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help="Explicit output path for threat_embeddings.json (overrides auto-resolve).",
    )
    parser.add_argument(
        "--license-key",
        metavar="KEY",
        default=None,
        help=(
            "License key (overrides $ETHICORE_LICENSE_KEY env var). "
            "Enables licensed 51-category fingerprint set."
        ),
    )
    parser.add_argument(
        "--assets-dir",
        metavar="DIR",
        default=None,
        help=(
            "Path to extracted asset bundle (overrides $ETHICORE_ASSETS_DIR env var, "
            "then ~/.ethicore)."
        ),
    )
    args = parser.parse_args(argv)

    # Resolve credentials: CLI arg > env var
    license_key = args.license_key or os.environ.get("ETHICORE_LICENSE_KEY") or None
    assets_dir = args.assets_dir or os.environ.get("ETHICORE_ASSETS_DIR") or None

    # Trim whitespace so copy-paste from shell doesn't silently break validation
    if license_key:
        license_key = license_key.strip()
    if assets_dir:
        assets_dir = assets_dir.strip()

    output_path = _resolve_output_path(args.out, assets_dir, license_key)

    # Confirm overwrite unless --force or --dry-run
    if not args.dry_run and not args.force and output_path.exists():
        try:
            answer = input(
                f"\nthreat_embeddings.json already exists at:\n  {output_path}\n"
                "Overwrite? [y/N] "
            ).strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 0
        if answer not in ("y", "yes"):
            print("Aborted.")
            return 0

    return asyncio.run(
        _regenerate(
            dry_run=args.dry_run,
            license_key=license_key,
            assets_dir=assets_dir,
            output_path=output_path,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
