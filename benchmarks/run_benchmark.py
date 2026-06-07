#!/usr/bin/env python3
"""
Ethicore Engine™ — Guardian SDK Benchmark Suite
================================================

Evaluates Guardian SDK against public prompt injection / jailbreak datasets
and produces human-readable Markdown + machine-parseable JSON reports.

Usage:
    # Quick run (bundled data only, ~2 min, no internet required)
    python benchmarks/run_benchmark.py --quick

    # Full run (all HuggingFace datasets, ~30 min)
    python benchmarks/run_benchmark.py --full

    # Single dataset
    python benchmarks/run_benchmark.py --dataset neuralchemy

    # Licensed tier (requires ETHICORE_API_KEY + ETHICORE_ASSETS_DIR)
    python benchmarks/run_benchmark.py --full --tier licensed

    # Output to custom directory
    python benchmarks/run_benchmark.py --quick --output /tmp/results
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

# Ensure the repo root is on sys.path so we can import ethicore_guardian
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "sdks" / "Python"))
sys.path.insert(0, str(REPO_ROOT))

# Also try the project root directly (when run from project root)
sys.path.insert(0, str(Path(__file__).parent.parent))

# Auto-load .env from project root if python-dotenv is available.
# This surfaces ETHICORE_API_KEY and ETHICORE_ASSETS_DIR for licensed runs
# without requiring manual `source .env` before the benchmark.
try:
    from dotenv import load_dotenv  # noqa: PLC0415
    _env_file = REPO_ROOT / ".env"
    if _env_file.exists():
        load_dotenv(_env_file, override=False)   # override=False: shell env takes precedence
except ImportError:
    pass  # dotenv optional — env vars can be set manually

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("guardian.benchmark")

# ---------------------------------------------------------------------------
# Benchmarks package imports (relative to this file's directory)
# ---------------------------------------------------------------------------
BENCH_DIR = Path(__file__).parent
sys.path.insert(0, str(BENCH_DIR.parent))

from benchmarks.datasets.loader import DatasetRegistry, BenchmarkSample  # noqa: E402
from benchmarks.runners.guardian_runner import GuardianRunner, SCORE_THRESHOLD  # noqa: E402
from benchmarks.runners.baseline_runner import get_baselines               # noqa: E402
from benchmarks.metrics.classification import compute_metrics              # noqa: E402
from benchmarks.metrics.latency import compute_latency_metrics             # noqa: E402
from benchmarks.reports.markdown_reporter import MarkdownReporter, build_run_data  # noqa: E402
from benchmarks.reports.json_reporter import JSONReporter                  # noqa: E402


# ---------------------------------------------------------------------------
# Dataset configurations
# ---------------------------------------------------------------------------

QUICK_DATASETS = ["bundled", "agentic"]      # offline, no HF download
FULL_DATASETS = [
    "bundled",
    "agentic",     # agentic-AI surface: tool-calls, exec plans, carrier-framing, Gaps 77–83
    "neuralchemy",
    "deepset",
    "jailbreakbench",
    "advbench",
    "harmbench",
    "b3",          # Lakera Backbone Breaker Benchmark — agentic attack coverage
    "agentdojo",   # ETH/Princeton — agent tool-call injections (NeurIPS 2024)
    "injecagent",  # UIUC/Tsinghua/Stanford — indirect injection via tool outputs (ACL 2024)
    "agentharm",   # UK AISI/Anthropic — harmful agent tasks (2024; may be HF-gated)
    "salad_data",  # SALAD-Bench — broad LLM safety (ACL 2024)
]

# For attack-only datasets (all label=1), we supplement with
# benign samples from the bundled corpus so FPR can be computed.
ATTACK_ONLY_DATASETS = {"jailbreakbench", "advbench", "harmbench", "b3", "salad_data"}


def _load_samples(
    dataset_name: str,
    sample_limit: int = 0,
    benign_supplement: Optional[List[BenchmarkSample]] = None,
) -> List[BenchmarkSample]:
    """Load samples for a dataset, supplementing benign if attack-only."""
    loader = DatasetRegistry.get(dataset_name)
    samples = loader.load(sample_limit=sample_limit)

    if dataset_name in ATTACK_ONLY_DATASETS and benign_supplement:
        # Add up to len(attacks) benign samples to enable FPR computation
        n_attacks = len(samples)
        supplement = benign_supplement[:n_attacks]
        logger.info(
            "%s: supplementing with %d benign samples for FPR computation",
            dataset_name, len(supplement),
        )
        samples = samples + supplement

    return samples


async def _run_one_dataset(
    runner: GuardianRunner,
    dataset_name: str,
    sample_limit: int,
    benign_supplement: Optional[List[BenchmarkSample]],
    samples: Optional[List[BenchmarkSample]] = None,
    pass_label: str = "",
) -> Tuple:
    """
    Run the benchmark for one dataset.

    Args:
        runner:           GuardianRunner instance (reused across passes).
        dataset_name:     Name of the dataset to evaluate.
        sample_limit:     Max samples to load (0 = no limit).
        benign_supplement: Benign samples to add to attack-only datasets.
        samples:          If provided, reuse these samples instead of reloading
                          (used for Pass 2 so the exact same inputs are evaluated).
        pass_label:       Optional label shown in logs ("Pass 1", "Pass 2", etc.).

    Returns:
        (ClassificationMetrics, LatencyMetrics, samples, predictions)
        The samples and predictions are returned so callers can feed
        misclassifications into the AdversarialLearner between passes.
    """
    label = f"[{pass_label}] " if pass_label else ""
    logger.info("─" * 60)
    logger.info("%sDataset: %s", label, dataset_name.upper())

    if samples is None:
        samples = _load_samples(dataset_name, sample_limit, benign_supplement)
    if not samples:
        logger.warning("No samples loaded for %s — skipping", dataset_name)
        return None, None, None, None

    n_attacks = sum(1 for s in samples if s.label == 1)
    n_benign = len(samples) - n_attacks
    logger.info(
        "%sLoaded %d samples (%d attacks, %d benign)",
        label, len(samples), n_attacks, n_benign,
    )

    # Run Guardian on all samples
    t_start = time.perf_counter()
    result = await runner.run_dataset(samples, show_progress=True)
    elapsed = time.perf_counter() - t_start
    logger.info(
        "%sInference complete in %.1fs (%.0f samples/sec)",
        label, elapsed, len(samples) / elapsed,
    )

    # Ground truth
    y_true = [s.label for s in samples]
    y_pred = result.y_pred
    y_scores = result.y_scores

    # Classification metrics
    clf_metrics = compute_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_scores=y_scores,
        dataset=dataset_name,
        runner=runner.name,
        threshold=SCORE_THRESHOLD,
    )

    # Latency metrics
    lat_metrics = compute_latency_metrics(result.latencies_ms)

    return clf_metrics, lat_metrics, samples, result.predictions


async def _run_learning_pass(
    runner: GuardianRunner,
    all_samples: List[List[BenchmarkSample]],
    all_predictions: List[List],
) -> dict:
    """
    Feed all false-negative misclassifications from Pass 1 into the
    AdversarialLearner so Pass 2 benefits from closed-loop learning.

    Args:
        runner:          The same GuardianRunner used for Pass 1 (shares Guardian instance).
        all_samples:     List of sample lists, one per dataset.
        all_predictions: List of prediction lists, one per dataset (same order).

    Returns:
        Aggregate learning stats dict.
    """
    # Flatten all samples and predictions across datasets
    flat_samples = [s for ds in all_samples for s in ds]
    flat_preds = [p for dp in all_predictions for p in dp]

    total_fn = sum(
        1 for s, p in zip(flat_samples, flat_preds)
        if s.label == 1 and p.predicted_label == 0
    )
    total_fp = sum(
        1 for s, p in zip(flat_samples, flat_preds)
        if s.label == 0 and p.predicted_label == 1
    )

    logger.info("=" * 60)
    logger.info("LEARNING PASS — AdversarialLearner closed-loop training")
    logger.info("  False Negatives (missed attacks) to learn: %d", total_fn)
    logger.info("  False Positives (benign over-flagged):     %d", total_fp)
    logger.info("=" * 60)

    if total_fn == 0:
        logger.info("No false negatives found — nothing to learn from. "
                    "Guardian already catches all attacks in Pass 1.")
        return {"fn_submitted": 0, "fn_deduplicated": 0,
                "fn_cap_reached": 0, "fn_errors": 0, "total_attempted": 0}

    stats = await runner.submit_corrections(flat_samples, flat_preds)

    logger.info("Learning results:")
    logger.info("  Fingerprints added (unique FNs learned): %d", stats["fn_submitted"])
    logger.info("  Deduplicated (already known):            %d", stats["fn_deduplicated"])
    logger.info("  Cap reached (increase max_fingerprints): %d", stats["fn_cap_reached"])
    logger.info("  Errors:                                  %d", stats["fn_errors"])

    return stats


def preload_datasets(
    datasets: List[str],
    sample_limit: int,
) -> dict:
    """
    Synchronously load all datasets BEFORE asyncio.run() is called.

    HuggingFace's fingerprint hasher (used for caching) calls dill.pickle on
    its data_files config.  Once asyncio.run() starts the event loop, Python
    creates internal threading.RLock objects that dill cannot serialize across
    process boundaries:
        "RLock objects should only be shared between processes through inheritance"

    Loading datasets here — before the event loop exists — avoids the issue
    entirely.  Cached datasets return instantly; only first-time downloads are
    slow (and must complete before inference begins anyway).

    Returns:
        dict mapping dataset_name → List[BenchmarkSample] (empty list on failure).
    """
    # Pre-load bundled benign samples first (needed for supplementing
    # attack-only datasets like jailbreakbench, advbench, harmbench).
    benign_supplement: List[BenchmarkSample] = []
    if any(d in ATTACK_ONLY_DATASETS for d in datasets):
        try:
            bundled_loader = DatasetRegistry.get("bundled")
            all_bundled = bundled_loader.load()
            benign_supplement = [s for s in all_bundled if s.label == 0]
            logger.info(
                "Pre-load: %d bundled benign samples ready for supplementation",
                len(benign_supplement),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pre-load: could not load bundled benign supplement: %s", exc)

    logger.info("Pre-loading all datasets before Guardian/asyncio initialization…")
    preloaded: dict = {}
    for dataset_name in datasets:
        try:
            samples = _load_samples(dataset_name, sample_limit, benign_supplement)
            preloaded[dataset_name] = samples
            logger.info("  ✓ %s: %d samples", dataset_name, len(samples))
        except Exception as exc:  # noqa: BLE001
            logger.error("  ✗ %s: pre-load failed — %s", dataset_name, exc)
            preloaded[dataset_name] = []   # empty sentinel → skipped in inference loop

    # Stash benign supplement so run_benchmark() can retrieve it from the dict.
    preloaded["_benign_supplement"] = benign_supplement

    usable = [d for d in datasets if d != "_benign_supplement" and preloaded.get(d)]
    logger.info(
        "Pre-load complete: %d/%d datasets ready (%s)",
        len(usable), len(datasets), ", ".join(usable) or "none",
    )
    return preloaded


async def run_benchmark(
    datasets: List[str],
    tier: str,
    sample_limit: int,
    output_dir: Path,
    preloaded_samples: Optional[dict] = None,
    license_key: Optional[str] = None,
    assets_dir: Optional[str] = None,
    enable_learning: bool = False,
) -> dict:
    """
    Main async benchmark orchestrator.

    When enable_learning=True, runs two passes:
      Pass 1 — standard evaluation, collecting misclassifications.
      Learning — feeds false negatives into AdversarialLearner (in-memory).
      Pass 2 — re-evaluates same samples; improvements from learning are visible.

    Returns the run_data dict for Pass 1 (or Pass 2 if learning enabled).
    """
    logger.info("=" * 60)
    logger.info("Ethicore Engine™ — Guardian SDK Benchmark Suite")
    logger.info("Tier: %s | Datasets: %s", tier, ", ".join(datasets))
    if enable_learning:
        logger.info("Mode: TWO-PASS (AdversarialLearner closed-loop training)")
    logger.info("=" * 60)

    # Determine Guardian SDK version
    try:
        import ethicore_guardian  # noqa: PLC0415
        guardian_version = ethicore_guardian.__version__
    except ImportError:
        guardian_version = "unknown"
        logger.warning("ethicore_guardian not importable — version unknown")

    # Initialize runner (single instance shared across both passes so the
    # Guardian's in-memory fingerprints carry over from learning → Pass 2).
    runner = GuardianRunner(
        license_key=license_key,
        assets_dir=assets_dir,
        tier=tier,
        concurrency=20,
    )

    # ── Resolve pre-loaded samples ───────────────────────────────────────────
    # Datasets are loaded synchronously in main() via preload_datasets()
    # *before* asyncio.run() so HuggingFace's dill-based fingerprint hasher
    # never encounters asyncio RLock objects.  If preloaded_samples was not
    # provided (e.g. programmatic call), fall back to loading here (risks the
    # RLock issue only if HF datasets are not already cached).
    if preloaded_samples is None:
        logger.warning(
            "preloaded_samples not provided — loading datasets inside asyncio "
            "(may fail for uncached HuggingFace datasets due to RLock pickling). "
            "Call preload_datasets() before asyncio.run() to avoid this."
        )
        preloaded_samples = {}
        benign_supplement: List[BenchmarkSample] = []
        if any(d in ATTACK_ONLY_DATASETS for d in datasets):
            try:
                bundled_loader = DatasetRegistry.get("bundled")
                all_bundled = bundled_loader.load()
                benign_supplement = [s for s in all_bundled if s.label == 0]
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not load bundled benign supplement: %s", exc)
        for dataset_name in datasets:
            try:
                preloaded_samples[dataset_name] = _load_samples(
                    dataset_name, sample_limit, benign_supplement
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Dataset %s load failed: %s", dataset_name, exc)
                preloaded_samples[dataset_name] = []
    else:
        benign_supplement = preloaded_samples.get("_benign_supplement", [])

    # ── Pass 1 ──────────────────────────────────────────────────────────────
    if enable_learning:
        logger.info("━" * 60)
        logger.info("PASS 1 — Baseline evaluation")
        logger.info("━" * 60)

    classification_results_p1 = []
    latency_results_p1 = []
    # For learning: track (dataset_name, samples, predictions) triples so
    # the Pass 2 loop stays correctly aligned even if some datasets fail.
    pass1_dataset_records: List[tuple] = []  # (dataset_name, samples, predictions)

    for dataset_name in datasets:
        preloaded = preloaded_samples.get(dataset_name)
        if not preloaded:
            continue   # already logged during pre-load
        try:
            clf, lat, samples, predictions = await _run_one_dataset(
                runner=runner,
                dataset_name=dataset_name,
                sample_limit=sample_limit,
                benign_supplement=benign_supplement,
                samples=preloaded,    # use pre-loaded samples — no HF call here
                pass_label="Pass 1" if enable_learning else "",
            )
            if clf is not None:
                classification_results_p1.append(clf)
                latency_results_p1.append(lat)
                if enable_learning and samples and predictions:
                    pass1_dataset_records.append((dataset_name, samples, predictions))
        except Exception as exc:  # noqa: BLE001
            logger.error("Dataset %s failed: %s", dataset_name, exc, exc_info=True)

    if not classification_results_p1:
        logger.error("No results produced — check dataset availability and Guardian installation")
        return {}

    run_data_p1 = build_run_data(
        classification_results=classification_results_p1,
        latency_results=latency_results_p1,
        guardian_version=guardian_version,
        tier=tier,
    )

    # Write Pass 1 reports
    pass1_suffix = "_pass1" if enable_learning else ""
    md_reporter = MarkdownReporter(output_dir=output_dir)
    json_reporter = JSONReporter(output_dir=output_dir)

    md_path_p1 = md_reporter.write(run_data_p1, filename_suffix=pass1_suffix)
    json_path_p1 = json_reporter.write(run_data_p1, baselines=get_baselines(),
                                       filename_suffix=pass1_suffix)

    _log_summary(run_data_p1, thresholds=json_reporter.check_thresholds(
        run_data_p1.get("summary", {})),
        label="PASS 1 RESULTS" if enable_learning else "RESULTS SUMMARY",
    )
    logger.info("  Markdown : %s", md_path_p1)
    logger.info("  JSON     : %s", json_path_p1)
    logger.info("=" * 60)

    # ── Learning pass ────────────────────────────────────────────────────────
    if not enable_learning:
        return run_data_p1

    # Only feed FNs from cl_safe datasets into the AdversarialLearner.
    # Datasets with unreliable benign labels (neuralchemy) or attack-only
    # datasets whose semantics differ from injection text (JBB, HarmBench,
    # B3, Salad-Data) are excluded to prevent contaminating the fingerprint
    # space and inflating FPR in Pass 2 on unrelated content.
    cl_safe_records = [
        r for r in pass1_dataset_records
        if DatasetRegistry._registry.get(r[0]) is not None
        and DatasetRegistry._registry[r[0]].cl_safe
    ]
    excluded = [r[0] for r in pass1_dataset_records if r not in cl_safe_records]
    if excluded:
        logger.info(
            "Learning pass: excluding %d dataset(s) not marked cl_safe: %s",
            len(excluded), ", ".join(excluded),
        )
    if not cl_safe_records:
        logger.warning(
            "Learning pass: no cl_safe datasets in this run — "
            "skipping AdversarialLearner training."
        )
        learning_stats = {"fn_submitted": 0, "fn_deduplicated": 0,
                          "fn_cap_reached": 0, "fn_errors": 0, "total_attempted": 0}
    else:
        all_pass1_samples = [r[1] for r in cl_safe_records]
        all_pass1_predictions = [r[2] for r in cl_safe_records]
        learning_stats = await _run_learning_pass(
            runner=runner,
            all_samples=all_pass1_samples,
            all_predictions=all_pass1_predictions,
        )

    # ── Pass 2 ──────────────────────────────────────────────────────────────
    logger.info("━" * 60)
    logger.info("PASS 2 — Post-learning evaluation (same samples, updated fingerprints)")
    logger.info("━" * 60)

    classification_results_p2 = []
    latency_results_p2 = []

    # Use the tracked records to ensure dataset names stay aligned with samples.
    for dataset_name, p1_samples, _ in pass1_dataset_records:
        try:
            clf, lat, _, _ = await _run_one_dataset(
                runner=runner,
                dataset_name=dataset_name,
                sample_limit=sample_limit,
                benign_supplement=benign_supplement,
                samples=p1_samples,        # reuse exact same samples
                pass_label="Pass 2",
            )
            if clf is not None:
                classification_results_p2.append(clf)
                latency_results_p2.append(lat)
        except Exception as exc:  # noqa: BLE001
            logger.error("Pass 2 dataset %s failed: %s", dataset_name, exc, exc_info=True)

    if classification_results_p2:
        run_data_p2 = build_run_data(
            classification_results=classification_results_p2,
            latency_results=latency_results_p2,
            guardian_version=guardian_version,
            tier=tier,
        )
        md_path_p2 = md_reporter.write(run_data_p2, filename_suffix="_pass2")
        json_path_p2 = json_reporter.write(run_data_p2, baselines=get_baselines(),
                                           filename_suffix="_pass2")

        _log_summary(run_data_p2,
                     thresholds=json_reporter.check_thresholds(run_data_p2.get("summary", {})),
                     label="PASS 2 RESULTS (after learning)")
        logger.info("  Markdown : %s", md_path_p2)
        logger.info("  JSON     : %s", json_path_p2)

        # Side-by-side delta log
        _log_learning_delta(run_data_p1, run_data_p2, learning_stats)

        return run_data_p2  # Return Pass 2 as primary result

    return run_data_p1


def _log_summary(run_data: dict, thresholds: dict, label: str = "RESULTS SUMMARY") -> None:
    """Log a standard metrics summary block."""
    logger.info("=" * 60)
    logger.info(label)
    logger.info("=" * 60)
    summary = run_data.get("summary", {})
    logger.info("  Precision : %.3f", summary.get("precision") or 0)
    logger.info("  Recall    : %.3f", summary.get("recall") or 0)
    logger.info("  F1        : %.3f", summary.get("f1") or 0)
    logger.info("  ROC-AUC   : %.3f", summary.get("roc_auc") or 0)
    logger.info("  FPR       : %.3f", summary.get("fpr") or 0)
    logger.info("  FNR       : %.3f", summary.get("fnr") or 0)
    logger.info("  p50 lat   : %.1fms", summary.get("p50_ms") or 0)
    logger.info("  p99 lat   : %.1fms", summary.get("p99_ms") or 0)
    logger.info("")
    passed = all(thresholds.values())
    logger.info("CI thresholds: %s", "✅ PASS" if passed else "❌ FAIL")
    for metric, ok in thresholds.items():
        logger.info("  %s %s", "✅" if ok else "❌", metric)
    logger.info("")


def _log_learning_delta(run_data_p1: dict, run_data_p2: dict,
                         learning_stats: dict) -> None:
    """Log a before/after comparison table after the learning pass."""
    s1 = run_data_p1.get("summary", {})
    s2 = run_data_p2.get("summary", {})

    def delta(key: str, invert: bool = False) -> str:
        v1 = s1.get(key) or 0.0
        v2 = s2.get(key) or 0.0
        d = v2 - v1
        if invert:
            d = -d
        better = "up" if d > 0.0001 else ("dn" if d < -0.0001 else " =")
        return f"{d:+.3f} {better}"

    logger.info("=" * 60)
    logger.info("LEARNING DELTA (Pass 2 vs Pass 1)")
    logger.info("  FNs learned into fingerprints: %d", learning_stats.get("fn_submitted", 0))
    logger.info("  FNs deduplicated:              %d", learning_stats.get("fn_deduplicated", 0))
    logger.info("")
    logger.info("  %-12s  %8s  %8s  %10s", "Metric", "Pass 1", "Pass 2", "Delta")
    logger.info("  %s", "-" * 44)
    for key, label, invert in [
        ("precision",  "Precision", False),
        ("recall",     "Recall",    False),
        ("f1",         "F1",        False),
        ("roc_auc",    "ROC-AUC",  False),
        ("fpr",        "FPR",       True),   # lower is better
        ("fnr",        "FNR",       True),   # lower is better
    ]:
        v1 = s1.get(key) or 0.0
        v2 = s2.get(key) or 0.0
        logger.info("  %-12s  %8.3f  %8.3f  %s", label, v1, v2, delta(key, invert))
    logger.info("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ethicore Engine™ Guardian SDK Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Run mode (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick",
        action="store_true",
        help="Quick run: bundled data only, ~2 min, no internet required",
    )
    mode_group.add_argument(
        "--full",
        action="store_true",
        help="Full run: all HuggingFace datasets, ~30 min",
    )
    mode_group.add_argument(
        "--dataset",
        metavar="NAME",
        help=f"Single dataset. Available: {', '.join(DatasetRegistry.all_names())}",
    )

    # Tier
    parser.add_argument(
        "--tier",
        choices=["community", "licensed", "auto"],
        default="auto",
        help="Guardian tier. 'auto' uses licensed if ETHICORE_API_KEY is set (default: auto)",
    )

    # Sample limit
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Max samples per dataset (0 = no limit; for --quick defaults to 200)",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCH_DIR / "results",
        metavar="DIR",
        help="Output directory for reports (default: benchmarks/results/)",
    )

    # Two-pass learning mode
    parser.add_argument(
        "--learn",
        action="store_true",
        help=(
            "Enable two-pass benchmark with AdversarialLearner closed-loop training. "
            "Pass 1 runs normally; false negatives are fed into the AdversarialLearner "
            "in memory; Pass 2 re-evaluates the same samples with updated fingerprints. "
            "Produces _pass1 and _pass2 reports plus a delta comparison log."
        ),
    )

    # Verbosity
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Determine datasets to run
    if args.quick:
        datasets = QUICK_DATASETS
        sample_limit = args.limit or 200
        logger.info("Mode: QUICK (bundled data, limit=%d)", sample_limit)
    elif args.full:
        datasets = FULL_DATASETS
        sample_limit = args.limit
        logger.info("Mode: FULL (all datasets)")
    elif args.dataset:
        if args.dataset not in DatasetRegistry.all_names():
            print(
                f"ERROR: Unknown dataset '{args.dataset}'. "
                f"Available: {', '.join(DatasetRegistry.all_names())}",
                file=sys.stderr,
            )
            return 1
        datasets = [args.dataset]
        sample_limit = args.limit
        logger.info("Mode: SINGLE dataset=%s", args.dataset)
    else:
        # Default: quick mode
        datasets = QUICK_DATASETS
        sample_limit = args.limit or 200
        logger.info("Mode: QUICK (default, use --full or --dataset for more)")

    # Credentials
    license_key = os.environ.get("ETHICORE_API_KEY")
    assets_dir = os.environ.get("ETHICORE_ASSETS_DIR")

    if args.tier == "licensed" and not license_key:
        logger.warning(
            "--tier licensed requested but ETHICORE_API_KEY not set. "
            "Falling back to community tier."
        )

    # Pre-load all datasets BEFORE asyncio.run() to avoid HuggingFace's
    # dill-based fingerprint hasher encountering asyncio RLock objects.
    preloaded = preload_datasets(datasets, sample_limit)

    # Run
    try:
        run_data = asyncio.run(
            run_benchmark(
                datasets=datasets,
                tier=args.tier,
                sample_limit=sample_limit,
                output_dir=args.output,
                preloaded_samples=preloaded,
                license_key=license_key,
                assets_dir=assets_dir,
                enable_learning=args.learn,
            )
        )
        return 0 if run_data else 1

    except KeyboardInterrupt:
        logger.info("Benchmark interrupted by user.")
        return 130
    except Exception as exc:  # noqa: BLE001
        logger.error("Benchmark failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
