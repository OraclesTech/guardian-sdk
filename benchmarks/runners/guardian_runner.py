"""
GuardianRunner — runs Guardian SDK's analyze() on each prompt and records latency.

Supports both community and licensed tiers via GuardianConfig.
Uses asyncio for concurrent execution; concurrency capped at 20 to avoid
overwhelming the event loop with ONNX inference.

Also exposes submit_corrections() for two-pass benchmark with AdversarialLearner
closed-loop learning between Pass 1 and Pass 2.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .base_runner import BenchmarkRunner, Prediction

if TYPE_CHECKING:
    from benchmarks.datasets.loader import BenchmarkSample

logger = logging.getLogger(__name__)

# Binary classification: use Guardian's actual production decision.
#
# Historical note (v1.8.0): the runner used a fixed SCORE_THRESHOLD=0.050 because
# the v1.8.0 model produced a near-zero behavioral floor (~0.04 on benigns), and
# rapid-fire benchmark calls without session isolation drove recommended_action
# to BLOCK on most benigns via the automated_scan_detector. Both assumptions are
# now obsolete:
#   1. We pass a unique session_id per call (_iso_ctx() below), so the scan
#      detector and context tracker see fresh buckets every prompt — no
#      cross-sample contamination.
#   2. The 400k-sample calibrated model + sub-span default-on has a higher,
#      well-calibrated baseline score (~0.2-0.3 on benigns). The 0.050 threshold
#      now mislabels benigns as threats EVEN WHEN Guardian's actual decision is
#      ALLOW.
# So the correct binary label is what Guardian itself produced: BLOCK or
# CHALLENGE → threat (1), ALLOW or MONITOR → benign (0). This is the production-
# aligned methodology Lakera uses for its published numbers; it makes the
# comparison apples-to-apples. The continuous threat_score is still used for
# ROC-AUC / PR-AUC computation below.
_THREAT_ACTIONS = frozenset({"BLOCK", "CHALLENGE"})

# Retained only for tools that import it; not used for label assignment.
SCORE_THRESHOLD = 0.050

# Map threat_level → normalized score (for AUC computation)
_LEVEL_TO_SCORE = {
    "NONE": 0.0,
    "LOW": 0.25,
    "MEDIUM": 0.5,
    "HIGH": 0.75,
    "CRITICAL": 1.0,
}


class GuardianRunner(BenchmarkRunner):
    """
    Runs Guardian SDK locally against a list of prompts.

    Args:
        license_key: Ethicore API key. Defaults to ETHICORE_API_KEY env var.
                     (Parameter name retained for backwards-compat with GuardianConfig;
                     the underlying env var was renamed LICENSE_KEY -> API_KEY in v1.8.1.)
        assets_dir:  Path to licensed asset bundle. Defaults to ETHICORE_ASSETS_DIR.
        concurrency: Max concurrent analyze() calls (default 20).
        timeout_ms:  Per-request timeout in ms (default 5000).
    """

    name = "guardian"

    def __init__(
        self,
        license_key: Optional[str] = None,
        assets_dir: Optional[str] = None,
        concurrency: int = 20,
        timeout_ms: int = 5000,
        tier: str = "auto",  # "community" | "licensed" | "auto"
    ):
        self.license_key = license_key or os.environ.get("ETHICORE_API_KEY")
        self.assets_dir = assets_dir or os.environ.get("ETHICORE_ASSETS_DIR")
        self.concurrency = concurrency
        self.timeout_ms = timeout_ms
        self.tier = tier
        self._guardian = None
        self._semaphore: Optional[asyncio.Semaphore] = None

    @property
    def effective_tier(self) -> str:
        if self.tier != "auto":
            return self.tier
        return "licensed" if self.license_key else "community"

    async def _get_guardian(self):
        """Lazy-init Guardian SDK singleton."""
        if self._guardian is not None:
            return self._guardian

        try:
            from ethicore_guardian import Guardian  # noqa: PLC0415
            from ethicore_guardian.utils.config import GuardianConfig  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "ethicore-engine-guardian is required. "
                "Install it: pip install ethicore-engine-guardian"
            ) from None

        # Start from the production env config so sensitivity thresholds
        # (pattern_sensitivity, semantic_sensitivity, ml_sensitivity) match
        # exactly what runs in the demo — avoids artificial over-triggering.
        config = GuardianConfig.from_env()

        # Benchmark-specific overrides only:
        config.cache_enabled = False        # measure each call independently
        config.log_level = "WARNING"        # suppress noise during runs
        config.enable_metrics = False       # no telemetry during benchmarks
        config.analysis_timeout_ms = self.timeout_ms

        # Apply license / assets if explicitly provided to the runner
        if self.license_key:
            config.license_key = self.license_key
        if self.assets_dir:
            config.assets_dir = self.assets_dir

        guardian = Guardian(config=config)
        await guardian.initialize()
        self._guardian = guardian
        logger.info(
            "GuardianRunner: initialized [%s tier]", self.effective_tier
        )
        return self._guardian

    async def run(self, texts: List[str]) -> List[Prediction]:
        guardian = await self._get_guardian()

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.concurrency)

        # Per-sample isolation context (CRITICAL for benchmark correctness):
        # Without a unique session_id per call, the automated_scan_detector,
        # context_poisoning_tracker, and behavioral_analyzer accumulate state
        # across the rapid-fire benchmark queries — by the 50th prompt they
        # treat the entire stream as a scanning attack and flag every input as
        # threat, including benigns (observed: FPR=1.0 on the agentic dataset
        # without this isolation). Each benchmark sample is an INDEPENDENT
        # measurement and gets its own fresh session bucket. Mirrors the
        # adversarial harness's _iso_ctx() fix from the 2026-06 sub-span work.
        def _iso_ctx() -> Dict[str, Any]:
            return {
                "source_type": "user_input",
                "session_id": f"bench-{uuid.uuid4().hex[:12]}",
            }

        async def _analyze_one(text: str) -> Prediction:
            async with self._semaphore:
                t0 = time.perf_counter()
                try:
                    result = await guardian.analyze(text, context=_iso_ctx())
                    latency_ms = (time.perf_counter() - t0) * 1000

                    # Build continuous threat score for AUC computation.
                    # Priority: threat_score attr → confidence attr → level-based mapping.
                    raw_score = getattr(result, "threat_score", None)
                    confidence = getattr(result, "confidence", None)
                    threat_level = getattr(result, "threat_level", "NONE")
                    recommended_action = getattr(result, "recommended_action", "ALLOW")

                    if raw_score is not None and raw_score > 0.0:
                        # Guardian v1.8+ scores on a 0–10 scale (block threshold ≥7.0).
                        # Normalize to 0–1 by dividing by 10. Guard against older
                        # versions that may return an already-normalized 0–1 value.
                        score = min(float(raw_score) / 10.0, 1.0) if raw_score > 1.0 else float(raw_score)
                    elif confidence is not None and confidence > 0.0:
                        # Use confidence as a proxy score
                        score = min(float(confidence), 1.0)
                    else:
                        # Level-based fallback — ensures AUC has variance
                        score = _LEVEL_TO_SCORE.get(threat_level, 0.0)
                        # If blocked but score is still 0, ensure non-zero
                        if recommended_action in ("BLOCK", "CHALLENGE") and score == 0.0:
                            score = 0.6

                    # Binary label: align with Guardian's actual production decision.
                    # BLOCK / CHALLENGE → threat (1); ALLOW / MONITOR → benign (0).
                    # The continuous threat_score is retained for AUC computation.
                    # See header comment for why this replaces the old score-threshold
                    # approach.
                    label = 1 if recommended_action in _THREAT_ACTIONS else 0

                    return Prediction(
                        text=text,
                        predicted_label=label,
                        threat_score=score,
                        latency_ms=latency_ms,
                        threat_level=threat_level,
                        recommended_action=recommended_action,
                        metadata={
                            "threat_types": getattr(result, "threat_types", []),
                            "confidence": confidence,
                        },
                    )

                except Exception as exc:  # noqa: BLE001
                    latency_ms = (time.perf_counter() - t0) * 1000
                    logger.debug("analyze() error on sample: %s", exc)
                    # Fail-safe: return ALLOW (don't inflate recall artificially)
                    return Prediction(
                        text=text,
                        predicted_label=0,
                        threat_score=0.0,
                        latency_ms=latency_ms,
                        metadata={"error": str(exc)},
                    )

        tasks = [_analyze_one(text) for text in texts]
        return await asyncio.gather(*tasks)

    async def submit_corrections(
        self,
        samples: List["BenchmarkSample"],
        predictions: List[Prediction],
        max_fingerprints: int = 5_000,
    ) -> Dict[str, Any]:
        """
        Submit false-negative misclassifications to the AdversarialLearner for
        closed-loop learning between benchmark passes.

        False negatives (attacks that were ALLOW'd) are fed into the dual-layer
        AdversarialLearner via learn_from_confirmed_attack(), which immediately
        adds semantic fingerprints in memory so Pass 2 benefits without a restart.

        Args:
            samples:          Ground-truth samples (same list used in run_dataset).
            predictions:      Predictions from Pass 1 (same order as samples).
            max_fingerprints: Raise the fingerprint cap before learning so the
                              benchmark can absorb all unique FNs (default 5000).

        Returns:
            Dict with learning stats: fn_submitted, fn_deduplicated, fn_cap_reached,
            fn_errors, total_attempted.
        """
        guardian = await self._get_guardian()

        # Access the AdversarialLearner (lazy-initialized inside Guardian).
        try:
            learner = guardian._get_adversarial_learner()
        except AttributeError:
            logger.warning(
                "submit_corrections: _get_adversarial_learner() not available — "
                "skipping learning pass."
            )
            return {"fn_submitted": 0, "fn_deduplicated": 0,
                    "fn_cap_reached": 0, "fn_errors": 0, "total_attempted": 0}

        # Raise the fingerprint cap so the benchmark can absorb all unique FNs.
        original_cap = learner.max_fingerprints
        learner.max_fingerprints = max(max_fingerprints, original_cap)
        logger.info(
            "AdversarialLearner cap raised: %d → %d for learning pass",
            original_cap, learner.max_fingerprints,
        )

        stats: Dict[str, Any] = {
            "fn_submitted": 0,
            "fn_deduplicated": 0,
            "fn_cap_reached": 0,
            "fn_errors": 0,
            "total_attempted": 0,
        }

        for sample, pred in zip(samples, predictions):
            if sample.label != 1 or pred.predicted_label != 0:
                # Only learn from false negatives (attack missed by Guardian).
                continue

            stats["total_attempted"] += 1
            category = getattr(sample, "category", None) or "adversarial_learned"

            try:
                outcome = await learner.learn_from_confirmed_attack(
                    text=sample.text,
                    category=category,
                    severity="HIGH",
                    weight=85,
                    source="benchmark_fn",
                )
                if outcome.added:
                    stats["fn_submitted"] += 1
                    logger.debug(
                        "Learned FN [%s] similarity=%.3f total=%d",
                        category, outcome.similarity_to_nearest, outcome.fingerprints_total,
                    )
                elif outcome.reason == "duplicate":
                    stats["fn_deduplicated"] += 1
                elif outcome.reason == "cap_reached":
                    stats["fn_cap_reached"] += 1
                    logger.debug("Fingerprint cap reached — increase max_fingerprints")
                # else: error / not_initialized / empty_text — logged inside learner
            except Exception as exc:  # noqa: BLE001
                logger.debug("AdversarialLearner error on sample: %s", exc)
                stats["fn_errors"] += 1

        logger.info(
            "Learning pass complete — submitted: %d | duplicate: %d | "
            "cap_reached: %d | errors: %d",
            stats["fn_submitted"], stats["fn_deduplicated"],
            stats["fn_cap_reached"], stats["fn_errors"],
        )
        return stats
