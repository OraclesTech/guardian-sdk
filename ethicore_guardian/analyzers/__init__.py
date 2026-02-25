"""
Ethicore Engine™ - Guardian SDK — Analysers package

Exports all eight detection-layer classes so integrators can import directly
from ``ethicore_guardian.analyzers`` without knowing the internal module layout.

Each analyser is imported defensively so a missing optional dependency (e.g.
``onnxruntime`` not installed) prevents only that analyser from loading, not
the entire package.

Phase 2 additions (pure-Python, no optional dependencies):
  - IndirectInjectionAnalyzer : source-type-aware external content scanning
  - ContextPoisoningTracker   : multi-turn conversation trajectory analysis
  - AutomatedScanDetector     : trigram similarity + jailbreak template rotation
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# PatternAnalyzer — always available (pure-Python, no optional deps)
# ---------------------------------------------------------------------------
from ethicore_guardian.analyzers.pattern_analyzer import PatternAnalyzer  # noqa: E402

# ---------------------------------------------------------------------------
# SemanticAnalyzer — requires onnxruntime; falls back gracefully when absent
# ---------------------------------------------------------------------------
try:
    from ethicore_guardian.analyzers.semantic_analyzer import SemanticAnalyzer
except ImportError as _e:
    logger.warning("SemanticAnalyzer unavailable: %s", _e)
    SemanticAnalyzer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# BehavioralAnalyzer — pure-Python, always available
# ---------------------------------------------------------------------------
try:
    from ethicore_guardian.analyzers.behavioral_analyzer import BehavioralAnalyzer
except ImportError as _e:
    logger.warning("BehavioralAnalyzer unavailable: %s", _e)
    BehavioralAnalyzer = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# MLInferenceEngine — optional transformers/torch; heuristic fallback built in
# ---------------------------------------------------------------------------
try:
    from ethicore_guardian.analyzers.ml_inference_engine import MLInferenceEngine
except ImportError as _e:
    logger.warning("MLInferenceEngine unavailable: %s", _e)
    MLInferenceEngine = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# ThreatDetector — the multi-layer orchestrator
# ---------------------------------------------------------------------------
try:
    from ethicore_guardian.analyzers.threat_detector import ThreatDetector
except ImportError as _e:
    logger.warning("ThreatDetector unavailable: %s", _e)
    ThreatDetector = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# IndirectInjectionAnalyzer — Phase 2, Layer 5 (pure-Python, no optional deps)
# ---------------------------------------------------------------------------
try:
    from ethicore_guardian.analyzers.indirect_injection_analyzer import (
        IndirectInjectionAnalyzer,
        SourceType,
    )
except ImportError as _e:
    logger.warning("IndirectInjectionAnalyzer unavailable: %s", _e)
    IndirectInjectionAnalyzer = None  # type: ignore[assignment,misc]
    SourceType = None                 # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# ContextPoisoningTracker — Phase 2, Layer 6 (pure-Python, no optional deps)
# ---------------------------------------------------------------------------
try:
    from ethicore_guardian.analyzers.context_tracker import ContextPoisoningTracker
except ImportError as _e:
    logger.warning("ContextPoisoningTracker unavailable: %s", _e)
    ContextPoisoningTracker = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# AutomatedScanDetector — Phase 2, Layer 7 (pure-Python, no optional deps)
# ---------------------------------------------------------------------------
try:
    from ethicore_guardian.analyzers.automated_scan_detector import AutomatedScanDetector
except ImportError as _e:
    logger.warning("AutomatedScanDetector unavailable: %s", _e)
    AutomatedScanDetector = None  # type: ignore[assignment,misc]

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
__all__ = [
    # Original layers
    "PatternAnalyzer",
    "SemanticAnalyzer",
    "BehavioralAnalyzer",
    "MLInferenceEngine",
    "ThreatDetector",
    # Phase 2 additions
    "IndirectInjectionAnalyzer",
    "SourceType",
    "ContextPoisoningTracker",
    "AutomatedScanDetector",
]
