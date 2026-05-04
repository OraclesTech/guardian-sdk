"""
Ethicore Engine™ - Guardian SDK - Community Edition
Pure-Python threat detection — no external ML dependencies.

Covers 5 OWASP LLM Top-10 aligned categories:
  - instructionOverride
  - jailbreakActivation
  - safetyBypass
  - roleHijacking
  - systemPromptLeaks

To unlock 50+ categories, full semantic embeddings, agentic pipeline
protection, and multi-turn analysis, supply an ETHICORE_API_KEY:

    guardian = Guardian(api_key="eg-sk-...")
    # or: export ETHICORE_API_KEY=eg-sk-...

API contract: identical to the API tier — same class names, same method
signatures. Code written against the community edition works unchanged
when credentials are supplied.

Copyright © 2026 Oracles Technologies LLC. All Rights Reserved.
Framework: MIT License. Full threat library: Proprietary.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .data.threat_patterns import (
    THREAT_PATTERNS,
    ThreatSeverity,
    get_all_patterns,
    get_semantic_fingerprints,
    calculate_threat_score,
    determine_threat_level,
)

logger = logging.getLogger(__name__)

_UPGRADE_NOTE = (
    "Upgrade to the API tier for 50+ threat categories, full semantic "
    "embeddings, agentic pipeline protection, and multi-turn analysis. "
    "Set ETHICORE_API_KEY or pass api_key= to Guardian()."
)

# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------

@dataclass
class GuardianConfig:
    """Configuration for a Guardian instance (community edition)."""
    api_key: Optional[str] = None
    block_threshold: float = 50.0   # threat score at which to block
    log_threats: bool = True
    timeout: float = 5.0            # reserved for API-tier async calls


@dataclass
class ThreatAnalysis:
    """Result of a threat analysis scan."""
    is_threat: bool
    threat_level: str                       # NONE / LOW / MEDIUM / HIGH / CRITICAL
    threat_score: float
    categories: List[str]
    action: str                             # ALLOW / BLOCK / CHALLENGE
    matched_patterns: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Convenience
    @property
    def blocked(self) -> bool:
        return self.action == "BLOCK"


@dataclass
class OutputAnalysisResult:
    """Result of an output / response analysis scan."""
    is_threat: bool
    threat_level: str
    threat_score: float
    categories: List[str]
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallScanResult:
    """Result of a tool-call scan (community: always ALLOW with upgrade note)."""
    allowed: bool
    action: str
    reason: str
    categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolOutputScanResult:
    """Result of a tool-output scan (community: always ALLOW with upgrade note)."""
    allowed: bool
    action: str
    reason: str
    categories: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class ThreatChallengeException(Exception):
    """Raised when a request is blocked by Guardian."""

    def __init__(
        self,
        message: str = "Request blocked by Ethicore Guardian",
        threat_analysis: Optional[ThreatAnalysis] = None,
    ) -> None:
        super().__init__(message)
        self.threat_analysis = threat_analysis


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _HashBasedLearner:
    """
    Closed-loop adversarial learner — community edition.

    Stores SHA-256 hashes of confirmed threat texts so that exact (or
    near-exact) repetitions are flagged immediately on the next scan,
    without requiring an LLM round-trip.
    """

    def __init__(self) -> None:
        self._hashes: set[str] = set()

    @staticmethod
    def _normalise(text: str) -> str:
        """Lower-case, collapse whitespace."""
        return re.sub(r"\s+", " ", text.strip().lower())

    def _digest(self, text: str) -> str:
        return hashlib.sha256(self._normalise(text).encode()).hexdigest()

    def record_threat(self, text: str) -> None:
        """Record a confirmed threat so future identical inputs are caught."""
        self._digest(text)  # validate it hashes cleanly
        self._hashes.add(self._digest(text))

    def is_known_threat(self, text: str) -> bool:
        """Return True if *text* exactly matches a previously recorded threat."""
        return self._digest(text) in self._hashes

    @property
    def known_count(self) -> int:
        return len(self._hashes)


class _CommunityDetector:
    """
    Two-layer threat detector — community edition.

    Layer 1: compiled regex patterns from ``data/threat_patterns.py``
    Layer 2: word-overlap fingerprint matching on semantic fingerprints
             (all-but-one unique words must appear in the input)
    """

    _MITIGATOR_ROLES = frozenset(
        ["tutor", "teacher", "instructor", "guide", "mentor", "educator", "professor"]
    )

    def __init__(self) -> None:
        # Compile all patterns once at init time
        self._compiled: List[Dict[str, Any]] = []
        for entry in get_all_patterns():
            try:
                self._compiled.append({
                    **entry,
                    "_re": re.compile(entry["pattern"], re.IGNORECASE),
                })
            except re.error as exc:
                logger.warning("Pattern compile error (%s): %s", entry["pattern"], exc)

        # Build fingerprint word-sets for fast overlap matching
        self._fingerprints: List[Dict[str, Any]] = []
        for fp in get_semantic_fingerprints():
            words = set(re.split(r"\W+", fp["text"].lower())) - {""}
            self._fingerprints.append({**fp, "_words": words})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, text: str) -> List[Dict[str, Any]]:
        """
        Run both layers and return a list of match dicts.

        Each match contains: ``category``, ``layer``, ``pattern``/``fingerprint``,
        ``severity``, ``weight``, ``count``.
        """
        matches: List[Dict[str, Any]] = []
        matches.extend(self._layer1_regex(text))
        matches.extend(self._layer2_fingerprint(text))
        return matches

    # ------------------------------------------------------------------
    # Layer 1 — regex
    # ------------------------------------------------------------------

    def _layer1_regex(self, text: str) -> List[Dict[str, Any]]:
        found: Dict[str, Dict[str, Any]] = {}
        for entry in self._compiled:
            cat = entry["category"]
            hits = entry["_re"].findall(text)
            if not hits:
                continue
            if cat in found:
                found[cat]["count"] += len(hits)
            else:
                found[cat] = {
                    "category": cat,
                    "layer": "regex",
                    "pattern": entry["pattern"],
                    "severity": entry["severity"],
                    "weight": entry["weight"],
                    "count": len(hits),
                }
        # Apply mitigators for roleHijacking
        if "roleHijacking" in found:
            text_lower = text.lower()
            if any(role in text_lower for role in self._MITIGATOR_ROLES):
                del found["roleHijacking"]
        return list(found.values())

    # ------------------------------------------------------------------
    # Layer 2 — word-overlap fingerprints
    # ------------------------------------------------------------------

    def _layer2_fingerprint(self, text: str) -> List[Dict[str, Any]]:
        text_words = set(re.split(r"\W+", text.lower())) - {""}
        found: Dict[str, Dict[str, Any]] = {}
        for fp in self._fingerprints:
            fp_words = fp["_words"]
            if len(fp_words) < 2:
                continue
            # Require all-but-one words to be present
            threshold = max(1, len(fp_words) - 1)
            overlap = len(fp_words & text_words)
            if overlap < threshold:
                continue
            cat = fp["category"]
            if cat not in found:
                found[cat] = {
                    "category": cat,
                    "layer": "fingerprint",
                    "fingerprint": fp["text"],
                    "severity": fp["severity"],
                    "weight": fp["weight"],
                    "count": 1,
                }
        return list(found.values())


# ---------------------------------------------------------------------------
# Guardian — community edition
# ---------------------------------------------------------------------------

class Guardian:
    """
    Ethicore Engine™ Guardian — Community Edition.

    Drop-in replacement for the API-tier Guardian with the same public
    interface. Covers 5 threat categories using pure-Python detection
    (regex + hash-based fingerprints). No external ML dependencies.

    Usage::

        from ethicore_guardian.community_guardian import Guardian
        guardian = Guardian()
        result = guardian.analyze("Ignore all previous instructions")
        if result.blocked:
            raise ValueError("Threat detected")

    For full 50+ category coverage pass an ``api_key`` (or set the
    ``ETHICORE_API_KEY`` environment variable) and use the main
    ``Guardian`` class from ``ethicore_guardian``.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[GuardianConfig] = None,
    ) -> None:
        self.config = config or GuardianConfig(
            api_key=api_key or os.environ.get("ETHICORE_API_KEY")
        )
        self._detector = _CommunityDetector()
        self._learner = _HashBasedLearner()
        logger.debug("Guardian (community edition) initialised — 5 categories active")

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyze(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> ThreatAnalysis:
        """
        Synchronously analyse *text* for threats.

        Returns a :class:`ThreatAnalysis` instance.  Call ``result.blocked``
        to check whether to reject the input.
        """
        if not isinstance(text, str) or not text.strip():
            return ThreatAnalysis(
                is_threat=False,
                threat_level="NONE",
                threat_score=0.0,
                categories=[],
                action="ALLOW",
                matched_patterns=[],
                metadata={"reason": "empty_input"},
            )

        # Layer 0 — previously recorded known threats
        if self._learner.is_known_threat(text):
            return ThreatAnalysis(
                is_threat=True,
                threat_level="CRITICAL",
                threat_score=200.0,
                categories=["knownThreat"],
                action="BLOCK",
                matched_patterns=[{"category": "knownThreat", "layer": "learner"}],
                metadata={"source": "adversarial_learner"},
            )

        matches = self._detector.scan(text)
        categories = list({m["category"] for m in matches})

        match_summaries = [
            {"category": m["category"], "count": m["count"]} for m in matches
        ]
        score = calculate_threat_score(match_summaries)
        level = determine_threat_level(score)
        is_threat = score >= self.config.block_threshold
        action = "BLOCK" if is_threat else "ALLOW"

        if is_threat and self.config.log_threats:
            logger.warning(
                "Guardian blocked: score=%.1f level=%s categories=%s",
                score, level, categories,
            )

        return ThreatAnalysis(
            is_threat=is_threat,
            threat_level=level,
            threat_score=score,
            categories=categories,
            action=action,
            matched_patterns=matches,
            metadata={"edition": "community", **(metadata or {})},
        )

    async def analyze_async(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ThreatAnalysis:
        """Async wrapper — runs synchronous detection in the default executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze, text, metadata)

    # ------------------------------------------------------------------
    # HTML analysis
    # ------------------------------------------------------------------

    def analyze_html(
        self, html: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ThreatAnalysis:
        """
        Analyse HTML content by stripping tags and running standard detection.

        Community edition does not include full DOM-level browser analysis
        (visual steganography, hidden form field scanning, JS exfiltration
        detection). Upgrade to the API tier for Layer 13 browser protection.
        """
        # Strip HTML tags with a simple regex pass
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"&[a-zA-Z]{2,6};", " ", text)   # basic entity decode
        text = re.sub(r"\s+", " ", text).strip()
        result = self.analyze(text, metadata)
        result.metadata["source"] = "html_stripped"
        result.metadata["_community_note"] = (
            "Full DOM/browser analysis requires API tier. " + _UPGRADE_NOTE
        )
        return result

    async def analyze_html_async(
        self, html: str, metadata: Optional[Dict[str, Any]] = None
    ) -> ThreatAnalysis:
        """Async wrapper for :meth:`analyze_html`."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.analyze_html, html, metadata)

    # ------------------------------------------------------------------
    # Output / response analysis
    # ------------------------------------------------------------------

    def analyze_response(
        self,
        response: str,
        original_prompt: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OutputAnalysisResult:
        """
        Analyse an AI response for threats.

        Community edition scans response text with the same 5-category
        detector.  The API tier adds output-specific patterns (data
        exfiltration, PII leakage, prompt reflection detection).
        """
        result = self.analyze(response, metadata)
        return OutputAnalysisResult(
            is_threat=result.is_threat,
            threat_level=result.threat_level,
            threat_score=result.threat_score,
            categories=result.categories,
            action=result.action,
            metadata={**result.metadata, "source": "response_analysis"},
        )

    # ------------------------------------------------------------------
    # Agentic pipeline — community stubs (permissive)
    # ------------------------------------------------------------------

    def scan_tool_call(
        self,
        tool_name: str,
        tool_input: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolCallScanResult:
        """
        Scan a tool call before execution.

        Community edition always returns ALLOW with an upgrade notice.
        Agentic pipeline protection (tool provenance, indirect injection
        in tool inputs, multi-step attack chains) requires the API tier.
        """
        return ToolCallScanResult(
            allowed=True,
            action="ALLOW",
            reason="Community edition: agentic protection requires API tier.",
            metadata={"upgrade_note": _UPGRADE_NOTE},
        )

    def scan_tool_output(
        self,
        tool_name: str,
        tool_output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolOutputScanResult:
        """
        Scan tool output for injection payloads.

        Community edition always returns ALLOW with an upgrade notice.
        Full tool output scanning (HTML, JSON, indirect prompt injection)
        requires the API tier.
        """
        return ToolOutputScanResult(
            allowed=True,
            action="ALLOW",
            reason="Community edition: tool output scanning requires API tier.",
            metadata={"upgrade_note": _UPGRADE_NOTE},
        )

    async def scan_tool_call_async(
        self,
        tool_name: str,
        tool_input: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolCallScanResult:
        return self.scan_tool_call(tool_name, tool_input, metadata)

    async def scan_tool_output_async(
        self,
        tool_name: str,
        tool_output: Any,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ToolOutputScanResult:
        return self.scan_tool_output(tool_name, tool_output, metadata)

    # ------------------------------------------------------------------
    # Adversarial learning
    # ------------------------------------------------------------------

    def record_confirmed_threat(self, text: str) -> None:
        """
        Record a confirmed threat text so identical future inputs are
        caught immediately (hash-based learner).
        """
        self._learner.record_threat(text)

    # ------------------------------------------------------------------
    # Provider wrapping — thin wrappers that call analyze() pre-request
    # ------------------------------------------------------------------

    def wrap(self, client: Any) -> Any:
        """
        Wrap an AI provider client so every request is scanned first.

        Community edition supports OpenAI, xAI/Grok, and Azure OpenAI
        clients (anything with a ``.chat.completions.create`` interface).
        """
        return _WrappedClient(client, self)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Guardian(edition='community', categories={len(THREAT_PATTERNS)}, "
            f"known_threats={self._learner.known_count})"
        )


# ---------------------------------------------------------------------------
# Minimal provider wrapper
# ---------------------------------------------------------------------------

class _WrappedClient:
    """
    Transparent proxy that runs Guardian.analyze() before every chat
    completion request and raises ThreatChallengeException if blocked.
    """

    def __init__(self, client: Any, guardian: Guardian) -> None:
        self._client = client
        self._guardian = guardian
        self.chat = _WrappedChat(client, guardian)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


class _WrappedChat:
    def __init__(self, client: Any, guardian: Guardian) -> None:
        self._client = client
        self._guardian = guardian
        self.completions = _WrappedCompletions(client, guardian)


class _WrappedCompletions:
    def __init__(self, client: Any, guardian: Guardian) -> None:
        self._client = client
        self._guardian = guardian

    def create(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                result = self._guardian.analyze(content)
                if result.blocked:
                    raise ThreatChallengeException(
                        f"Guardian blocked request: {result.threat_level} "
                        f"threat detected in message ({', '.join(result.categories)})",
                        threat_analysis=result,
                    )
        return self._client.chat.completions.create(**kwargs)

    async def acreate(self, **kwargs: Any) -> Any:
        messages = kwargs.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                result = await self._guardian.analyze_async(content)
                if result.blocked:
                    raise ThreatChallengeException(
                        f"Guardian blocked request: {result.threat_level} "
                        f"threat detected in message ({', '.join(result.categories)})",
                        threat_analysis=result,
                    )
        return await self._client.chat.completions.acreate(**kwargs)


# ---------------------------------------------------------------------------
# Convenience functions (mirror public API tier signatures)
# ---------------------------------------------------------------------------

def analyze_text(
    text: str,
    api_key: Optional[str] = None,
    config: Optional[GuardianConfig] = None,
) -> ThreatAnalysis:
    """One-shot text analysis — creates a Guardian, scans, returns result."""
    return Guardian(api_key=api_key, config=config).analyze(text)


def protect_openai(client: Any, api_key: Optional[str] = None) -> Any:
    """Wrap an OpenAI client with Guardian protection."""
    return Guardian(api_key=api_key).wrap(client)


def protect_xai(client: Any, api_key: Optional[str] = None) -> Any:
    """Wrap an xAI / Grok client with Guardian protection."""
    return Guardian(api_key=api_key).wrap(client)


def protect_azure(client: Any, api_key: Optional[str] = None) -> Any:
    """Wrap an Azure OpenAI client with Guardian protection."""
    return Guardian(api_key=api_key).wrap(client)


def protect_gemini(client: Any, api_key: Optional[str] = None) -> Any:
    """
    Community stub for Gemini protection.

    Full Gemini integration (streaming, function-calling, multi-turn) is
    available in the API tier.
    """
    logger.warning(
        "protect_gemini: community edition wraps text content only. "
        + _UPGRADE_NOTE
    )
    return Guardian(api_key=api_key).wrap(client)


def protect_bedrock(client: Any, api_key: Optional[str] = None) -> Any:
    """
    Community stub for AWS Bedrock protection.

    Full Bedrock integration requires the API tier.
    """
    logger.warning(
        "protect_bedrock: community edition wraps text content only. "
        + _UPGRADE_NOTE
    )
    return Guardian(api_key=api_key).wrap(client)


def protect_litellm(client: Any, api_key: Optional[str] = None) -> Any:
    """
    Community stub for LiteLLM protection.

    Full LiteLLM integration requires the API tier.
    """
    logger.warning(
        "protect_litellm: community edition wraps text content only. "
        + _UPGRADE_NOTE
    )
    return Guardian(api_key=api_key).wrap(client)
